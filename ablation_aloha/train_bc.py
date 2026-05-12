#!/usr/bin/env python3
"""Train a small image+state behavior cloning policy on LeRobot ALOHA data."""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


@dataclass
class DatasetConfig:
    root: str
    image_key: str = "observation.images.top"
    state_key: str = "observation.state"
    action_key: str = "action"
    val_fraction: float = 0.2
    frame_stride: int = 1
    max_episodes: int | None = None


@dataclass
class TrainConfig:
    run_name: str
    dataset: DatasetConfig
    checkpoint_dir: str
    image_size: int = 128
    batch_size: int = 64
    epochs: int = 25
    lr: float = 3e-4
    weight_decay: float = 1e-5
    num_workers: int = 4
    seed: int = 0
    device: str = "auto"
    log_every: int = 50
    sample_limit: int | None = None
    tensorboard: bool = True
    tensorboard_dir: str | None = None


class RunningNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)

    @classmethod
    def from_arrays(cls, values: np.ndarray) -> "RunningNormalizer":
        return cls(values.mean(axis=0), values.std(axis=0))

    @classmethod
    def identity(cls, dim: int) -> "RunningNormalizer":
        return cls(np.zeros(dim, dtype=np.float32), np.ones(dim, dtype=np.float32))

    def normalize_np(self, value: np.ndarray) -> np.ndarray:
        return (value.astype(np.float32) - self.mean) / self.std

    def denormalize_torch(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=value.device, dtype=value.dtype)
        std = torch.as_tensor(self.std, device=value.device, dtype=value.dtype)
        return value * std + mean

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def load(cls, payload: dict[str, Any]) -> "RunningNormalizer":
        return cls(np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["std"], dtype=np.float32))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    dataset = DatasetConfig(**raw["dataset"])
    return TrainConfig(dataset=dataset, **{k: v for k, v in raw.items() if k != "dataset"})


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def lerobot_data_files(root: Path) -> list[Path]:
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
    return files


def vector_from_cell(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def decode_image_cell(value: Any) -> Image.Image:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path") is not None:
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    raise TypeError(f"Unsupported image cell type: {type(value)!r}")


class LeRobotFrameDataset(Dataset):
    def __init__(
        self,
        cfg: DatasetConfig,
        split: str,
        image_size: int,
        seed: int,
        state_normalizer: RunningNormalizer | None = None,
        action_normalizer: RunningNormalizer | None = None,
        sample_limit: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.root = Path(cfg.root)
        self.split = split
        self.image_size = image_size
        self.files = lerobot_data_files(self.root)
        self.columns = [cfg.image_key, cfg.state_key, cfg.action_key, "episode_index", "frame_index"]
        self._cache: OrderedDict[Path, pd.DataFrame] = OrderedDict()
        self._cache_size = 2

        self.index = self._build_index(seed)
        if sample_limit is not None:
            self.index = self.index[:sample_limit]
        if not self.index:
            raise ValueError(f"{split} split is empty for {self.root}")

        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer

    def _build_index(self, seed: int) -> list[tuple[Path, int]]:
        frames: list[tuple[Path, int, int, int]] = []
        episodes: set[int] = set()
        for path in self.files:
            df = pd.read_parquet(path, columns=["episode_index", "frame_index"])
            for row_idx, row in enumerate(df.itertuples(index=False)):
                episode = int(row.episode_index)
                frame = int(row.frame_index)
                episodes.add(episode)
                if frame % self.cfg.frame_stride == 0:
                    frames.append((path, row_idx, episode, frame))

        episode_list = sorted(episodes)
        rng = random.Random(seed)
        rng.shuffle(episode_list)
        if self.cfg.max_episodes is not None:
            episode_list = episode_list[: self.cfg.max_episodes]

        val_count = max(1, int(round(len(episode_list) * self.cfg.val_fraction))) if len(episode_list) > 1 else 0
        val_eps = set(sorted(episode_list[-val_count:])) if val_count else set()
        train_eps = set(episode_list) - val_eps
        selected = val_eps if self.split == "val" else train_eps
        return [(path, row_idx) for path, row_idx, episode, _ in frames if episode in selected]

    def _read_file(self, path: Path) -> pd.DataFrame:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        df = pd.read_parquet(path, columns=self.columns)
        self._cache[path] = df
        self._cache.move_to_end(path)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return df

    def collect_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        by_file: dict[Path, list[int]] = {}
        for path, row_idx in self.index:
            by_file.setdefault(path, []).append(row_idx)
        for path, row_indices in by_file.items():
            df = pd.read_parquet(path, columns=[self.cfg.state_key, self.cfg.action_key])
            rows = df.iloc[row_indices]
            states.extend(vector_from_cell(v) for v in rows[self.cfg.state_key].to_list())
            actions.extend(vector_from_cell(v) for v in rows[self.cfg.action_key].to_list())
        return np.stack(states), np.stack(actions)

    def set_normalizers(self, state_normalizer: RunningNormalizer, action_normalizer: RunningNormalizer) -> None:
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path, row_idx = self.index[idx]
        row = self._read_file(path).iloc[row_idx]

        image = decode_image_cell(row[self.cfg.image_key])
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        state = vector_from_cell(row[self.cfg.state_key])
        action = vector_from_cell(row[self.cfg.action_key])
        if self.state_normalizer is not None:
            state = self.state_normalizer.normalize_np(state)
        if self.action_normalizer is not None:
            action = self.action_normalizer.normalize_np(action)

        return {
            "image": image_tensor,
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action),
        }


class ImageStateBC(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        image_feat = self.image_encoder(image)
        state_feat = self.state_encoder(state)
        return self.head(torch.cat([image_feat, state_feat], dim=-1))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    action_normalizer: RunningNormalizer | None = None,
) -> dict[str, float | list[float]]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_raw_mse = 0.0
    total_raw_mae = 0.0
    total_count = 0
    per_dim_abs_error = None
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        pred = model(image, state)
        batch_size = image.shape[0]
        total_mse += F.mse_loss(pred, action, reduction="mean").item() * batch_size
        total_mae += F.l1_loss(pred, action, reduction="mean").item() * batch_size

        if action_normalizer is not None:
            pred_raw = action_normalizer.denormalize_torch(pred)
            action_raw = action_normalizer.denormalize_torch(action)
            abs_error = torch.abs(pred_raw - action_raw)
            total_raw_mse += F.mse_loss(pred_raw, action_raw, reduction="mean").item() * batch_size
            total_raw_mae += abs_error.mean().item() * batch_size
            dim_error = abs_error.sum(dim=0).detach().cpu()
            per_dim_abs_error = dim_error if per_dim_abs_error is None else per_dim_abs_error + dim_error

        total_count += batch_size

    metrics: dict[str, float | list[float]] = {
        "mse": total_mse / total_count,
        "mae": total_mae / total_count,
    }
    if action_normalizer is not None and per_dim_abs_error is not None:
        metrics.update(
            {
                "raw_mse": total_raw_mse / total_count,
                "raw_rmse": (total_raw_mse / total_count) ** 0.5,
                "raw_mae": total_raw_mae / total_count,
                "raw_mae_per_action_dim": (per_dim_abs_error / total_count).tolist(),
            }
        )
    return metrics


def make_summary_writer(cfg: TrainConfig, checkpoint_dir: Path) -> Any:
    if not cfg.tensorboard:
        return None
    if SummaryWriter is None:
        print("TensorBoard is not installed; continuing without TensorBoard logging.")
        return None
    log_dir = Path(cfg.tensorboard_dir) if cfg.tensorboard_dir else checkpoint_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("config/yaml", f"```yaml\n{yaml.safe_dump(cfg_to_dict(cfg), sort_keys=False)}\n```", 0)
    print(f"TensorBoard log dir: {log_dir}")
    return writer


def log_epoch_to_tensorboard(writer: Any, epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, Any]) -> None:
    if writer is None:
        return
    writer.add_scalar("objective/train_mse_epoch", train_metrics["mse"], epoch)
    writer.add_scalar("objective/val_mse_epoch", val_metrics["mse"], epoch)
    writer.add_scalar("objective/val_mae_epoch", val_metrics["mae"], epoch)
    if "raw_mse" in val_metrics:
        writer.add_scalar("raw_action/val_mse", val_metrics["raw_mse"], epoch)
        writer.add_scalar("raw_action/val_rmse", val_metrics["raw_rmse"], epoch)
        writer.add_scalar("raw_action/val_mae", val_metrics["raw_mae"], epoch)
    for dim, value in enumerate(val_metrics.get("raw_mae_per_action_dim", [])):
        writer.add_scalar(f"raw_action_dim/val_mae_dim_{dim:02d}", value, epoch)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    state_normalizer: RunningNormalizer,
    action_normalizer: RunningNormalizer,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg_to_dict(cfg),
            "state_normalizer": state_normalizer.state_dict(),
            "action_normalizer": action_normalizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def cfg_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    payload = cfg.__dict__.copy()
    payload["dataset"] = cfg.dataset.__dict__.copy()
    return payload


def train(cfg: TrainConfig) -> Path:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_ds = LeRobotFrameDataset(cfg.dataset, "train", cfg.image_size, cfg.seed, sample_limit=cfg.sample_limit)
    val_ds = LeRobotFrameDataset(cfg.dataset, "val", cfg.image_size, cfg.seed, sample_limit=cfg.sample_limit)

    print(f"Dataset: {cfg.dataset.root}")
    print(f"Frames: train={len(train_ds)} val={len(val_ds)}")
    states, actions = train_ds.collect_vectors()
    state_normalizer = RunningNormalizer.from_arrays(states)
    action_normalizer = RunningNormalizer.from_arrays(actions)
    train_ds.set_normalizers(state_normalizer, action_normalizer)
    val_ds.set_normalizers(state_normalizer, action_normalizer)

    state_dim = states.shape[-1]
    action_dim = actions.shape[-1]
    model = ImageStateBC(state_dim=state_dim, action_dim=action_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )

    best_val = math.inf
    log_path = checkpoint_dir / "train_log.jsonl"
    writer = make_summary_writer(cfg, checkpoint_dir)
    global_step = 0
    with open(log_path, "a", encoding="utf-8") as log_f:
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running = 0.0
            seen = 0
            pbar = tqdm(train_loader, desc=f"{cfg.run_name} epoch {epoch}/{cfg.epochs}")
            for step, batch in enumerate(pbar, start=1):
                image = batch["image"].to(device, non_blocking=True)
                state = batch["state"].to(device, non_blocking=True)
                action = batch["action"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    pred = model(image, state)
                    loss = F.mse_loss(pred, action)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()

                batch_size = image.shape[0]
                running += loss.item() * batch_size
                seen += batch_size
                global_step += 1
                if writer is not None:
                    writer.add_scalar("objective/train_mse_step", loss.item(), global_step)
                    writer.add_scalar("optimizer/lr", optimizer.param_groups[0]["lr"], global_step)
                if step % cfg.log_every == 0:
                    pbar.set_postfix(train_mse=running / seen)

            train_metrics = {"mse": running / seen}
            val_metrics = evaluate(model, val_loader, device, action_normalizer)
            record = {
                "time": time.time(),
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "run_name": cfg.run_name,
            }
            print(json.dumps(record, indent=2))
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()
            log_epoch_to_tensorboard(writer, epoch, train_metrics, val_metrics)
            if writer is not None:
                writer.flush()

            save_checkpoint(
                checkpoint_dir / "last.pt",
                model,
                optimizer,
                cfg,
                state_normalizer,
                action_normalizer,
                epoch,
                val_metrics,
            )
            if val_metrics["mse"] < best_val:
                best_val = val_metrics["mse"]
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    cfg,
                    state_normalizer,
                    action_normalizer,
                    epoch,
                    val_metrics,
                )

    if writer is not None:
        writer.close()

    return checkpoint_dir / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--dataset-root", help="Override dataset.root.")
    parser.add_argument("--checkpoint-dir", help="Override checkpoint_dir.")
    parser.add_argument("--epochs", type=int, help="Override epochs.")
    parser.add_argument("--batch-size", type=int, help="Override batch_size.")
    parser.add_argument("--num-workers", type=int, help="Override num_workers.")
    parser.add_argument("--sample-limit", type=int, help="Use a small number of frames for a smoke test.")
    parser.add_argument("--device", help="Override device, e.g. cuda:0 or cpu.")
    parser.add_argument("--tensorboard-dir", help="Override TensorBoard log directory.")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.dataset.root = args.dataset_root
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.sample_limit is not None:
        cfg.sample_limit = args.sample_limit
    if args.device:
        cfg.device = args.device
    if args.tensorboard_dir:
        cfg.tensorboard_dir = args.tensorboard_dir
    if args.no_tensorboard:
        cfg.tensorboard = False

    best_path = train(cfg)
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
