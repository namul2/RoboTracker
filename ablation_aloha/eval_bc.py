#!/usr/bin/env python3
"""Evaluate one or more ALOHA behavior cloning checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_bc import ImageStateBC, LeRobotFrameDataset, RunningNormalizer, DatasetConfig, resolve_device


def load_policy(checkpoint_path: Path, device: torch.device) -> tuple[ImageStateBC, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    state_norm = RunningNormalizer.load(ckpt["state_normalizer"])
    action_norm = RunningNormalizer.load(ckpt["action_normalizer"])
    state_dim = len(state_norm.mean)
    action_dim = len(action_norm.mean)
    model = ImageStateBC(state_dim=state_dim, action_dim=action_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, {**ckpt, "state_norm": state_norm, "action_norm": action_norm, "config": cfg}


@torch.no_grad()
def eval_offline(
    checkpoint_path: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    dataset_root_override: str | None,
    sample_limit: int | None,
) -> dict[str, Any]:
    model, payload = load_policy(checkpoint_path, device)
    cfg = payload["config"]
    dataset_cfg = DatasetConfig(**cfg["dataset"])
    if dataset_root_override:
        dataset_cfg.root = dataset_root_override

    dataset = LeRobotFrameDataset(
        dataset_cfg,
        split=split,
        image_size=int(cfg["image_size"]),
        seed=int(cfg["seed"]),
        state_normalizer=payload["state_norm"],
        action_normalizer=payload["action_norm"],
        sample_limit=sample_limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    norm_mse = 0.0
    norm_mae = 0.0
    raw_mse = 0.0
    raw_mae = 0.0
    count = 0
    per_dim_abs: list[np.ndarray] = []

    for batch in tqdm(loader, desc=f"eval {checkpoint_path.parent.name}:{split}"):
        image = batch["image"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        pred = model(image, state)

        pred_raw = payload["action_norm"].denormalize_torch(pred)
        action_raw = payload["action_norm"].denormalize_torch(action)
        batch_size_actual = image.shape[0]
        norm_mse += F.mse_loss(pred, action, reduction="mean").item() * batch_size_actual
        norm_mae += F.l1_loss(pred, action, reduction="mean").item() * batch_size_actual
        raw_mse += F.mse_loss(pred_raw, action_raw, reduction="mean").item() * batch_size_actual
        raw_mae += F.l1_loss(pred_raw, action_raw, reduction="mean").item() * batch_size_actual
        per_dim_abs.append(torch.abs(pred_raw - action_raw).mean(dim=0).cpu().numpy())
        count += batch_size_actual

    per_dim = np.stack(per_dim_abs).mean(axis=0)
    return {
        "checkpoint": str(checkpoint_path),
        "run_name": cfg["run_name"],
        "split": split,
        "frames": len(dataset),
        "norm_mse": norm_mse / count,
        "norm_mae": norm_mae / count,
        "raw_mse": raw_mse / count,
        "raw_rmse": (raw_mse / count) ** 0.5,
        "raw_mae": raw_mae / count,
        "raw_mae_per_action_dim": per_dim.tolist(),
        "saved_epoch": payload.get("epoch"),
        "saved_metrics": payload.get("metrics", {}),
    }


def print_table(results: list[dict[str, Any]]) -> None:
    headers = ["run", "split", "frames", "raw_mae", "raw_rmse", "norm_mse", "epoch"]
    rows = []
    for result in results:
        rows.append(
            [
                result["run_name"],
                result["split"],
                str(result["frames"]),
                f"{result['raw_mae']:.6f}",
                f"{result['raw_rmse']:.6f}",
                f"{result['norm_mse']:.6f}",
                str(result["saved_epoch"]),
            ]
        )
    widths = [max(len(str(x)) for x in [header] + [row[i] for row in rows]) for i, header in enumerate(headers)]
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint paths, e.g. checkpoints/human/best.pt")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-root", help="Override dataset root for all checkpoints.")
    parser.add_argument("--sample-limit", type=int, help="Evaluate a subset of frames for a smoke test.")
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    results = [
        eval_offline(
            Path(path),
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            dataset_root_override=args.dataset_root,
            sample_limit=args.sample_limit,
        )
        for path in args.checkpoints
    ]
    print_table(results)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
