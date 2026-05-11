#!/usr/bin/env python3
"""Train/test out-of-distribution analyzer for LeRobot datasets.

This script compares a train distribution and a test distribution from either:

1. A random episode split inside each dataset, or
2. Two explicitly provided LeRobot dataset roots.

The comparison is intentionally multi-signal rather than a single hard label:
- trajectory flow and speed
- initial robot pose
- initial camera pose distribution
- first-frame image embedding distribution, when cached embeddings are available
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing base dependencies. Run: python3 -m pip install -r requirements.txt"
    ) from exc

from image_clustering import resolve_camera_poses
from visualize import (
    DatasetConfig,
    compute_episode_metrics,
    discover_dataset_roots,
    list_data_files,
    load_positions_by_episode,
    resolve_dataset_config,
)


@dataclass(frozen=True)
class Split:
    train: Tuple[int, ...]
    test: Tuple[int, ...]


@dataclass
class DistributionGroup:
    name: str
    episode_ids: Tuple[int, ...]
    positions: np.ndarray
    progress: np.ndarray
    starts: np.ndarray
    changes: np.ndarray
    speeds: np.ndarray
    descriptors: np.ndarray
    descriptor_names: Tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare train/test OOD signals for LeRobot robot datasets."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="LeRobot dataset root, or parent directory with --all-datasets.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Analyze every child LeRobot dataset under --dataset-root.",
    )
    parser.add_argument(
        "--dataset-name",
        action="append",
        default=None,
        help="Dataset child name to include when --dataset-root is a parent. Repeatable.",
    )
    parser.add_argument(
        "--train-dataset-root",
        type=Path,
        default=None,
        help="Optional explicit train dataset root. Requires --test-dataset-root.",
    )
    parser.add_argument(
        "--test-dataset-root",
        type=Path,
        default=None,
        help="Optional explicit test dataset root. Requires --train-dataset-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ood_detection"),
        help="Directory for OOD reports and plots.",
    )
    parser.add_argument(
        "--image-output-dir",
        type=Path,
        default=Path("outputs/image_distribution"),
        help="Directory containing image_clustering.py outputs.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train episode ratio for random in-dataset splits.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on episodes before random splitting.",
    )
    parser.add_argument(
        "--sample-points",
        type=int,
        default=50000,
        help="Max trajectory points per train/test group in scatter plots.",
    )
    parser.add_argument("--bins", type=int, default=80, help="Histogram bin count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--state-key", type=str, default=None, help="Override state key.")
    parser.add_argument("--episode-key", type=str, default=None, help="Override episode key.")
    parser.add_argument("--timestamp-key", type=str, default=None, help="Override timestamp key.")
    parser.add_argument(
        "--ee-indices",
        type=int,
        nargs="+",
        default=None,
        help="Override coordinate indices used for robot position.",
    )
    parser.add_argument(
        "--gripper-index",
        type=int,
        default=None,
        help="Override gripper index.",
    )
    parser.add_argument(
        "--gripper-change-threshold",
        type=float,
        default=1e-6,
        help="Threshold for detecting first gripper change.",
    )
    return parser.parse_args()


def _as_config_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        state_key=args.state_key,
        episode_key=args.episode_key,
        timestamp_key=args.timestamp_key,
        ee_indices=args.ee_indices,
        gripper_index=args.gripper_index,
    )


def _select_episode_ids(
    episode_ids: Sequence[int],
    max_episodes: Optional[int],
    seed: int,
) -> np.ndarray:
    ids = np.asarray(sorted(set(int(ep) for ep in episode_ids)), dtype=np.int64)
    if max_episodes is not None and ids.size > max_episodes:
        rng = np.random.default_rng(seed)
        ids = np.sort(rng.choice(ids, size=max_episodes, replace=False))
    return ids


def _random_split(episode_ids: Sequence[int], train_ratio: float, seed: int) -> Split:
    ids = np.asarray(sorted(set(int(ep) for ep in episode_ids)), dtype=np.int64)
    if ids.size < 2:
        raise ValueError("Need at least two episodes for a train/test split.")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    n_train = int(round(ids.size * train_ratio))
    n_train = max(1, min(n_train, ids.size - 1))
    train = tuple(sorted(int(ep) for ep in shuffled[:n_train]))
    test = tuple(sorted(int(ep) for ep in shuffled[n_train:]))
    return Split(train=train, test=test)


def _subset_episode_data(
    episode_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    episode_ids: Iterable[int],
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    wanted = set(int(ep) for ep in episode_ids)
    return {ep: value for ep, value in episode_data.items() if int(ep) in wanted}


def _episode_descriptors(
    episode_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    rows: List[List[float]] = []
    coord_dim = 0
    for _, (ts, pos, _) in sorted(episode_data.items()):
        if ts.size == 0 or pos.size == 0:
            continue
        order = np.argsort(ts)
        ts_s = ts[order]
        pos_s = pos[order]
        coord_dim = pos_s.shape[1]
        diffs = np.diff(pos_s, axis=0)
        step_dist = np.linalg.norm(diffs, axis=1) if diffs.size else np.empty(0)
        duration = max(float(ts_s[-1] - ts_s[0]), 0.0)
        path_length = float(np.sum(step_dist)) if step_dist.size else 0.0
        mean_speed = path_length / duration if duration > 1e-9 else 0.0
        max_step = float(np.max(step_dist)) if step_dist.size else 0.0
        row = [
            *pos_s[0].tolist(),
            *pos_s[-1].tolist(),
            *(pos_s[-1] - pos_s[0]).tolist(),
            path_length,
            duration,
            mean_speed,
            max_step,
        ]
        rows.append(row)

    axis = list("XYZ")[:coord_dim]
    names = (
        tuple(f"start_{label}" for label in axis)
        + tuple(f"end_{label}" for label in axis)
        + tuple(f"delta_{label}" for label in axis)
        + ("path_length", "duration", "mean_speed", "max_step")
    )
    if not rows:
        return np.empty((0, len(names)), dtype=np.float64), names
    return np.asarray(rows, dtype=np.float64), names


def _make_group(
    name: str,
    episode_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    episode_ids: Sequence[int],
    gripper_change_threshold: float,
) -> DistributionGroup:
    subset = _subset_episode_data(episode_data, episode_ids)
    positions, progress, starts, changes, speeds = compute_episode_metrics(
        subset, gripper_change_threshold
    )
    descriptors, descriptor_names = _episode_descriptors(subset)
    return DistributionGroup(
        name=name,
        episode_ids=tuple(sorted(int(ep) for ep in subset)),
        positions=positions,
        progress=progress,
        starts=starts,
        changes=changes,
        speeds=speeds,
        descriptors=descriptors,
        descriptor_names=descriptor_names,
    )


def _safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_percentile(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def _axis_summary(arr: np.ndarray) -> Dict[str, Any]:
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.shape[0]),
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
        "p05": np.percentile(arr, 5, axis=0).tolist(),
        "median": np.percentile(arr, 50, axis=0).tolist(),
        "p95": np.percentile(arr, 95, axis=0).tolist(),
    }


def _scalar_summary(arr: np.ndarray) -> Dict[str, Any]:
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _standardized_mean_shift(train: np.ndarray, test: np.ndarray) -> float:
    if train.size == 0 or test.size == 0:
        return float("nan")
    train_2d = np.atleast_2d(train)
    test_2d = np.atleast_2d(test)
    std = np.std(train_2d, axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    diff = (np.mean(test_2d, axis=0) - np.mean(train_2d, axis=0)) / std
    return float(np.sqrt(np.mean(np.square(diff))))


def _nearest_distances(query: np.ndarray, ref: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    if query.size == 0 or ref.size == 0:
        return np.empty((0,), dtype=np.float64)
    query_2d = np.atleast_2d(query).astype(np.float64)
    ref_2d = np.atleast_2d(ref).astype(np.float64)
    out = np.empty((query_2d.shape[0],), dtype=np.float64)
    ref_norm = np.sum(ref_2d * ref_2d, axis=1)
    for start in range(0, query_2d.shape[0], chunk_size):
        block = query_2d[start : start + chunk_size]
        d2 = np.sum(block * block, axis=1, keepdims=True) + ref_norm[None, :]
        d2 -= 2.0 * (block @ ref_2d.T)
        out[start : start + block.shape[0]] = np.sqrt(np.maximum(np.min(d2, axis=1), 0.0))
    return out


def _standardized_nn_report(train: np.ndarray, test: np.ndarray) -> Dict[str, float]:
    if train.size == 0 or test.size == 0:
        return {}
    train_2d = np.atleast_2d(train).astype(np.float64)
    test_2d = np.atleast_2d(test).astype(np.float64)
    center = np.mean(train_2d, axis=0)
    scale = np.std(train_2d, axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    train_z = (train_2d - center) / scale
    test_z = (test_2d - center) / scale

    test_nn = _nearest_distances(test_z, train_z)
    if train_z.shape[0] >= 2:
        train_loo = np.empty((train_z.shape[0],), dtype=np.float64)
        for i in range(train_z.shape[0]):
            ref = np.concatenate([train_z[:i], train_z[i + 1 :]], axis=0)
            train_loo[i] = _nearest_distances(train_z[i : i + 1], ref)[0]
    else:
        train_loo = np.empty((0,), dtype=np.float64)

    threshold = _safe_percentile(train_loo, 95)
    ratio = float(np.percentile(test_nn, 95) / threshold) if threshold > 1e-9 else float("nan")
    return {
        "test_nn_mean": _safe_mean(test_nn),
        "test_nn_p95": _safe_percentile(test_nn, 95),
        "test_nn_max": float(np.max(test_nn)) if test_nn.size else float("nan"),
        "train_leave_one_out_nn_p95": threshold,
        "test_p95_to_train_p95_ratio": ratio,
    }


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norm, 1e-9)


def _cosine_nn_report(train: np.ndarray, test: np.ndarray) -> Dict[str, float]:
    if train.size == 0 or test.size == 0:
        return {}
    train_n = _normalize_rows(np.atleast_2d(train).astype(np.float64))
    test_n = _normalize_rows(np.atleast_2d(test).astype(np.float64))
    sims = test_n @ train_n.T
    test_dist = 1.0 - np.max(sims, axis=1)
    if train_n.shape[0] >= 2:
        train_sims = train_n @ train_n.T
        np.fill_diagonal(train_sims, -np.inf)
        train_dist = 1.0 - np.max(train_sims, axis=1)
    else:
        train_dist = np.empty((0,), dtype=np.float64)
    threshold = _safe_percentile(train_dist, 95)
    ratio = float(np.percentile(test_dist, 95) / threshold) if threshold > 1e-9 else float("nan")
    return {
        "test_cosine_nn_mean": _safe_mean(test_dist),
        "test_cosine_nn_p95": _safe_percentile(test_dist, 95),
        "test_cosine_nn_max": float(np.max(test_dist)) if test_dist.size else float("nan"),
        "train_leave_one_out_cosine_nn_p95": threshold,
        "test_p95_to_train_p95_ratio": ratio,
    }


def _downsample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def _ensure_3d(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    pts = np.atleast_2d(points)
    if pts.shape[1] >= 3:
        return pts[:, :3]
    z = np.zeros((pts.shape[0], 3 - pts.shape[1]), dtype=pts.dtype)
    return np.concatenate([pts, z], axis=1)


def _set_equal_3d_limits(ax: Any, points: np.ndarray) -> None:
    if points.size == 0:
        return
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    centers = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1e-6)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_trajectory_split(
    train: DistributionGroup,
    test: DistributionGroup,
    coord_labels: Sequence[str],
    sample_points: int,
    seed: int,
    out_path: Path,
) -> None:
    train_idx = _downsample_indices(train.positions.shape[0], sample_points, seed)
    test_idx = _downsample_indices(test.positions.shape[0], sample_points, seed + 17)
    train_pts = train.positions[train_idx]
    test_pts = test.positions[test_idx]

    if train.positions.shape[1] >= 3:
        fig = plt.figure(figsize=(8.5, 7.2))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            train_pts[:, 0], train_pts[:, 1], train_pts[:, 2],
            c="#3b82f6", s=2.0, alpha=0.35, linewidths=0, label="train trajectory",
        )
        ax.scatter(
            test_pts[:, 0], test_pts[:, 1], test_pts[:, 2],
            c="#f97316", s=2.8, alpha=0.55, linewidths=0, label="test trajectory",
        )
        if train.starts.size:
            ax.scatter(
                train.starts[:, 0], train.starts[:, 1], train.starts[:, 2],
                c="#1d4ed8", marker="^", s=30, alpha=0.9, label="train starts",
            )
        if test.starts.size:
            ax.scatter(
                test.starts[:, 0], test.starts[:, 1], test.starts[:, 2],
                c="#c2410c", marker="^", s=38, alpha=0.95, label="test starts",
            )
        ax.set_xlabel(coord_labels[0])
        ax.set_ylabel(coord_labels[1])
        ax.set_zlabel(coord_labels[2])
        all_pts = np.vstack([_ensure_3d(train_pts), _ensure_3d(test_pts)])
        _set_equal_3d_limits(ax, all_pts)
    else:
        fig, ax = plt.subplots(figsize=(7.4, 6.4))
        ax.scatter(
            train_pts[:, 0], train_pts[:, 1],
            c="#3b82f6", s=3.0, alpha=0.35, linewidths=0, label="train trajectory",
        )
        ax.scatter(
            test_pts[:, 0], test_pts[:, 1],
            c="#f97316", s=4.0, alpha=0.55, linewidths=0, label="test trajectory",
        )
        if train.starts.size:
            ax.scatter(
                train.starts[:, 0], train.starts[:, 1],
                c="#1d4ed8", marker="^", s=30, alpha=0.9, label="train starts",
            )
        if test.starts.size:
            ax.scatter(
                test.starts[:, 0], test.starts[:, 1],
                c="#c2410c", marker="^", s=38, alpha=0.95, label="test starts",
            )
        ax.set_xlabel(coord_labels[0])
        ax.set_ylabel(coord_labels[1])
        ax.grid(True, color="#dddddd", alpha=0.55)
    ax.set_title("Train vs Test Trajectory Distribution")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_speed_distribution(
    train_speeds: np.ndarray,
    test_speeds: np.ndarray,
    bins: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    if train_speeds.size:
        ax.hist(train_speeds, bins=bins, density=True, alpha=0.52, color="#3b82f6", label="train")
    if test_speeds.size:
        ax.hist(test_speeds, bins=bins, density=True, alpha=0.52, color="#f97316", label="test")
    ax.set_title("Train vs Test Speed Distribution")
    ax.set_xlabel("Speed (units/s)")
    ax.set_ylabel("Density")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_initial_environment_3d(
    train_starts: np.ndarray,
    test_starts: np.ndarray,
    train_camera_pos: np.ndarray,
    test_camera_pos: np.ndarray,
    coord_labels: Sequence[str],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(8.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    train_starts_3d = _ensure_3d(train_starts)
    test_starts_3d = _ensure_3d(test_starts)

    if train_starts_3d.size:
        ax.scatter(
            train_starts_3d[:, 0], train_starts_3d[:, 1], train_starts_3d[:, 2],
            c="#2563eb", marker="^", s=36, alpha=0.75, label="train initial robot pose",
        )
    if test_starts_3d.size:
        ax.scatter(
            test_starts_3d[:, 0], test_starts_3d[:, 1], test_starts_3d[:, 2],
            c="#ea580c", marker="^", s=44, alpha=0.9, label="test initial robot pose",
        )
    if train_camera_pos.size:
        ax.scatter(
            train_camera_pos[:, 0], train_camera_pos[:, 1], train_camera_pos[:, 2],
            c="#0891b2", marker="s", s=55, alpha=0.45, label="train camera pose",
        )
    if test_camera_pos.size:
        ax.scatter(
            test_camera_pos[:, 0], test_camera_pos[:, 1], test_camera_pos[:, 2],
            c="#dc2626", marker="s", s=70, alpha=0.7, label="test camera pose",
        )

    origin = np.asarray([[0.0, 0.0, 0.0]])
    ax.scatter(origin[:, 0], origin[:, 1], origin[:, 2], c="#111111", s=60, label="robot base")
    all_points = [origin, train_starts_3d, test_starts_3d]
    if train_camera_pos.size:
        all_points.append(train_camera_pos)
    if test_camera_pos.size:
        all_points.append(test_camera_pos)
    _set_equal_3d_limits(ax, np.vstack([p for p in all_points if p.size]))

    labels = list(coord_labels) + ["Z"]
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title("Initial Robot Pose and Camera Pose Map")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_embedding_split(
    coords_2d: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    if np.any(train_mask):
        ax.scatter(
            coords_2d[train_mask, 0], coords_2d[train_mask, 1],
            c="#3b82f6", marker="o", s=28, alpha=0.58, edgecolors="none", label="train",
        )
    if np.any(test_mask):
        ax.scatter(
            coords_2d[test_mask, 0], coords_2d[test_mask, 1],
            c="#f97316", marker="D", s=34, alpha=0.78, edgecolors="white",
            linewidths=0.35, label="test",
        )
    ax.set_title("Train vs Test First-Frame Image Embeddings")
    ax.set_xlabel("Embedding dim 1")
    ax.set_ylabel("Embedding dim 2")
    ax.grid(True, color="#dddddd", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_shift_bar(metrics: Dict[str, float], out_path: Path) -> None:
    clean = {k: v for k, v in metrics.items() if v is not None and np.isfinite(v)}
    if not clean:
        return
    labels = list(clean)
    values = [clean[label] for label in labels]
    fig, ax = plt.subplots(figsize=(max(7.5, 0.72 * len(labels)), 5.0))
    colors = ["#3b82f6" if value < 1.0 else "#f97316" if value < 2.0 else "#dc2626" for value in values]
    ax.bar(labels, values, color=colors, alpha=0.86)
    ax.axhline(1.0, color="#555555", linewidth=1.0, linestyle="--", alpha=0.65)
    ax.set_title("OOD Signal Strength")
    ax.set_ylabel("Standardized shift / distance ratio")
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(axis="y", color="#dddddd", alpha=0.45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_mean_comparison_bar(
    train_values: Dict[str, float],
    test_values: Dict[str, float],
    train_scales: Dict[str, float],
    out_path: Path,
) -> None:
    labels = [
        label
        for label in train_values
        if label in test_values
        and np.isfinite(train_values[label])
        and np.isfinite(test_values[label])
    ]
    if not labels:
        return
    train_norm = []
    test_norm = []
    for label in labels:
        scale = train_scales.get(label, 1.0)
        if not np.isfinite(scale) or abs(scale) < 1e-9:
            scale = 1.0
        train_norm.append(0.0)
        test_norm.append((test_values[label] - train_values[label]) / scale)

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8.0, 0.66 * len(labels)), 5.2))
    ax.bar(x - width / 2, train_norm, width=width, color="#3b82f6", alpha=0.8, label="train mean")
    ax.bar(x + width / 2, test_norm, width=width, color="#f97316", alpha=0.86, label="test mean")
    ax.axhline(0.0, color="#555555", linewidth=1.0)
    ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--", alpha=0.55)
    ax.axhline(-1.0, color="#777777", linewidth=0.8, linestyle="--", alpha=0.55)
    ax.set_title("Test Mean Relative to Train Mean and Std")
    ax.set_ylabel("(mean - train_mean) / train_std")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(loc="best")
    ax.grid(axis="y", color="#dddddd", alpha=0.45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _load_dataset_trajectory(
    dataset_root: Path,
    args: argparse.Namespace,
) -> Tuple[DatasetConfig, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    parquet_files = list_data_files(dataset_root)
    config = resolve_dataset_config(dataset_root, parquet_files, _as_config_args(args))
    episode_data = load_positions_by_episode(
        parquet_files,
        state_key=config.state_key,
        episode_key=config.episode_key,
        timestamp_key=config.timestamp_key,
        coord_indices=config.coord_indices,
        gripper_index=config.gripper_index,
    )
    return config, episode_data


def _embedding_path(image_output_dir: Path, dataset_name: str) -> Path:
    return image_output_dir / dataset_name / "embeddings.npz"


def _load_embedding_split(
    image_output_dir: Path,
    dataset_name: str,
    split: Split,
) -> Optional[Dict[str, Any]]:
    path = _embedding_path(image_output_dir, dataset_name)
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=True)
    episodes = np.asarray(data["episode_index"], dtype=np.int64)
    train_set = set(split.train)
    test_set = set(split.test)
    train_mask = np.asarray([int(ep) in train_set for ep in episodes], dtype=bool)
    test_mask = np.asarray([int(ep) in test_set for ep in episodes], dtype=bool)
    return {
        "path": path,
        "embeddings": np.asarray(data["embeddings"]),
        "coords_2d": np.asarray(data["coords_2d"]),
        "image_key": np.asarray(data["image_key"]).astype(str),
        "episode_index": episodes,
        "train_mask": train_mask,
        "test_mask": test_mask,
    }


def _camera_positions_from_embeddings(embedding_data: Optional[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    if embedding_data is None:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    image_keys = embedding_data["image_key"]
    unique_keys = sorted(set(str(key) for key in image_keys.tolist()))
    poses = resolve_camera_poses(unique_keys, {})
    pose_by_key = {pose.key: pose.position for pose in poses}
    positions = np.asarray([pose_by_key[str(key)] for key in image_keys], dtype=np.float64)
    return positions[embedding_data["train_mask"]], positions[embedding_data["test_mask"]]


def _camera_key_counts(embedding_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if embedding_data is None:
        return {}
    counts: Dict[str, Dict[str, int]] = {}
    for split_name, mask in (("train", embedding_data["train_mask"]), ("test", embedding_data["test_mask"])):
        keys, key_counts = np.unique(embedding_data["image_key"][mask], return_counts=True)
        counts[split_name] = {str(key): int(count) for key, count in zip(keys, key_counts)}
    return counts


def _collect_mean_bar_values(
    train: DistributionGroup,
    test: DistributionGroup,
    embedding_data: Optional[Dict[str, Any]],
    train_camera: np.ndarray,
    test_camera: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    train_values: Dict[str, float] = {}
    test_values: Dict[str, float] = {}
    scales: Dict[str, float] = {}

    axis = list("XYZ")[: train.positions.shape[1]]
    for idx, label in enumerate(axis):
        key = f"traj_{label.lower()}"
        train_values[key] = float(np.mean(train.positions[:, idx]))
        test_values[key] = float(np.mean(test.positions[:, idx]))
        scales[key] = float(np.std(train.positions[:, idx]))
        start_key = f"start_{label.lower()}"
        train_values[start_key] = float(np.mean(train.starts[:, idx]))
        test_values[start_key] = float(np.mean(test.starts[:, idx]))
        scales[start_key] = float(np.std(train.starts[:, idx]))

    train_values["speed"] = _safe_mean(train.speeds)
    test_values["speed"] = _safe_mean(test.speeds)
    scales["speed"] = float(np.std(train.speeds)) if train.speeds.size else 1.0

    for descriptor_name in ("path_length", "duration", "mean_speed"):
        if descriptor_name in train.descriptor_names:
            idx = train.descriptor_names.index(descriptor_name)
            train_values[descriptor_name] = float(np.mean(train.descriptors[:, idx]))
            test_values[descriptor_name] = float(np.mean(test.descriptors[:, idx]))
            scales[descriptor_name] = float(np.std(train.descriptors[:, idx]))

    if embedding_data is not None:
        coords = embedding_data["coords_2d"]
        train_mask = embedding_data["train_mask"]
        test_mask = embedding_data["test_mask"]
        if np.any(train_mask) and np.any(test_mask):
            for idx, label in enumerate(("embed_u", "embed_v")):
                train_values[label] = float(np.mean(coords[train_mask, idx]))
                test_values[label] = float(np.mean(coords[test_mask, idx]))
                scales[label] = float(np.std(coords[train_mask, idx]))

    if train_camera.size and test_camera.size:
        for idx, label in enumerate(("cam_x", "cam_y", "cam_z")):
            train_values[label] = float(np.mean(train_camera[:, idx]))
            test_values[label] = float(np.mean(test_camera[:, idx]))
            scales[label] = float(np.std(train_camera[:, idx]))

    return train_values, test_values, scales


def _severity_from_shift(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < 0.5:
        return "low"
    if value < 1.5:
        return "moderate"
    return "high"


def _write_report_text(summary: Dict[str, Any], out_path: Path) -> None:
    lines = [
        f"OOD report: {summary['name']}",
        "",
        f"Train episodes: {summary['split']['train_episode_count']}",
        f"Test episodes: {summary['split']['test_episode_count']}",
        "",
        "Signal shifts:",
    ]
    for key, value in summary.get("shift_metrics", {}).items():
        if value is None or not np.isfinite(value):
            lines.append(f"- {key}: n/a")
        else:
            lines.append(f"- {key}: {value:.4f} ({_severity_from_shift(float(value))})")

    lines.extend(["", "Trajectory nearest-neighbor report:"])
    for key, value in summary.get("trajectory", {}).get("descriptor_nn", {}).items():
        lines.append(f"- {key}: {value:.4f}" if np.isfinite(value) else f"- {key}: n/a")

    if summary.get("image_embeddings", {}).get("available"):
        lines.extend(["", "Image embedding nearest-neighbor report:"])
        for key, value in summary["image_embeddings"]["cosine_nn"].items():
            lines.append(f"- {key}: {value:.6f}" if np.isfinite(value) else f"- {key}: n/a")
    else:
        lines.extend(["", "Image embeddings: not available"])

    lines.extend(["", "Plots:"])
    for label, path in summary.get("outputs", {}).items():
        lines.append(f"- {label}: {path}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_random_split_dataset(
    dataset_root: Path,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    config, episode_data = _load_dataset_trajectory(dataset_root, args)
    available_ids = _select_episode_ids(episode_data.keys(), args.max_episodes, args.seed)
    split = _random_split(available_ids, args.train_ratio, args.seed)
    train = _make_group("train", episode_data, split.train, args.gripper_change_threshold)
    test = _make_group("test", episode_data, split.test, args.gripper_change_threshold)

    dataset_output = output_dir / dataset_root.name
    dataset_output.mkdir(parents=True, exist_ok=True)

    embedding_data = _load_embedding_split(args.image_output_dir.resolve(), dataset_root.name, split)
    train_camera, test_camera = _camera_positions_from_embeddings(embedding_data)

    trajectory_plot = dataset_output / "trajectory_train_test_3d.png"
    if train.positions.shape[1] < 3:
        trajectory_plot = dataset_output / "trajectory_train_test_2d.png"
    plot_trajectory_split(
        train, test, config.coord_labels, args.sample_points, args.seed, trajectory_plot
    )
    plot_speed_distribution(train.speeds, test.speeds, args.bins, dataset_output / "speed_train_test.png")
    plot_initial_environment_3d(
        train.starts,
        test.starts,
        train_camera,
        test_camera,
        config.coord_labels,
        dataset_output / "initial_environment_3d.png",
    )

    image_summary: Dict[str, Any] = {"available": False}
    embedding_plot: Optional[Path] = None
    if embedding_data is not None and np.any(embedding_data["train_mask"]) and np.any(embedding_data["test_mask"]):
        embedding_plot = dataset_output / "image_embedding_train_test.png"
        plot_embedding_split(
            embedding_data["coords_2d"],
            embedding_data["train_mask"],
            embedding_data["test_mask"],
            embedding_plot,
        )
        train_embed = embedding_data["embeddings"][embedding_data["train_mask"]]
        test_embed = embedding_data["embeddings"][embedding_data["test_mask"]]
        image_summary = {
            "available": True,
            "embedding_path": str(embedding_data["path"]),
            "train_sample_count": int(train_embed.shape[0]),
            "test_sample_count": int(test_embed.shape[0]),
            "mean_cosine_shift": float(
                1.0
                - np.dot(
                    _normalize_rows(np.mean(train_embed, axis=0, keepdims=True))[0],
                    _normalize_rows(np.mean(test_embed, axis=0, keepdims=True))[0],
                )
            ),
            "cosine_nn": _cosine_nn_report(train_embed, test_embed),
        }

    camera_summary = {
        "available": bool(train_camera.size and test_camera.size),
        "train": _axis_summary(train_camera),
        "test": _axis_summary(test_camera),
        "mean_shift_z": _standardized_mean_shift(train_camera, test_camera)
        if train_camera.size and test_camera.size else float("nan"),
        "image_key_counts": _camera_key_counts(embedding_data),
    }

    trajectory_summary = {
        "position_mean_shift_z": _standardized_mean_shift(train.positions, test.positions),
        "start_pose_mean_shift_z": _standardized_mean_shift(train.starts, test.starts),
        "speed_mean_shift_z": _standardized_mean_shift(train.speeds[:, None], test.speeds[:, None])
        if train.speeds.size and test.speeds.size else float("nan"),
        "descriptor_names": list(train.descriptor_names),
        "descriptor_nn": _standardized_nn_report(train.descriptors, test.descriptors),
        "train": {
            "positions": _axis_summary(train.positions),
            "starts": _axis_summary(train.starts),
            "speeds": _scalar_summary(train.speeds),
        },
        "test": {
            "positions": _axis_summary(test.positions),
            "starts": _axis_summary(test.starts),
            "speeds": _scalar_summary(test.speeds),
        },
    }

    shift_metrics = {
        "trajectory_position": trajectory_summary["position_mean_shift_z"],
        "initial_robot_pose": trajectory_summary["start_pose_mean_shift_z"],
        "speed": trajectory_summary["speed_mean_shift_z"],
        "trajectory_descriptor_nn_ratio": trajectory_summary["descriptor_nn"].get(
            "test_p95_to_train_p95_ratio", float("nan")
        ),
        "camera_pose": camera_summary["mean_shift_z"],
    }
    if image_summary.get("available"):
        shift_metrics["image_embedding_mean_cosine"] = image_summary["mean_cosine_shift"]
        shift_metrics["image_embedding_nn_ratio"] = image_summary["cosine_nn"].get(
            "test_p95_to_train_p95_ratio", float("nan")
        )

    plot_shift_bar(shift_metrics, dataset_output / "ood_signal_bar.png")
    train_values, test_values, train_scales = _collect_mean_bar_values(
        train, test, embedding_data, train_camera, test_camera
    )
    plot_mean_comparison_bar(
        train_values, test_values, train_scales, dataset_output / "mean_comparison_bar.png"
    )

    outputs = {
        "trajectory": str(trajectory_plot),
        "speed": str(dataset_output / "speed_train_test.png"),
        "initial_environment_3d": str(dataset_output / "initial_environment_3d.png"),
        "ood_signal_bar": str(dataset_output / "ood_signal_bar.png"),
        "mean_comparison_bar": str(dataset_output / "mean_comparison_bar.png"),
    }
    if embedding_plot is not None:
        outputs["image_embeddings"] = str(embedding_plot)

    summary: Dict[str, Any] = {
        "name": dataset_root.name,
        "mode": "random_episode_split",
        "dataset_root": str(dataset_root),
        "split": {
            "seed": int(args.seed),
            "train_ratio": float(args.train_ratio),
            "train_episode_count": len(split.train),
            "test_episode_count": len(split.test),
            "train_episodes": list(split.train),
            "test_episodes": list(split.test),
        },
        "config": {
            "state_key": config.state_key,
            "episode_key": config.episode_key,
            "timestamp_key": config.timestamp_key,
            "coord_indices": list(config.coord_indices),
            "coord_labels": list(config.coord_labels),
            "gripper_index": config.gripper_index,
            "notes": list(config.notes),
        },
        "shift_metrics": shift_metrics,
        "trajectory": trajectory_summary,
        "camera_pose": camera_summary,
        "image_embeddings": image_summary,
        "mean_bar_values": {
            "train": train_values,
            "test": test_values,
            "train_scale": train_scales,
        },
        "outputs": outputs,
    }
    with (dataset_output / "ood_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _write_report_text(summary, dataset_output / "ood_report.txt")
    return summary


def analyze_explicit_train_test(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    if args.train_dataset_root is None or args.test_dataset_root is None:
        raise ValueError("Both --train-dataset-root and --test-dataset-root are required.")

    train_root = args.train_dataset_root.resolve()
    test_root = args.test_dataset_root.resolve()
    train_config, train_episode_data = _load_dataset_trajectory(train_root, args)
    test_config, test_episode_data = _load_dataset_trajectory(test_root, args)
    if train_config.coord_indices != test_config.coord_indices:
        print("warning: train/test coordinate indices differ; comparing resolved coordinates anyway")

    train_ids = _select_episode_ids(train_episode_data.keys(), args.max_episodes, args.seed)
    test_ids = _select_episode_ids(test_episode_data.keys(), args.max_episodes, args.seed + 17)

    merged_root_name = f"{train_root.name}_vs_{test_root.name}"
    dataset_output = output_dir / merged_root_name
    dataset_output.mkdir(parents=True, exist_ok=True)

    train = _make_group("train", train_episode_data, train_ids, args.gripper_change_threshold)
    test = _make_group("test", test_episode_data, test_ids, args.gripper_change_threshold)

    split = Split(train=tuple(int(ep) for ep in train_ids), test=tuple(int(ep) for ep in test_ids))
    train_embedding = _load_embedding_split(args.image_output_dir.resolve(), train_root.name, split)
    test_embedding = _load_embedding_split(args.image_output_dir.resolve(), test_root.name, split)
    embedding_data = None
    if train_embedding is not None and test_embedding is not None:
        embedding_data = {
            "path": f"{train_embedding['path']} ; {test_embedding['path']}",
            "embeddings": np.concatenate(
                [
                    train_embedding["embeddings"][train_embedding["train_mask"]],
                    test_embedding["embeddings"][test_embedding["test_mask"]],
                ],
                axis=0,
            ),
            "coords_2d": np.concatenate(
                [
                    train_embedding["coords_2d"][train_embedding["train_mask"]],
                    test_embedding["coords_2d"][test_embedding["test_mask"]],
                ],
                axis=0,
            ),
            "image_key": np.concatenate(
                [
                    train_embedding["image_key"][train_embedding["train_mask"]],
                    test_embedding["image_key"][test_embedding["test_mask"]],
                ],
                axis=0,
            ),
        }
        n_train = int(np.sum(train_embedding["train_mask"]))
        n_test = int(np.sum(test_embedding["test_mask"]))
        embedding_data["train_mask"] = np.r_[np.ones(n_train, dtype=bool), np.zeros(n_test, dtype=bool)]
        embedding_data["test_mask"] = np.r_[np.zeros(n_train, dtype=bool), np.ones(n_test, dtype=bool)]

    train_camera, test_camera = _camera_positions_from_embeddings(embedding_data)
    trajectory_plot = dataset_output / "trajectory_train_test_3d.png"
    if train.positions.shape[1] < 3:
        trajectory_plot = dataset_output / "trajectory_train_test_2d.png"
    plot_trajectory_split(
        train, test, train_config.coord_labels, args.sample_points, args.seed, trajectory_plot
    )
    plot_speed_distribution(train.speeds, test.speeds, args.bins, dataset_output / "speed_train_test.png")
    plot_initial_environment_3d(
        train.starts,
        test.starts,
        train_camera,
        test_camera,
        train_config.coord_labels,
        dataset_output / "initial_environment_3d.png",
    )

    embedding_plot: Optional[Path] = None
    image_summary: Dict[str, Any] = {"available": False}
    if embedding_data is not None and np.any(embedding_data["train_mask"]) and np.any(embedding_data["test_mask"]):
        embedding_plot = dataset_output / "image_embedding_train_test.png"
        plot_embedding_split(
            embedding_data["coords_2d"],
            embedding_data["train_mask"],
            embedding_data["test_mask"],
            embedding_plot,
        )
        train_embed = embedding_data["embeddings"][embedding_data["train_mask"]]
        test_embed = embedding_data["embeddings"][embedding_data["test_mask"]]
        image_summary = {
            "available": True,
            "embedding_path": embedding_data["path"],
            "train_sample_count": int(train_embed.shape[0]),
            "test_sample_count": int(test_embed.shape[0]),
            "mean_cosine_shift": float(
                1.0
                - np.dot(
                    _normalize_rows(np.mean(train_embed, axis=0, keepdims=True))[0],
                    _normalize_rows(np.mean(test_embed, axis=0, keepdims=True))[0],
                )
            ),
            "cosine_nn": _cosine_nn_report(train_embed, test_embed),
        }

    trajectory_summary = {
        "position_mean_shift_z": _standardized_mean_shift(train.positions, test.positions),
        "start_pose_mean_shift_z": _standardized_mean_shift(train.starts, test.starts),
        "speed_mean_shift_z": _standardized_mean_shift(train.speeds[:, None], test.speeds[:, None])
        if train.speeds.size and test.speeds.size else float("nan"),
        "descriptor_names": list(train.descriptor_names),
        "descriptor_nn": _standardized_nn_report(train.descriptors, test.descriptors),
        "train": {
            "positions": _axis_summary(train.positions),
            "starts": _axis_summary(train.starts),
            "speeds": _scalar_summary(train.speeds),
        },
        "test": {
            "positions": _axis_summary(test.positions),
            "starts": _axis_summary(test.starts),
            "speeds": _scalar_summary(test.speeds),
        },
    }
    camera_summary = {
        "available": bool(train_camera.size and test_camera.size),
        "train": _axis_summary(train_camera),
        "test": _axis_summary(test_camera),
        "mean_shift_z": _standardized_mean_shift(train_camera, test_camera)
        if train_camera.size and test_camera.size else float("nan"),
        "image_key_counts": _camera_key_counts(embedding_data),
    }
    shift_metrics = {
        "trajectory_position": trajectory_summary["position_mean_shift_z"],
        "initial_robot_pose": trajectory_summary["start_pose_mean_shift_z"],
        "speed": trajectory_summary["speed_mean_shift_z"],
        "trajectory_descriptor_nn_ratio": trajectory_summary["descriptor_nn"].get(
            "test_p95_to_train_p95_ratio", float("nan")
        ),
        "camera_pose": camera_summary["mean_shift_z"],
    }
    if image_summary.get("available"):
        shift_metrics["image_embedding_mean_cosine"] = image_summary["mean_cosine_shift"]
        shift_metrics["image_embedding_nn_ratio"] = image_summary["cosine_nn"].get(
            "test_p95_to_train_p95_ratio", float("nan")
        )

    plot_shift_bar(shift_metrics, dataset_output / "ood_signal_bar.png")
    train_values, test_values, train_scales = _collect_mean_bar_values(
        train, test, embedding_data, train_camera, test_camera
    )
    plot_mean_comparison_bar(
        train_values, test_values, train_scales, dataset_output / "mean_comparison_bar.png"
    )
    outputs = {
        "trajectory": str(trajectory_plot),
        "speed": str(dataset_output / "speed_train_test.png"),
        "initial_environment_3d": str(dataset_output / "initial_environment_3d.png"),
        "ood_signal_bar": str(dataset_output / "ood_signal_bar.png"),
        "mean_comparison_bar": str(dataset_output / "mean_comparison_bar.png"),
    }
    if embedding_plot is not None:
        outputs["image_embeddings"] = str(embedding_plot)

    summary: Dict[str, Any] = {
        "name": merged_root_name,
        "mode": "explicit_train_test_roots",
        "train_dataset_root": str(train_root),
        "test_dataset_root": str(test_root),
        "split": {
            "train_episode_count": len(train.episode_ids),
            "test_episode_count": len(test.episode_ids),
            "train_episodes": list(train.episode_ids),
            "test_episodes": list(test.episode_ids),
        },
        "config": {
            "train": train_config.__dict__,
            "test": test_config.__dict__,
        },
        "shift_metrics": shift_metrics,
        "trajectory": trajectory_summary,
        "camera_pose": camera_summary,
        "image_embeddings": image_summary,
        "mean_bar_values": {
            "train": train_values,
            "test": test_values,
            "train_scale": train_scales,
        },
        "outputs": outputs,
    }
    with (dataset_output / "ood_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _write_report_text(summary, dataset_output / "ood_report.txt")
    return summary


def _resolve_dataset_roots(args: argparse.Namespace) -> List[Path]:
    roots = discover_dataset_roots(args.dataset_root.resolve(), args.all_datasets)
    if args.dataset_name:
        wanted = set(args.dataset_name)
        roots = [root for root in roots if root.name in wanted]
        missing = sorted(wanted - {root.name for root in roots})
        if missing:
            raise FileNotFoundError(f"Requested dataset name(s) not found: {missing}")
    return roots


def analyze() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    if args.train_dataset_root is not None or args.test_dataset_root is not None:
        summaries.append(analyze_explicit_train_test(args, output_dir))
    else:
        dataset_roots = _resolve_dataset_roots(args)
        for dataset_root in dataset_roots:
            print(f"Analyzing OOD split: {dataset_root.name}")
            summaries.append(analyze_random_split_dataset(dataset_root.resolve(), args, output_dir))

    merged = {
        "output_dir": str(output_dir),
        "summary_count": len(summaries),
        "summaries": summaries,
    }
    with (output_dir / "summary_all.json").open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"Saved OOD analysis to: {output_dir}")
    for summary in summaries:
        print(f"- {summary['name']}: {summary['outputs'].get('ood_signal_bar')}")


if __name__ == "__main__":
    analyze()
