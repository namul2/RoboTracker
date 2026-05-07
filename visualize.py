#!/usr/bin/env python3
"""Trajectory-based analyzer for LeRobot-format datasets.

Outputs:
- 2D/3D position scatter plot
- End-effector speed distribution histogram
- JSON summary with key statistics
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pyarrow.parquet as pq
    from matplotlib.colors import LinearSegmentedColormap
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependencies. Run: python3 -m pip install -r requirements.txt"
    ) from exc

SOFT_TRAJ_CMAP = LinearSegmentedColormap.from_list(
    "soft_purple_blue_yellow",
    ["#0D00FF", "#ff5454ff"],
    # ["#0D00FF", "#ff5f50ff"],
    # ["#3850FF", "#6ea8ff", "#6eff86", "#ffc337", "#ff5148"],
)


@dataclass(frozen=True)
class DatasetConfig:
    state_key: str
    episode_key: str
    timestamp_key: str
    coord_indices: Tuple[int, ...]
    coord_labels: Tuple[str, ...]
    gripper_index: Optional[int]
    notes: Tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze trajectory-based position/speed from LeRobot datasets."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/droid_100"),
        help=(
            "Path to a LeRobot dataset root, or a parent directory when "
            "--all-datasets is set."
        ),
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Analyze every child directory that looks like a LeRobot dataset.",
    )
    parser.add_argument(
        "--state-key",
        type=str,
        default=None,
        help="Feature key containing robot state vector. Default: auto-detect.",
    )
    parser.add_argument(
        "--episode-key",
        type=str,
        default=None,
        help="Episode index column key. Default: auto-detect.",
    )
    parser.add_argument(
        "--timestamp-key",
        type=str,
        default=None,
        help="Timestamp column key in seconds. Default: auto-detect.",
    )
    parser.add_argument(
        "--ee-indices",
        type=int,
        nargs="+",
        default=None,
        metavar="IDX",
        help=(
            "Indices in state vector for position coordinates. "
            "Use 2 or 3 values. Default: auto-detect from feature names/shape."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to store generated plots and summary JSON.",
    )
    parser.add_argument(
        "--sample-points",
        type=int,
        default=50000,
        help="Max points shown in scatter plots (downsampled for readability).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of bins for speed histogram.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling plot points.",
    )
    parser.add_argument(
        "--gripper-index",
        type=int,
        default=None,
        help="Index in state vector used for gripper open/close signal. Default: auto-detect.",
    )
    parser.add_argument(
        "--gripper-change-threshold",
        type=float,
        default=1e-6,
        help="Threshold for detecting first gripper change from initial value.",
    )
    parser.add_argument(
        "--viz-jitter-std",
        type=float,
        default=0.0,
        help=(
            "Std-dev of Gaussian jitter added only for visualization points "
            "(same unit as state coords)."
        ),
    )
    parser.add_argument(
        "--print-config-only",
        action="store_true",
        help="Print resolved dataset configuration without generating plots.",
    )
    return parser.parse_args()


def read_dataset_info(dataset_root: Path) -> Dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return {}
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_data_files(dataset_root: Path) -> List[Path]:
    data_root = dataset_root / "data"
    files = sorted(data_root.glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {data_root}")
    return files


def discover_dataset_roots(root: Path, all_datasets: bool) -> List[Path]:
    if (root / "data").is_dir() and (root / "meta").is_dir():
        return [root]
    if not all_datasets:
        raise FileNotFoundError(
            f"{root} is not a LeRobot dataset root. Use --all-datasets for a parent directory."
        )

    roots = [
        child
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / "data").is_dir() and (child / "meta").is_dir()
    ]
    if not roots:
        raise FileNotFoundError(f"No LeRobot dataset roots found under: {root}")
    return roots


def _feature_dim(feature: Dict[str, Any]) -> int:
    shape = feature.get("shape")
    if isinstance(shape, list) and len(shape) == 1 and isinstance(shape[0], int):
        return int(shape[0])
    return 0


def _feature_names(feature: Dict[str, Any], dim: int) -> List[str]:
    names = feature.get("names")
    flat: List[str] = []
    if isinstance(names, dict):
        for value in names.values():
            if isinstance(value, list):
                flat.extend(str(item) for item in value)
    elif isinstance(names, list) and len(names) == dim:
        flat = [str(item) for item in names]
    return flat if len(flat) == dim else []


def _is_numeric_vector_feature(feature: Dict[str, Any]) -> bool:
    dtype = str(feature.get("dtype", "")).lower()
    return _feature_dim(feature) >= 2 and (
        dtype.startswith("float")
        or dtype.startswith("int")
        or dtype in {"double", "halffloat"}
    )


def _column_names(parquet_file: Path) -> List[str]:
    return list(pq.read_schema(parquet_file).names)


def _resolve_key(
    explicit: Optional[str],
    preferred: Sequence[str],
    available_columns: Sequence[str],
    label: str,
) -> str:
    if explicit is not None:
        if explicit not in available_columns:
            raise KeyError(
                f"{label} key '{explicit}' is not in parquet columns: {available_columns}"
            )
        return explicit
    for key in preferred:
        if key in available_columns:
            return key
    raise KeyError(f"Could not auto-detect {label} key from columns: {available_columns}")


def _resolve_state_key(
    explicit: Optional[str],
    info: Dict[str, Any],
    available_columns: Sequence[str],
) -> str:
    if explicit is not None:
        if explicit not in available_columns:
            raise KeyError(
                f"state key '{explicit}' is not in parquet columns: {available_columns}"
            )
        return explicit

    features = info.get("features", {})
    for key in ("observation.state", "state", "agent_pos", "action"):
        feature = features.get(key)
        if key in available_columns and isinstance(feature, dict) and _is_numeric_vector_feature(feature):
            return key

    candidates = [
        key
        for key, feature in features.items()
        if key in available_columns
        and isinstance(feature, dict)
        and key.startswith("observation")
        and _is_numeric_vector_feature(feature)
    ]
    if not candidates:
        candidates = [
            key
            for key, feature in features.items()
            if key in available_columns
            and isinstance(feature, dict)
            and _is_numeric_vector_feature(feature)
        ]
    if not candidates:
        raise KeyError("Could not auto-detect a numeric vector state feature.")
    return candidates[0]


def _match_coord_names(names: Sequence[str]) -> Optional[Tuple[int, ...]]:
    lowered = [name.lower() for name in names]
    exact_xyz = []
    for axis in ("x", "y", "z"):
        if axis in lowered:
            exact_xyz.append(lowered.index(axis))
    if len(exact_xyz) == 3:
        return tuple(exact_xyz)

    exact_xy = []
    for axis in ("x", "y"):
        if axis in lowered:
            exact_xy.append(lowered.index(axis))
    if len(exact_xy) == 2:
        return tuple(exact_xy)
    return None


def _resolve_coord_indices(
    explicit: Optional[Sequence[int]],
    feature: Dict[str, Any],
    notes: List[str],
) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
    dim = _feature_dim(feature)
    names = _feature_names(feature, dim)

    if explicit is not None:
        indices = tuple(int(index) for index in explicit)
        if len(indices) not in {2, 3}:
            raise ValueError("--ee-indices must contain 2 or 3 indices.")
        if min(indices) < 0 or max(indices) >= dim:
            raise IndexError(f"--ee-indices {indices} out of range for state dim {dim}.")
        labels = tuple(names[index] if names else "XYZ"[i] for i, index in enumerate(indices))
        return indices, labels

    matched = _match_coord_names(names)
    if matched is not None:
        labels = tuple(names[index] for index in matched)
        notes.append(f"coordinate indices inferred from names: {labels}")
        return matched, labels

    if dim >= 3:
        notes.append(
            "coordinate names were not found; using first 3 state dimensions as coordinates"
        )
        return (0, 1, 2), ("X", "Y", "Z")
    if dim == 2:
        notes.append("2D state vector detected; using both state dimensions")
        return (0, 1), ("X", "Y")
    raise ValueError(f"state feature must have at least 2 dimensions. Got {dim}.")


def _resolve_gripper_index(
    explicit: Optional[int],
    feature: Dict[str, Any],
    notes: List[str],
) -> Optional[int]:
    dim = _feature_dim(feature)
    names = _feature_names(feature, dim)

    if explicit is not None:
        if explicit < 0 or explicit >= dim:
            raise IndexError(f"--gripper-index {explicit} out of range for state dim {dim}.")
        return int(explicit)

    for index, name in enumerate(names):
        if "gripper" in name.lower():
            notes.append(f"gripper index inferred from name: {index} ({name})")
            return index

    if dim == 7:
        notes.append("gripper name was not found; using index 6 for 7D robot state")
        return 6

    notes.append("gripper index was not inferred; gripper change markers disabled")
    return None


def resolve_dataset_config(
    dataset_root: Path,
    parquet_files: Sequence[Path],
    args: argparse.Namespace,
) -> DatasetConfig:
    info = read_dataset_info(dataset_root)
    available_columns = _column_names(parquet_files[0])
    features = info.get("features", {})
    notes: List[str] = []

    state_key = _resolve_state_key(args.state_key, info, available_columns)
    episode_key = _resolve_key(
        args.episode_key,
        ("episode_index", "episode", "episode_id"),
        available_columns,
        "episode",
    )
    timestamp_key = _resolve_key(
        args.timestamp_key,
        ("timestamp", "timestamps", "time"),
        available_columns,
        "timestamp",
    )

    feature = features.get(state_key)
    if not isinstance(feature, dict):
        # Fall back to the parquet sample if metadata is unavailable.
        table = pq.read_table(parquet_files[0], columns=[state_key])
        sample = _to_2d_state_array(table[state_key].slice(0, min(8, table.num_rows)).to_pylist())
        feature = {"dtype": "float64", "shape": [sample.shape[1]], "names": None}
        notes.append("meta/info.json feature metadata was unavailable; inferred state dim from parquet")

    coord_indices, coord_labels = _resolve_coord_indices(args.ee_indices, feature, notes)
    gripper_index = _resolve_gripper_index(args.gripper_index, feature, notes)

    return DatasetConfig(
        state_key=state_key,
        episode_key=episode_key,
        timestamp_key=timestamp_key,
        coord_indices=coord_indices,
        coord_labels=coord_labels,
        gripper_index=gripper_index,
        notes=tuple(notes),
    )


def _to_2d_state_array(state_pylist: Sequence[Sequence[float]]) -> np.ndarray:
    state_array = np.asarray(state_pylist, dtype=np.float64)
    if state_array.ndim != 2:
        raise ValueError(f"State array must be 2D. Got shape: {state_array.shape}")
    return state_array


def load_positions_by_episode(
    parquet_files: Iterable[Path],
    state_key: str,
    episode_key: str,
    timestamp_key: str,
    coord_indices: Sequence[int],
    gripper_index: Optional[int],
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    per_episode_ts: Dict[int, List[np.ndarray]] = defaultdict(list)
    per_episode_pos: Dict[int, List[np.ndarray]] = defaultdict(list)
    per_episode_gripper: Dict[int, List[np.ndarray]] = defaultdict(list)

    columns = [state_key, episode_key, timestamp_key]
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file, columns=columns)

        state_array = _to_2d_state_array(table[state_key].to_pylist())
        if max(coord_indices) >= state_array.shape[1]:
            raise IndexError(
                f"coordinate index out of range: max index={max(coord_indices)}, "
                f"state dim={state_array.shape[1]}"
            )
        if gripper_index is not None and gripper_index >= state_array.shape[1]:
            raise IndexError(
                f"gripper index out of range: index={gripper_index}, "
                f"state dim={state_array.shape[1]}"
            )

        ee_pos = state_array[:, coord_indices]
        gripper = (
            state_array[:, gripper_index]
            if gripper_index is not None
            else np.empty((state_array.shape[0],), dtype=np.float64)
        )
        episodes = np.asarray(table[episode_key].to_pylist(), dtype=np.int64)
        timestamps = np.asarray(table[timestamp_key].to_pylist(), dtype=np.float64)

        unique_eps = np.unique(episodes)
        for ep in unique_eps:
            mask = episodes == ep
            per_episode_ts[int(ep)].append(timestamps[mask])
            per_episode_pos[int(ep)].append(ee_pos[mask])
            per_episode_gripper[int(ep)].append(gripper[mask])

    episode_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for ep, ts_parts in per_episode_ts.items():
        pos_parts = per_episode_pos[ep]
        gripper_parts = per_episode_gripper[ep]
        ts = np.concatenate(ts_parts, axis=0)
        pos = np.concatenate(pos_parts, axis=0)
        g = np.concatenate(gripper_parts, axis=0)
        episode_data[ep] = (ts, pos, g)

    return episode_data


def compute_episode_metrics(
    episode_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    gripper_change_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_positions_chunks: List[np.ndarray] = []
    all_progress_chunks: List[np.ndarray] = []
    speed_chunks: List[np.ndarray] = []
    start_points: List[np.ndarray] = []
    first_gripper_change_points: List[np.ndarray] = []

    for _, (ts, pos, gripper) in episode_data.items():
        if ts.size == 0:
            continue

        order = np.argsort(ts)
        ts_sorted = ts[order]
        pos_sorted = pos[order]
        gripper_sorted = gripper[order] if gripper.size else gripper

        n = ts_sorted.size
        if n == 1:
            progress = np.array([0.0], dtype=np.float64)
        else:
            progress = np.linspace(0.0, 1.0, n, endpoint=True, dtype=np.float64)

        all_positions_chunks.append(pos_sorted)
        all_progress_chunks.append(progress)

        start_points.append(pos_sorted[0])
        if gripper_sorted.size:
            first_value = gripper_sorted[0]
            diff = np.abs(gripper_sorted - first_value)
            changed = np.flatnonzero(diff > gripper_change_threshold)
            if changed.size > 0:
                first_gripper_change_points.append(pos_sorted[changed[0]])

        if ts.size < 2:
            continue

        dt = np.diff(ts_sorted)
        dp = np.linalg.norm(np.diff(pos_sorted, axis=0), axis=1)
        valid = dt > 1e-9
        if not np.any(valid):
            continue

        speed = dp[valid] / dt[valid]
        speed_chunks.append(speed)

    if not all_positions_chunks:
        raise ValueError("No valid points found to analyze.")

    all_positions = np.concatenate(all_positions_chunks, axis=0)
    all_progress = np.concatenate(all_progress_chunks, axis=0)
    start_points_arr = np.asarray(start_points, dtype=np.float64)
    position_dim = all_positions.shape[1]
    change_points_arr = (
        np.asarray(first_gripper_change_points, dtype=np.float64)
        if first_gripper_change_points
        else np.empty((0, position_dim), dtype=np.float64)
    )
    if not speed_chunks:
        speeds = np.empty((0,), dtype=np.float64)
    else:
        speeds = np.concatenate(speed_chunks, axis=0)
    return all_positions, all_progress, start_points_arr, change_points_arr, speeds


def maybe_downsample(
    points: np.ndarray, progress: np.ndarray, max_points: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= max_points:
        return points, progress
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx], progress[idx]


def apply_visual_jitter(points: np.ndarray, jitter_std: float, seed: int) -> np.ndarray:
    if jitter_std <= 0:
        return points
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=jitter_std, size=points.shape)
    return points + noise


def plot_ee_distribution(
    points_xyz: np.ndarray,
    episode_progress: np.ndarray,
    start_points: np.ndarray,
    change_points: np.ndarray,
    coord_labels: Sequence[str],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        points_xyz[:, 0],
        points_xyz[:, 1],
        points_xyz[:, 2],
        c=episode_progress,
        cmap=SOFT_TRAJ_CMAP,
        s=2.2,
        alpha=0.7,
        linewidths=0.0,
    )

    if start_points.size > 0:
        ax.scatter(
            start_points[:, 0],
            start_points[:, 1],
            start_points[:, 2],
            marker="^",
            c="#48e1ff",
            s=36,
            alpha=0.9,
            label="Episode Start",
        )
    if change_points.size > 0:
        ax.scatter(
            change_points[:, 0],
            change_points[:, 1],
            change_points[:, 2],
            marker="x",
            c="#d62728",
            s=42,
            alpha=0.9,
            label="First Gripper Change",
        )

    ax.set_title("End-Effector 3D Position Distribution")
    ax.set_xlabel(coord_labels[0])
    ax.set_ylabel(coord_labels[1])
    ax.set_zlabel(coord_labels[2])
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[1], xlim[0])
    ax.set_ylim(ylim[1], ylim[0])
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.75, pad=0.1)
    cbar.set_label("Episode Progress (frame step ratio)")
    if start_points.size > 0 or change_points.size > 0:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_position_distribution_2d(
    points_xy: np.ndarray,
    episode_progress: np.ndarray,
    start_points: np.ndarray,
    change_points: np.ndarray,
    coord_labels: Sequence[str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    scatter = ax.scatter(
        points_xy[:, 0],
        points_xy[:, 1],
        c=episode_progress,
        cmap=SOFT_TRAJ_CMAP,
        s=2.2,
        alpha=0.7,
        linewidths=0.0,
    )

    if start_points.size > 0:
        ax.scatter(
            start_points[:, 0],
            start_points[:, 1],
            marker="^",
            c="#A1FFFA",
            s=20,
            alpha=0.9,
            label="Episode Start",
        )
    if change_points.size > 0:
        ax.scatter(
            change_points[:, 0],
            change_points[:, 1],
            marker="x",
            c="#ff0000",
            s=42,
            alpha=0.9,
            label="First Gripper Change",
        )

    ax.set_title("2D Position Distribution")
    ax.set_xlabel(coord_labels[0])
    ax.set_ylabel(coord_labels[1])
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Episode Progress (frame step ratio)")
    if start_points.size > 0 or change_points.size > 0:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_ee_projection(
    points_xyz: np.ndarray,
    episode_progress: np.ndarray,
    start_points: np.ndarray,
    change_points: np.ndarray,
    axis_i: int,
    axis_j: int,
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    scatter = ax.scatter(
        points_xyz[:, axis_i],
        points_xyz[:, axis_j],
        c=episode_progress,
        cmap=SOFT_TRAJ_CMAP,
        s=2.2,
        alpha=0.7,
        linewidths=0.0,
    )

    if start_points.size > 0:
        ax.scatter(
            start_points[:, axis_i],
            start_points[:, axis_j],
            marker="^",
            c="#A1FFFA",
            s=20,
            alpha=0.9,
            label="Episode Start",
        )
    if change_points.size > 0:
        ax.scatter(
            change_points[:, axis_i],
            change_points[:, axis_j],
            marker="x",
            c="#ff0000",
            s=42,
            alpha=0.9,
            label="First Gripper Change",
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Keep orientation consistent with the original 3D view convention.
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if x_label.upper() in {"X", "Y"}:
        ax.set_xlim(xlim[1], xlim[0])
    if y_label.upper() in {"X", "Y"}:
        ax.set_ylim(ylim[1], ylim[0])

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Episode Progress (frame step ratio)")

    if start_points.size > 0 or change_points.size > 0:
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_speed_histogram(speeds: np.ndarray, bins: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(speeds, bins=bins)
    ax.set_title("Position Speed Distribution")
    ax.set_xlabel("Speed (units/s)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def summarize(
    positions: np.ndarray,
    speeds: np.ndarray,
    episode_count: int,
    config: DatasetConfig,
    first_change_points_count: int,
    viz_jitter_std: float,
) -> dict:
    def safe_stat(arr: np.ndarray, fn, default=np.nan):
        return float(fn(arr)) if arr.size else float(default)

    return {
        "episode_count": int(episode_count),
        "frame_count": int(positions.shape[0]),
        "state_key": config.state_key,
        "episode_key": config.episode_key,
        "timestamp_key": config.timestamp_key,
        "position_dim": int(positions.shape[1]),
        "coord_indices": [int(i) for i in config.coord_indices],
        "coord_labels": list(config.coord_labels),
        "gripper_index": (
            int(config.gripper_index) if config.gripper_index is not None else None
        ),
        "config_notes": list(config.notes),
        "speed_is_normalized": False,
        "viz_jitter_std": float(viz_jitter_std),
        "position_min": positions.min(axis=0).tolist(),
        "position_max": positions.max(axis=0).tolist(),
        "position_mean": positions.mean(axis=0).tolist(),
        "episodes_with_gripper_change": int(first_change_points_count),
        "speed_count": int(speeds.size),
        "speed_min": safe_stat(speeds, np.min),
        "speed_max": safe_stat(speeds, np.max),
        "speed_mean": safe_stat(speeds, np.mean),
        "speed_median": safe_stat(speeds, np.median),
        "speed_p95": safe_stat(speeds, lambda x: np.percentile(x, 95)),
        "speed_p99": safe_stat(speeds, lambda x: np.percentile(x, 99)),
    }


def _print_resolved_config(dataset_root: Path, config: DatasetConfig) -> None:
    print(f"- Dataset: {dataset_root.name}")
    print(f"  state_key: {config.state_key}")
    print(f"  episode_key: {config.episode_key}")
    print(f"  timestamp_key: {config.timestamp_key}")
    print(f"  coord_indices: {list(config.coord_indices)}")
    print(f"  coord_labels: {list(config.coord_labels)}")
    print(f"  gripper_index: {config.gripper_index}")
    for note in config.notes:
        print(f"  note: {note}")


def analyze_dataset(
    dataset_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Optional[dict]:
    parquet_files = list_data_files(dataset_root)
    config = resolve_dataset_config(dataset_root, parquet_files, args)
    _print_resolved_config(dataset_root, config)
    if args.print_config_only:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    episode_data = load_positions_by_episode(
        parquet_files=parquet_files,
        state_key=config.state_key,
        episode_key=config.episode_key,
        timestamp_key=config.timestamp_key,
        coord_indices=config.coord_indices,
        gripper_index=config.gripper_index,
    )
    positions, progress, start_points, change_points, speeds = compute_episode_metrics(
        episode_data=episode_data,
        gripper_change_threshold=args.gripper_change_threshold,
    )

    points_for_plot, progress_for_plot = maybe_downsample(
        positions, progress, args.sample_points, args.seed
    )
    points_for_plot = apply_visual_jitter(
        points_for_plot, args.viz_jitter_std, args.seed + 17
    )
    position_dim = positions.shape[1]
    position_plot_path = (
        output_dir / "position_distribution_3d.png"
        if position_dim >= 3
        else output_dir / "position_distribution_2d.png"
    )
    speed_plot_path = output_dir / "ee_speed_distribution_hist.png"
    summary_path = output_dir / "summary.json"

    if position_dim >= 3:
        plot_ee_distribution(
            points_for_plot,
            progress_for_plot,
            start_points,
            change_points,
            config.coord_labels,
            position_plot_path,
        )
        projection_specs = (
            (0, 1, "XY Projection", config.coord_labels[0], config.coord_labels[1]),
            (1, 2, "YZ Projection", config.coord_labels[1], config.coord_labels[2]),
            (0, 2, "XZ Projection", config.coord_labels[0], config.coord_labels[2]),
        )
        for axis_i, axis_j, title, x_label, y_label in projection_specs:
            plot_ee_projection(
                points_for_plot,
                progress_for_plot,
                start_points,
                change_points,
                axis_i,
                axis_j,
                title,
                x_label,
                y_label,
                output_dir / f"position_projection_{x_label.lower()}{y_label.lower()}.png",
            )
    else:
        plot_position_distribution_2d(
            points_for_plot,
            progress_for_plot,
            start_points,
            change_points,
            config.coord_labels,
            position_plot_path,
        )
    plot_speed_histogram(speeds, args.bins, speed_plot_path)

    summary = summarize(
        positions,
        speeds,
        len(episode_data),
        config,
        change_points.shape[0],
        args.viz_jitter_std,
    )
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Analysis complete.")
    print(f"- Dataset root: {dataset_root}")
    print(f"- Parquet files: {len(parquet_files)}")
    print(f"- Episodes: {summary['episode_count']}")
    print(f"- Frames: {summary['frame_count']}")
    print(f"- Position plot: {position_plot_path}")
    print(f"- Speed plot: {speed_plot_path}")
    print(f"- Summary: {summary_path}")
    return summary


def main() -> None:
    args = parse_args()

    root = args.dataset_root.resolve()
    output_root = args.output_dir.resolve()
    dataset_roots = discover_dataset_roots(root, args.all_datasets)

    summaries: Dict[str, dict] = {}
    for dataset_root in dataset_roots:
        dataset_output_dir = (
            output_root / dataset_root.name if len(dataset_roots) > 1 else output_root
        )
        summary = analyze_dataset(dataset_root, dataset_output_dir, args)
        if summary is not None:
            summaries[dataset_root.name] = summary

    if len(summaries) > 1:
        output_root.mkdir(parents=True, exist_ok=True)
        combined_summary_path = output_root / "summary_all.json"
        with combined_summary_path.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"Combined summary: {combined_summary_path}")


if __name__ == "__main__":
    main()
