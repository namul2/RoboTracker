#!/usr/bin/env python3
"""Episode-level OOD inspector for LeRobot datasets.

The dataset-level OOD detector answers "are these two splits different?".
This inspector answers the next question: "which episodes look OOD, and why?".

It reuses the trajectory/config/image helpers from ood_detection.py and
visualize.py, then scores each episode against the train distribution across:
- trajectory descriptor nearest-neighbor distance
- velocity statistics
- initial robot pose
- first-frame image embeddings, when cached embeddings are available
- camera pose / camera-key coverage, when cached image samples are available
"""

from __future__ import annotations

import argparse
import csv
import json
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
from ood_detection import (
    _nearest_distances,
    _normalize_rows,
    _random_split,
    _select_episode_ids,
)
from visualize import (
    DatasetConfig,
    discover_dataset_roots,
    list_data_files,
    load_positions_by_episode,
    resolve_dataset_config,
)


SIGNAL_ORDER = (
    "trajectory",
    "velocity",
    "initial_pose",
    "image_embedding",
    "camera_pose",
)
DEFAULT_WEIGHTS = {
    "trajectory": 1.0,
    "velocity": 1.0,
    "initial_pose": 1.0,
    "image_embedding": 1.0,
    "camera_pose": 0.75,
}


@dataclass(frozen=True)
class EpisodeFeatures:
    dataset: str
    episode_id: int
    descriptor: np.ndarray
    start_pose: np.ndarray
    path_length: float
    duration: float
    mean_speed: float
    max_step: float
    speed_p95: float
    displacement: float
    embedding: Optional[np.ndarray]
    camera_position: Optional[np.ndarray]
    camera_keys: Tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.dataset}/ep{self.episode_id}"


@dataclass(frozen=True)
class EpisodeScore:
    feature: EpisodeFeatures
    signal_scores: Dict[str, float]
    raw_values: Dict[str, float]
    contributions: Dict[str, float]
    total_score: float
    reason: str
    missing_signals: Tuple[str, ...]


@dataclass(frozen=True)
class TrainReference:
    features: Tuple[EpisodeFeatures, ...]
    descriptor_keys: Tuple[str, ...]
    descriptor_center: np.ndarray
    descriptor_scale: np.ndarray
    descriptor_train_z: np.ndarray
    descriptor_train_loo: np.ndarray
    descriptor_loo_p95: float
    start_mean: np.ndarray
    start_std: np.ndarray
    start_train_norm: np.ndarray
    start_norm_p95: float
    velocity_names: Tuple[str, ...]
    velocity_mean: np.ndarray
    velocity_std: np.ndarray
    velocity_train_norm: np.ndarray
    velocity_norm_p95: float
    embeddings: Optional[np.ndarray]
    embedding_keys: Tuple[str, ...]
    embedding_train_loo: np.ndarray
    embedding_loo_p95: float
    camera_mean: Optional[np.ndarray]
    camera_std: Optional[np.ndarray]
    camera_train_norm: np.ndarray
    camera_norm_p95: float
    camera_key_set: Tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank and explain OOD episodes using train-normalized signals."
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
        help="Inspect every child dataset under --dataset-root using random splits.",
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
        help="Explicit train dataset root. Requires --test-dataset-root.",
    )
    parser.add_argument(
        "--test-dataset-root",
        type=Path,
        default=None,
        help="Explicit test dataset root. Requires --train-dataset-root.",
    )
    parser.add_argument(
        "--preset",
        choices=("none", "libero_droid"),
        default="none",
        help=(
            "Run canned examples. libero_droid writes LIBERO split, DROID split, "
            "and LIBERO-train vs DROID-test reports."
        ),
    )
    parser.add_argument(
        "--libero-name",
        type=str,
        default="libero_10_image",
        help="Dataset child name used by --preset libero_droid.",
    )
    parser.add_argument(
        "--droid-name",
        type=str,
        default="droid_100",
        help="Dataset child name used by --preset libero_droid.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ood_episode_inspector"),
        help="Directory for inspector reports and plots.",
    )
    parser.add_argument(
        "--image-output-dir",
        type=Path,
        default=Path("outputs/image_distribution"),
        help="Directory containing image_clustering.py embeddings.npz outputs.",
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
        help="Optional cap on episodes before splitting/comparison.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Episodes shown in plots.")
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
    parser.add_argument("--gripper-index", type=int, default=None, help="Override gripper index.")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help=(
            "Comma-separated signal weights, e.g. "
            "trajectory=1.2,velocity=1,initial_pose=1,image_embedding=0.8,camera_pose=0.5"
        ),
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


def _safe_p95(arr: np.ndarray, fallback: float = 1.0) -> float:
    if arr.size == 0:
        return fallback
    value = float(np.percentile(arr, 95))
    if not np.isfinite(value) or value < 1e-9:
        return fallback
    return value


def _safe_std(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    std = np.std(arr, axis=axis)
    return np.where(std < 1e-9, 1.0, std)


def _standardized_norm(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.empty((0,), dtype=np.float64)
    z = (np.atleast_2d(values) - mean) / std
    return np.sqrt(np.mean(np.square(z), axis=1))


def _episode_descriptor(ts: np.ndarray, pos: np.ndarray) -> Dict[str, Any]:
    order = np.argsort(ts)
    ts_s = ts[order]
    pos_s = pos[order]
    diffs = np.diff(pos_s, axis=0)
    step_dist = np.linalg.norm(diffs, axis=1) if diffs.size else np.empty(0)
    duration = max(float(ts_s[-1] - ts_s[0]), 0.0) if ts_s.size else 0.0
    path_length = float(np.sum(step_dist)) if step_dist.size else 0.0
    mean_speed = path_length / duration if duration > 1e-9 else 0.0
    max_step = float(np.max(step_dist)) if step_dist.size else 0.0
    if ts_s.size > 1 and step_dist.size:
        dt = np.diff(ts_s)
        valid = dt > 1e-9
        speed_p95 = float(np.percentile(step_dist[valid] / dt[valid], 95)) if np.any(valid) else 0.0
    else:
        speed_p95 = 0.0
    displacement = float(np.linalg.norm(pos_s[-1] - pos_s[0])) if pos_s.size else 0.0
    descriptor = np.asarray(
        [
            *pos_s[0].tolist(),
            *pos_s[-1].tolist(),
            *(pos_s[-1] - pos_s[0]).tolist(),
            path_length,
            duration,
            mean_speed,
            max_step,
            speed_p95,
            displacement,
        ],
        dtype=np.float64,
    )
    return {
        "descriptor": descriptor,
        "start_pose": pos_s[0].astype(np.float64),
        "path_length": path_length,
        "duration": duration,
        "mean_speed": mean_speed,
        "max_step": max_step,
        "speed_p95": speed_p95,
        "displacement": displacement,
    }


def _embedding_path(image_output_dir: Path, dataset_name: str) -> Path:
    return image_output_dir / dataset_name / "embeddings.npz"


def _load_episode_image_features(
    image_output_dir: Path,
    dataset_name: str,
) -> Dict[int, Dict[str, Any]]:
    path = _embedding_path(image_output_dir, dataset_name)
    if not path.is_file():
        return {}
    data = np.load(path, allow_pickle=True)
    episodes = np.asarray(data["episode_index"], dtype=np.int64)
    embeddings = _normalize_rows(np.asarray(data["embeddings"], dtype=np.float64))
    image_keys = np.asarray(data["image_key"]).astype(str)
    unique_keys = sorted(set(str(key) for key in image_keys.tolist()))
    poses = resolve_camera_poses(unique_keys, {})
    pose_by_key = {pose.key: pose.position for pose in poses}

    out: Dict[int, Dict[str, Any]] = {}
    for ep in sorted(set(int(value) for value in episodes.tolist())):
        mask = episodes == ep
        ep_embeddings = embeddings[mask]
        ep_keys = tuple(sorted(set(str(key) for key in image_keys[mask].tolist())))
        ep_camera = np.asarray([pose_by_key[str(key)] for key in image_keys[mask]], dtype=np.float64)
        mean_embedding = _normalize_rows(np.mean(ep_embeddings, axis=0, keepdims=True))[0]
        out[int(ep)] = {
            "embedding": mean_embedding,
            "camera_position": np.mean(ep_camera, axis=0),
            "camera_keys": ep_keys,
        }
    return out


def _build_episode_features(
    dataset_root: Path,
    episode_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    episode_ids: Iterable[int],
    image_output_dir: Path,
) -> Tuple[EpisodeFeatures, ...]:
    image_by_episode = _load_episode_image_features(image_output_dir, dataset_root.name)
    rows: List[EpisodeFeatures] = []
    for ep in sorted(int(value) for value in episode_ids):
        if ep not in episode_data:
            continue
        ts, pos, _ = episode_data[ep]
        if ts.size == 0 or pos.size == 0:
            continue
        desc = _episode_descriptor(ts, pos)
        image_info = image_by_episode.get(ep, {})
        rows.append(
            EpisodeFeatures(
                dataset=dataset_root.name,
                episode_id=ep,
                descriptor=desc["descriptor"],
                start_pose=desc["start_pose"],
                path_length=desc["path_length"],
                duration=desc["duration"],
                mean_speed=desc["mean_speed"],
                max_step=desc["max_step"],
                speed_p95=desc["speed_p95"],
                displacement=desc["displacement"],
                embedding=image_info.get("embedding"),
                camera_position=image_info.get("camera_position"),
                camera_keys=tuple(image_info.get("camera_keys", ())),
            )
        )
    return tuple(rows)


def _leave_one_out_distances(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.empty((0,), dtype=np.float64)
    out = np.empty((values.shape[0],), dtype=np.float64)
    for idx in range(values.shape[0]):
        ref = np.concatenate([values[:idx], values[idx + 1 :]], axis=0)
        out[idx] = _nearest_distances(values[idx : idx + 1], ref)[0]
    return out


def _cosine_leave_one_out(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.shape[0] < 2:
        return np.empty((0,), dtype=np.float64)
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)
    return 1.0 - np.max(sims, axis=1)


def _make_train_reference(train_features: Sequence[EpisodeFeatures]) -> TrainReference:
    descriptors = np.asarray([f.descriptor for f in train_features], dtype=np.float64)
    descriptor_center = np.mean(descriptors, axis=0)
    descriptor_scale = _safe_std(descriptors, axis=0)
    descriptor_train_z = (descriptors - descriptor_center) / descriptor_scale
    descriptor_train_loo = _leave_one_out_distances(descriptor_train_z)

    starts = np.asarray([f.start_pose for f in train_features], dtype=np.float64)
    start_mean = np.mean(starts, axis=0)
    start_std = _safe_std(starts, axis=0)
    start_train_norm = _standardized_norm(starts, start_mean, start_std)

    velocity_names = ("path_length", "duration", "mean_speed", "speed_p95", "displacement")
    velocity_values = np.asarray(
        [
            [f.path_length, f.duration, f.mean_speed, f.speed_p95, f.displacement]
            for f in train_features
        ],
        dtype=np.float64,
    )
    velocity_mean = np.mean(velocity_values, axis=0)
    velocity_std = _safe_std(velocity_values, axis=0)
    velocity_train_norm = _standardized_norm(velocity_values, velocity_mean, velocity_std)

    train_embeddings = [f.embedding for f in train_features if f.embedding is not None]
    embedding_keys = tuple(f.key for f in train_features if f.embedding is not None)
    embeddings = (
        _normalize_rows(np.asarray(train_embeddings, dtype=np.float64))
        if train_embeddings
        else None
    )
    embedding_train_loo = (
        _cosine_leave_one_out(embeddings)
        if embeddings is not None
        else np.empty((0,), dtype=np.float64)
    )

    train_camera = [f.camera_position for f in train_features if f.camera_position is not None]
    if train_camera:
        camera_values = np.asarray(train_camera, dtype=np.float64)
        camera_mean = np.mean(camera_values, axis=0)
        camera_std = _safe_std(camera_values, axis=0)
        camera_train_norm = _standardized_norm(camera_values, camera_mean, camera_std)
    else:
        camera_mean = None
        camera_std = None
        camera_train_norm = np.empty((0,), dtype=np.float64)

    camera_key_set = tuple(sorted(set(key for f in train_features for key in f.camera_keys)))

    return TrainReference(
        features=tuple(train_features),
        descriptor_keys=tuple(f.key for f in train_features),
        descriptor_center=descriptor_center,
        descriptor_scale=descriptor_scale,
        descriptor_train_z=descriptor_train_z,
        descriptor_train_loo=descriptor_train_loo,
        descriptor_loo_p95=_safe_p95(descriptor_train_loo),
        start_mean=start_mean,
        start_std=start_std,
        start_train_norm=start_train_norm,
        start_norm_p95=_safe_p95(start_train_norm),
        velocity_names=velocity_names,
        velocity_mean=velocity_mean,
        velocity_std=velocity_std,
        velocity_train_norm=velocity_train_norm,
        velocity_norm_p95=_safe_p95(velocity_train_norm),
        embeddings=embeddings,
        embedding_keys=embedding_keys,
        embedding_train_loo=embedding_train_loo,
        embedding_loo_p95=_safe_p95(embedding_train_loo),
        camera_mean=camera_mean,
        camera_std=camera_std,
        camera_train_norm=camera_train_norm,
        camera_norm_p95=_safe_p95(camera_train_norm),
        camera_key_set=camera_key_set,
    )


def _parse_weights(raw: Optional[str]) -> Dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if not raw:
        return weights
    for part in raw.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --weights item: {part!r}")
        key, value = part.split("=", 1)
        key = key.strip()
        if key not in weights:
            raise KeyError(f"Unknown signal weight '{key}'. Valid: {sorted(weights)}")
        weights[key] = float(value)
    return weights


def _feature_velocity_vector(feature: EpisodeFeatures) -> np.ndarray:
    return np.asarray(
        [
            feature.path_length,
            feature.duration,
            feature.mean_speed,
            feature.speed_p95,
            feature.displacement,
        ],
        dtype=np.float64,
    )


def _score_one(
    feature: EpisodeFeatures,
    reference: TrainReference,
    weights: Dict[str, float],
    exclude_self: bool = False,
) -> EpisodeScore:
    signal_scores: Dict[str, float] = {}
    raw_values: Dict[str, float] = {}
    missing: List[str] = []

    descriptor_z = (feature.descriptor - reference.descriptor_center) / reference.descriptor_scale
    descriptor_ref = reference.descriptor_train_z
    if exclude_self and feature.key in reference.descriptor_keys and descriptor_ref.shape[0] > 1:
        idx = reference.descriptor_keys.index(feature.key)
        descriptor_ref = np.concatenate([descriptor_ref[:idx], descriptor_ref[idx + 1 :]], axis=0)
    trajectory_distance = _nearest_distances(
        descriptor_z[None, :],
        descriptor_ref,
    )[0]
    signal_scores["trajectory"] = float(trajectory_distance / reference.descriptor_loo_p95)
    raw_values["trajectory_nn_distance"] = float(trajectory_distance)
    raw_values["path_length"] = float(feature.path_length)
    raw_values["duration"] = float(feature.duration)
    raw_values["displacement"] = float(feature.displacement)

    velocity_values = _feature_velocity_vector(feature)
    velocity_norm = _standardized_norm(
        velocity_values[None, :],
        reference.velocity_mean,
        reference.velocity_std,
    )[0]
    signal_scores["velocity"] = float(velocity_norm / reference.velocity_norm_p95)
    raw_values["velocity_z_norm"] = float(velocity_norm)
    raw_values["mean_speed"] = float(feature.mean_speed)
    raw_values["speed_p95"] = float(feature.speed_p95)

    start_norm = _standardized_norm(
        feature.start_pose[None, :],
        reference.start_mean,
        reference.start_std,
    )[0]
    signal_scores["initial_pose"] = float(start_norm / reference.start_norm_p95)
    raw_values["initial_pose_z_norm"] = float(start_norm)
    for idx, value in enumerate(feature.start_pose):
        raw_values[f"start_{idx}"] = float(value)

    if feature.embedding is not None and reference.embeddings is not None:
        query = _normalize_rows(feature.embedding[None, :])
        embedding_ref = reference.embeddings
        if exclude_self and feature.key in reference.embedding_keys and embedding_ref.shape[0] > 1:
            idx = reference.embedding_keys.index(feature.key)
            embedding_ref = np.concatenate([embedding_ref[:idx], embedding_ref[idx + 1 :]], axis=0)
        image_distance = float(1.0 - np.max(query @ embedding_ref.T))
        signal_scores["image_embedding"] = image_distance / reference.embedding_loo_p95
        raw_values["image_cosine_nn_distance"] = image_distance
    else:
        signal_scores["image_embedding"] = 0.0
        missing.append("image_embedding")

    if feature.camera_position is not None and reference.camera_mean is not None:
        camera_norm = _standardized_norm(
            feature.camera_position[None, :],
            reference.camera_mean,
            reference.camera_std,
        )[0]
        train_key_set = set(reference.camera_key_set)
        episode_key_set = set(feature.camera_keys)
        missing_key_ratio = (
            len(episode_key_set - train_key_set) / max(len(episode_key_set), 1)
            if episode_key_set
            else 0.0
        )
        camera_score = (camera_norm / reference.camera_norm_p95) + missing_key_ratio
        signal_scores["camera_pose"] = float(camera_score)
        raw_values["camera_pose_z_norm"] = float(camera_norm)
        raw_values["camera_missing_key_ratio"] = float(missing_key_ratio)
    else:
        signal_scores["camera_pose"] = 0.0
        missing.append("camera_pose")

    contributions = {
        name: float(signal_scores[name] * max(weights.get(name, 0.0), 0.0))
        for name in SIGNAL_ORDER
    }
    total_weight = sum(max(weights.get(name, 0.0), 0.0) for name in SIGNAL_ORDER)
    total_score = (
        float(sum(contributions.values()) / total_weight)
        if total_weight > 1e-9
        else 0.0
    )
    reason = _make_reason(signal_scores, raw_values, missing)
    return EpisodeScore(
        feature=feature,
        signal_scores=signal_scores,
        raw_values=raw_values,
        contributions=contributions,
        total_score=total_score,
        reason=reason,
        missing_signals=tuple(missing),
    )


def _make_reason(
    signal_scores: Dict[str, float],
    raw_values: Dict[str, float],
    missing_signals: Sequence[str],
) -> str:
    ranked = sorted(
        ((name, signal_scores.get(name, 0.0)) for name in SIGNAL_ORDER),
        key=lambda item: item[1],
        reverse=True,
    )
    reasons: List[str] = []
    for name, score in ranked[:3]:
        if score <= 0.05:
            continue
        if name == "trajectory":
            reasons.append(
                f"trajectory descriptor is {score:.2f}x the train 95th-percentile NN baseline"
            )
        elif name == "velocity":
            reasons.append(
                f"velocity profile is shifted (mean speed {raw_values.get('mean_speed', 0.0):.4g})"
            )
        elif name == "initial_pose":
            reasons.append(
                f"initial pose is {raw_values.get('initial_pose_z_norm', 0.0):.2f} train-std units away"
            )
        elif name == "image_embedding":
            reasons.append(
                f"first-frame embedding is visually distant (cosine NN {raw_values.get('image_cosine_nn_distance', 0.0):.4f})"
            )
        elif name == "camera_pose":
            reasons.append("camera pose/key pattern differs from train")
    if missing_signals:
        reasons.append("missing " + ", ".join(missing_signals) + " signal")
    return "; ".join(reasons) if reasons else "close to train distribution across available signals"


def _score_features(
    features: Sequence[EpisodeFeatures],
    reference: TrainReference,
    weights: Dict[str, float],
    exclude_self: bool = False,
) -> Tuple[EpisodeScore, ...]:
    return tuple(_score_one(feature, reference, weights, exclude_self=exclude_self) for feature in features)


def _score_to_dict(score: EpisodeScore) -> Dict[str, Any]:
    return {
        "dataset": score.feature.dataset,
        "episode_id": int(score.feature.episode_id),
        "episode_key": score.feature.key,
        "total_score": score.total_score,
        "reason": score.reason,
        "signal_scores": score.signal_scores,
        "weighted_contributions": score.contributions,
        "raw_values": score.raw_values,
        "camera_keys": list(score.feature.camera_keys),
        "missing_signals": list(score.missing_signals),
    }


def _write_scores_csv(scores: Sequence[EpisodeScore], out_path: Path) -> None:
    fields = [
        "rank",
        "episode_key",
        "dataset",
        "episode_id",
        "total_score",
        *[f"score_{name}" for name in SIGNAL_ORDER],
        *[f"contrib_{name}" for name in SIGNAL_ORDER],
        "mean_speed",
        "path_length",
        "duration",
        "displacement",
        "reason",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, score in enumerate(scores, start=1):
            row = {
                "rank": rank,
                "episode_key": score.feature.key,
                "dataset": score.feature.dataset,
                "episode_id": score.feature.episode_id,
                "total_score": score.total_score,
                "mean_speed": score.feature.mean_speed,
                "path_length": score.feature.path_length,
                "duration": score.feature.duration,
                "displacement": score.feature.displacement,
                "reason": score.reason,
            }
            row.update({f"score_{name}": score.signal_scores.get(name, 0.0) for name in SIGNAL_ORDER})
            row.update({f"contrib_{name}": score.contributions.get(name, 0.0) for name in SIGNAL_ORDER})
            writer.writerow(row)


def _write_report_text(
    name: str,
    scores: Sequence[EpisodeScore],
    train_scores: Sequence[EpisodeScore],
    out_path: Path,
) -> None:
    train_values = np.asarray([score.total_score for score in train_scores], dtype=np.float64)
    lines = [
        f"OOD Episode Inspector: {name}",
        "",
        f"Train average score: {float(np.mean(train_values)):.4f}" if train_values.size else "Train average score: n/a",
        f"Train p95 score: {float(np.percentile(train_values, 95)):.4f}" if train_values.size else "Train p95 score: n/a",
        "",
        "Top OOD episodes:",
    ]
    for idx, score in enumerate(scores[:20], start=1):
        lines.append(
            f"{idx:02d}. {score.feature.key} score={score.total_score:.4f} - {score.reason}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_ranked_scores(
    scores: Sequence[EpisodeScore],
    train_scores: Sequence[EpisodeScore],
    top_k: int,
    out_path: Path,
) -> None:
    shown = list(scores[:top_k])
    if not shown:
        return
    labels = [score.feature.key for score in shown]
    y = np.arange(len(shown))
    fig, ax = plt.subplots(figsize=(11.5, max(5.0, 0.36 * len(shown) + 1.8)))
    left = np.zeros(len(shown), dtype=np.float64)
    colors = {
        "trajectory": "#2563eb",
        "velocity": "#f97316",
        "initial_pose": "#16a34a",
        "image_embedding": "#9333ea",
        "camera_pose": "#0891b2",
    }
    for name in SIGNAL_ORDER:
        values = np.asarray([score.contributions.get(name, 0.0) for score in shown], dtype=np.float64)
        ax.barh(y, values, left=left, color=colors[name], alpha=0.86, label=name)
        left += values
    train_total = np.asarray([score.total_score for score in train_scores], dtype=np.float64)
    if train_total.size:
        ax.axvline(
            float(np.mean(train_total)),
            color="#facc15",
            linewidth=2.8,
            label="train avg score",
        )
        ax.axvline(
            float(np.percentile(train_total, 95)),
            color="#facc15",
            linestyle="--",
            linewidth=2.2,
            label="train p95 score",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Weighted signal contribution")
    ax.set_title("Top OOD Episodes by Weighted Signal Score")
    ax.grid(axis="x", color="#dddddd", alpha=0.45)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_score_distribution(
    test_scores: Sequence[EpisodeScore],
    train_scores: Sequence[EpisodeScore],
    out_path: Path,
) -> None:
    train_values = np.asarray([score.total_score for score in train_scores], dtype=np.float64)
    test_values = np.asarray([score.total_score for score in test_scores], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    if train_values.size:
        ax.hist(train_values, bins=30, alpha=0.52, density=True, color="#2563eb", label="train episodes")
        ax.axvline(float(np.mean(train_values)), color="#1d4ed8", linewidth=1.2, label="train avg")
        ax.axvline(float(np.percentile(train_values, 95)), color="#1d4ed8", linestyle="--", linewidth=1.0, label="train p95")
    if test_values.size:
        ax.hist(test_values, bins=30, alpha=0.52, density=True, color="#f97316", label="test episodes")
        ax.axvline(float(np.mean(test_values)), color="#c2410c", linewidth=1.2, label="test avg")
    ax.set_title("Episode OOD Score Distribution")
    ax.set_xlabel("Weighted normalized OOD score")
    ax.set_ylabel("Density")
    ax.grid(axis="y", color="#dddddd", alpha=0.45)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_reason_heatmap(
    scores: Sequence[EpisodeScore],
    train_scores: Sequence[EpisodeScore],
    top_k: int,
    out_path: Path,
) -> None:
    shown = list(scores[:top_k])
    if not shown:
        return
    train_avg = [
        float(np.mean([score.signal_scores.get(name, 0.0) for score in train_scores]))
        if train_scores
        else 0.0
        for name in SIGNAL_ORDER
    ]
    rows = [train_avg] + [
        [score.signal_scores.get(name, 0.0) for name in SIGNAL_ORDER]
        for score in shown
    ]
    labels = ["train avg"] + [score.feature.key for score in shown]
    arr = np.asarray(rows, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9.0, max(5.0, 0.34 * len(labels) + 1.8)))
    im = ax.imshow(arr, aspect="auto", cmap="magma", vmin=0.0)
    ax.set_xticks(np.arange(len(SIGNAL_ORDER)))
    ax.set_xticklabels(SIGNAL_ORDER, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Per-Episode Reason Signals (Train Average Included)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Normalized signal score")
    for y_idx in range(arr.shape[0]):
        for x_idx in range(arr.shape[1]):
            ax.text(
                x_idx,
                y_idx,
                f"{arr[y_idx, x_idx]:.2f}",
                ha="center",
                va="center",
                color="white" if arr[y_idx, x_idx] > np.nanmax(arr) * 0.45 else "#222222",
                fontsize=7,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_metric_comparison(
    scores: Sequence[EpisodeScore],
    reference: TrainReference,
    top_k: int,
    out_path: Path,
) -> None:
    shown = list(scores[: min(top_k, 10)])
    if not shown:
        return
    metrics = ("path_length", "duration", "mean_speed", "speed_p95", "displacement")
    metric_index = {name: idx for idx, name in enumerate(reference.velocity_names)}
    x = np.arange(len(metrics))
    width = min(0.11, 0.72 / max(len(shown), 1))
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.axhline(0.0, color="#222222", linewidth=1.0, label="train avg")
    for idx, score in enumerate(shown):
        values = []
        for metric in metrics:
            m_idx = metric_index[metric]
            raw = getattr(score.feature, metric)
            values.append((raw - reference.velocity_mean[m_idx]) / reference.velocity_std[m_idx])
        offset = (idx - (len(shown) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, alpha=0.78, label=score.feature.key)
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
    ax.axhline(-1.0, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_ylabel("(episode value - train avg) / train std")
    ax.set_title("Top Episode Motion Metrics vs Average Train Value")
    ax.grid(axis="y", color="#dddddd", alpha=0.45)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_outputs(
    name: str,
    output_dir: Path,
    train_features: Sequence[EpisodeFeatures],
    test_features: Sequence[EpisodeFeatures],
    train_scores: Sequence[EpisodeScore],
    test_scores: Sequence[EpisodeScore],
    reference: TrainReference,
    weights: Dict[str, float],
    mode: str,
    top_k: int,
    config_summary: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = tuple(sorted(test_scores, key=lambda score: score.total_score, reverse=True))
    ranked_train = tuple(sorted(train_scores, key=lambda score: score.total_score, reverse=True))

    _write_scores_csv(ranked, output_dir / "episode_scores.csv")
    _write_scores_csv(ranked_train, output_dir / "train_episode_scores.csv")
    _write_report_text(name, ranked, train_scores, output_dir / "episode_report.txt")
    _plot_ranked_scores(ranked, train_scores, top_k, output_dir / "ranked_episode_scores.png")
    _plot_score_distribution(ranked, train_scores, output_dir / "score_distribution.png")
    _plot_reason_heatmap(ranked, train_scores, top_k, output_dir / "reason_signal_heatmap.png")
    _plot_metric_comparison(ranked, reference, top_k, output_dir / "motion_metric_comparison.png")

    train_total = np.asarray([score.total_score for score in train_scores], dtype=np.float64)
    test_total = np.asarray([score.total_score for score in test_scores], dtype=np.float64)
    summary = {
        "name": name,
        "mode": mode,
        "weights": weights,
        "train_episode_count": len(train_features),
        "test_episode_count": len(test_features),
        "train_score_mean": float(np.mean(train_total)) if train_total.size else None,
        "train_score_p95": float(np.percentile(train_total, 95)) if train_total.size else None,
        "test_score_mean": float(np.mean(test_total)) if test_total.size else None,
        "test_score_p95": float(np.percentile(test_total, 95)) if test_total.size else None,
        "top_episodes": [_score_to_dict(score) for score in ranked[:top_k]],
        "config": config_summary,
        "outputs": {
            "episode_scores_csv": str(output_dir / "episode_scores.csv"),
            "train_episode_scores_csv": str(output_dir / "train_episode_scores.csv"),
            "episode_report": str(output_dir / "episode_report.txt"),
            "ranked_episode_scores": str(output_dir / "ranked_episode_scores.png"),
            "score_distribution": str(output_dir / "score_distribution.png"),
            "reason_signal_heatmap": str(output_dir / "reason_signal_heatmap.png"),
            "motion_metric_comparison": str(output_dir / "motion_metric_comparison.png"),
        },
    }
    with (output_dir / "episode_inspector_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def inspect_random_split_dataset(
    dataset_root: Path,
    args: argparse.Namespace,
    output_root: Path,
    weights: Dict[str, float],
) -> Dict[str, Any]:
    config, episode_data = _load_dataset_trajectory(dataset_root, args)
    available_ids = _select_episode_ids(episode_data.keys(), args.max_episodes, args.seed)
    split = _random_split(available_ids, args.train_ratio, args.seed)
    train_features = _build_episode_features(
        dataset_root,
        episode_data,
        split.train,
        args.image_output_dir.resolve(),
    )
    test_features = _build_episode_features(
        dataset_root,
        episode_data,
        split.test,
        args.image_output_dir.resolve(),
    )
    reference = _make_train_reference(train_features)
    train_scores = _score_features(train_features, reference, weights, exclude_self=True)
    test_scores = _score_features(test_features, reference, weights)
    name = f"{dataset_root.name}_episode_split"
    config_summary = {
        "dataset_root": str(dataset_root),
        "state_key": config.state_key,
        "episode_key": config.episode_key,
        "timestamp_key": config.timestamp_key,
        "coord_indices": list(config.coord_indices),
        "coord_labels": list(config.coord_labels),
        "gripper_index": config.gripper_index,
        "split": {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "train_episodes": list(split.train),
            "test_episodes": list(split.test),
        },
    }
    return _write_outputs(
        name=name,
        output_dir=output_root / dataset_root.name,
        train_features=train_features,
        test_features=test_features,
        train_scores=train_scores,
        test_scores=test_scores,
        reference=reference,
        weights=weights,
        mode="random_episode_split",
        top_k=args.top_k,
        config_summary=config_summary,
    )


def inspect_explicit_train_test(
    args: argparse.Namespace,
    output_root: Path,
    weights: Dict[str, float],
) -> Dict[str, Any]:
    if args.train_dataset_root is None or args.test_dataset_root is None:
        raise ValueError("Both --train-dataset-root and --test-dataset-root are required.")
    train_root = args.train_dataset_root.resolve()
    test_root = args.test_dataset_root.resolve()
    train_config, train_episode_data = _load_dataset_trajectory(train_root, args)
    test_config, test_episode_data = _load_dataset_trajectory(test_root, args)
    train_ids = _select_episode_ids(train_episode_data.keys(), args.max_episodes, args.seed)
    test_ids = _select_episode_ids(test_episode_data.keys(), args.max_episodes, args.seed + 17)
    train_features = _build_episode_features(
        train_root,
        train_episode_data,
        train_ids,
        args.image_output_dir.resolve(),
    )
    test_features = _build_episode_features(
        test_root,
        test_episode_data,
        test_ids,
        args.image_output_dir.resolve(),
    )
    reference = _make_train_reference(train_features)
    train_scores = _score_features(train_features, reference, weights, exclude_self=True)
    test_scores = _score_features(test_features, reference, weights)
    name = f"{train_root.name}_train_vs_{test_root.name}_test"
    config_summary = {
        "train_dataset_root": str(train_root),
        "test_dataset_root": str(test_root),
        "train_config": {
            "state_key": train_config.state_key,
            "episode_key": train_config.episode_key,
            "timestamp_key": train_config.timestamp_key,
            "coord_indices": list(train_config.coord_indices),
            "coord_labels": list(train_config.coord_labels),
            "gripper_index": train_config.gripper_index,
        },
        "test_config": {
            "state_key": test_config.state_key,
            "episode_key": test_config.episode_key,
            "timestamp_key": test_config.timestamp_key,
            "coord_indices": list(test_config.coord_indices),
            "coord_labels": list(test_config.coord_labels),
            "gripper_index": test_config.gripper_index,
        },
        "train_episodes": [int(ep) for ep in train_ids.tolist()],
        "test_episodes": [int(ep) for ep in test_ids.tolist()],
    }
    return _write_outputs(
        name=name,
        output_dir=output_root / name,
        train_features=train_features,
        test_features=test_features,
        train_scores=train_scores,
        test_scores=test_scores,
        reference=reference,
        weights=weights,
        mode="explicit_train_test_roots",
        top_k=args.top_k,
        config_summary=config_summary,
    )


def _resolve_dataset_roots(args: argparse.Namespace) -> List[Path]:
    roots = discover_dataset_roots(args.dataset_root.resolve(), args.all_datasets)
    if args.dataset_name:
        wanted = set(args.dataset_name)
        roots = [root for root in roots if root.name in wanted]
        missing = sorted(wanted - {root.name for root in roots})
        if missing:
            raise FileNotFoundError(f"Requested dataset name(s) not found: {missing}")
    return roots


def _preset_libero_droid(args: argparse.Namespace, weights: Dict[str, float]) -> List[Dict[str, Any]]:
    root = args.dataset_root.resolve()
    output_root = args.output_dir.resolve()
    libero_root = root / args.libero_name
    droid_root = root / args.droid_name
    if not libero_root.is_dir():
        raise FileNotFoundError(f"LIBERO dataset not found: {libero_root}")
    if not droid_root.is_dir():
        raise FileNotFoundError(f"DROID dataset not found: {droid_root}")

    summaries: List[Dict[str, Any]] = []
    for dataset_root in (libero_root, droid_root):
        print(f"Inspecting same-dataset split: {dataset_root.name}")
        summaries.append(inspect_random_split_dataset(dataset_root, args, output_root / "same_dataset_splits", weights))

    explicit_args = argparse.Namespace(**vars(args))
    explicit_args.train_dataset_root = libero_root
    explicit_args.test_dataset_root = droid_root
    print(f"Inspecting cross-dataset: {libero_root.name} train vs {droid_root.name} test")
    summaries.append(inspect_explicit_train_test(explicit_args, output_root / "cross_dataset", weights))
    return summaries


def main() -> None:
    args = parse_args()
    weights = _parse_weights(args.weights)
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.preset == "libero_droid":
        summaries = _preset_libero_droid(args, weights)
    elif args.train_dataset_root is not None or args.test_dataset_root is not None:
        summaries = [inspect_explicit_train_test(args, output_root, weights)]
    else:
        summaries = []
        for dataset_root in _resolve_dataset_roots(args):
            print(f"Inspecting episode split: {dataset_root.name}")
            summaries.append(inspect_random_split_dataset(dataset_root.resolve(), args, output_root, weights))

    merged = {
        "output_dir": str(output_root),
        "summary_count": len(summaries),
        "summaries": summaries,
    }
    with (output_root / "summary_all.json").open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"Saved episode inspector outputs to: {output_root}")
    for summary in summaries:
        print(f"- {summary['name']}: {summary['outputs']['episode_report']}")


if __name__ == "__main__":
    main()
