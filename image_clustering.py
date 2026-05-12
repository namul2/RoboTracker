#!/usr/bin/env python3
"""Image-first environment distribution analyzer for LeRobot datasets.

The tool samples the first image frame of each episode, embeds it with a DINOv3
vision backbone, clusters the embeddings, and writes 2D embedding plots plus a
camera-pose relationship map.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing base dependencies. Run: python3 -m pip install -r requirements.txt"
    ) from exc


DEFAULT_DINOV3_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


@dataclass
class ImageSample:
    dataset: str
    dataset_root: Path
    image_key: str
    episode_index: int
    frame_index: int
    parquet_file: Path
    local_row: int
    image: Image.Image
    cache_path: Optional[Path] = None

    @property
    def label(self) -> str:
        short_key = self.image_key.split(".")[-1]
        return f"{self.dataset}/ep{self.episode_index}/{short_key}"


@dataclass(frozen=True)
class CameraPose:
    key: str
    position: np.ndarray
    target: np.ndarray
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze environment distribution from first image frames in "
            "LeRobot-format datasets."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Path to a LeRobot dataset root, or a parent directory.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Analyze every child directory that looks like a LeRobot dataset.",
    )
    parser.add_argument(
        "--image-key",
        action="append",
        default=None,
        help=(
            "Image/video feature key to analyze. Repeat for multiple keys. "
            "Default: all image/video features in meta/info.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/image_distribution"),
        help="Directory to store plots, embeddings, and summaries.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_DINOV3_MODEL,
        help="Hugging Face model id or local path for DINOv3.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Embedding device: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="DINOv3 embedding batch size.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on sampled episodes per dataset.",
    )
    parser.add_argument(
        "--cluster-count",
        type=int,
        default=None,
        help="Number of KMeans clusters. Default: auto from sample count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for UMAP/KMeans and episode subsampling.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Only extract first frames and camera pose maps; skip DINO/UMAP.",
    )
    parser.add_argument(
        "--video-backend",
        choices=("pyav", "opencv", "auto"),
        default="pyav",
        help=(
            "Backend for video-backed LeRobot datasets. Default: pyav, which is "
            "usually quieter and more reliable for AV1 than OpenCV."
        ),
    )
    parser.add_argument(
        "--camera-pose-json",
        type=Path,
        default=None,
        help=(
            "Optional camera extrinsics JSON. Maps image keys to "
            "{position: [x,y,z], target: [x,y,z]} in robot-base coordinates."
        ),
    )
    parser.add_argument(
        "--thumbnail-size",
        type=int,
        default=128,
        help="Representative image size in the cluster contact sheet.",
    )
    parser.add_argument(
        "--no-image-cache",
        action="store_true",
        help="Do not save/reuse extracted first-frame images under the output directory.",
    )
    return parser.parse_args()


def read_dataset_info(dataset_root: Path) -> Dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_dataset_roots(root: Path, all_datasets: bool) -> List[Path]:
    if (root / "data").is_dir() and (root / "meta").is_dir():
        return [root]

    children = [
        child
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / "data").is_dir() and (child / "meta").is_dir()
    ]
    if all_datasets or children:
        if not children:
            raise FileNotFoundError(f"No LeRobot dataset roots found under: {root}")
        return children

    raise FileNotFoundError(
        f"{root} is not a LeRobot dataset root. Use --all-datasets for a parent directory."
    )


def list_data_files(dataset_root: Path) -> List[Path]:
    files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {dataset_root / 'data'}")
    return files


def image_feature_keys(info: Dict[str, Any], explicit: Optional[Sequence[str]]) -> List[str]:
    features = info.get("features", {})
    available = [
        key
        for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") in {"image", "video"}
    ]
    if explicit:
        missing = [key for key in explicit if key not in available]
        if missing:
            raise KeyError(f"Image key(s) not found in dataset metadata: {missing}")
        return list(explicit)
    return available


def _scalar_column(table: "pq.Table", key: str, dtype: Any) -> np.ndarray:
    return np.asarray(table[key].to_pylist(), dtype=dtype)


def _first_rows_by_episode(
    parquet_file: Path,
    max_episodes: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(
        parquet_file,
        columns=["episode_index", "frame_index"],
        use_threads=False,
    )
    episodes = _scalar_column(table, "episode_index", np.int64)
    frames = _scalar_column(table, "frame_index", np.int64)

    first_local_rows: Dict[int, int] = {}
    for row, episode in enumerate(episodes):
        first_local_rows.setdefault(int(episode), row)

    episode_ids = np.asarray(sorted(first_local_rows), dtype=np.int64)
    if max_episodes is not None and episode_ids.size > max_episodes:
        rng = np.random.default_rng(seed)
        episode_ids = np.sort(rng.choice(episode_ids, size=max_episodes, replace=False))

    local_rows = np.asarray([first_local_rows[int(ep)] for ep in episode_ids], dtype=np.int64)
    frame_ids = frames[local_rows]
    return episode_ids, frame_ids, local_rows


def _image_from_cell(cell: Any, dataset_root: Path) -> Image.Image:
    if isinstance(cell, dict):
        raw_bytes = cell.get("bytes")
        image_path = cell.get("path")
    else:
        raw_bytes = None
        image_path = cell

    if raw_bytes is not None:
        image = Image.open(io.BytesIO(raw_bytes))
    elif image_path:
        path = Path(str(image_path))
        if not path.is_absolute():
            path = dataset_root / path
        image = Image.open(path)
    else:
        raise ValueError("Image cell has neither bytes nor path.")
    return image.convert("RGB")


def _chunk_file_parts(parquet_file: Path) -> Tuple[str, str]:
    return parquet_file.parent.name, parquet_file.with_suffix(".mp4").name


def _video_path(dataset_root: Path, image_key: str, parquet_file: Path) -> Path:
    chunk_name, mp4_name = _chunk_file_parts(parquet_file)
    return dataset_root / "videos" / image_key / chunk_name / mp4_name


@contextmanager
def _suppress_native_stderr() -> Iterator[None]:
    """Temporarily silence native FFmpeg decoder logs emitted below Python."""
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)


def _video_frame_cv2(path: Path, frame_number: int) -> Image.Image:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required for video-backed datasets.") from exc

    with _suppress_native_stderr():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ok, frame = cap.read()
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode frame {frame_number} from {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame).convert("RGB")


def _video_frames_cv2(path: Path, frame_numbers: Sequence[int]) -> Dict[int, Image.Image]:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required for video-backed datasets.") from exc

    unique_frames = sorted(set(int(frame_number) for frame_number in frame_numbers))
    images: Dict[int, Image.Image] = {}
    with _suppress_native_stderr():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        try:
            for frame_number in unique_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Could not decode frame {frame_number} from {path}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images[frame_number] = Image.fromarray(frame).convert("RGB")
        finally:
            cap.release()
    return images


def _video_frame_av(path: Path, frame_number: int) -> Image.Image:
    try:
        import av
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyAV is required for AV fallback video decoding.") from exc

    if hasattr(av, "logging") and hasattr(av.logging, "PANIC"):
        av.logging.set_level(av.logging.PANIC)

    with _suppress_native_stderr():
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                if index == int(frame_number):
                    return frame.to_image().convert("RGB")
    raise RuntimeError(f"Could not decode frame {frame_number} from {path}")


def _video_frames_av(path: Path, frame_numbers: Sequence[int]) -> Dict[int, Image.Image]:
    try:
        import av
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyAV is required for AV fallback video decoding.") from exc

    if hasattr(av, "logging") and hasattr(av.logging, "PANIC"):
        av.logging.set_level(av.logging.PANIC)

    targets = sorted(set(int(frame_number) for frame_number in frame_numbers))
    if not targets:
        return {}
    remaining = set(targets)
    images: Dict[int, Image.Image] = {}
    max_target = targets[-1]

    with _suppress_native_stderr():
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                if index in remaining:
                    images[index] = frame.to_image().convert("RGB")
                    remaining.remove(index)
                    if not remaining:
                        break
                if index > max_target:
                    break

    if remaining:
        missing = ", ".join(str(frame) for frame in sorted(remaining)[:8])
        suffix = "..." if len(remaining) > 8 else ""
        raise RuntimeError(f"Could not decode frame(s) {missing}{suffix} from {path}")
    return images


def _video_frame(path: Path, frame_number: int, backend: str) -> Image.Image:
    if backend == "pyav":
        return _video_frame_av(path, frame_number)
    if backend == "opencv":
        return _video_frame_cv2(path, frame_number)

    av_error: Optional[Exception] = None
    try:
        return _video_frame_av(path, frame_number)
    except Exception as exc:  # pragma: no cover - backend dependent
        av_error = exc
    try:
        return _video_frame_cv2(path, frame_number)
    except Exception as cv2_error:  # pragma: no cover - backend dependent
        raise RuntimeError(f"PyAV failed: {av_error}; OpenCV failed: {cv2_error}") from cv2_error


def _video_frames(path: Path, frame_numbers: Sequence[int], backend: str) -> Dict[int, Image.Image]:
    if backend == "pyav":
        return _video_frames_av(path, frame_numbers)
    if backend == "opencv":
        return _video_frames_cv2(path, frame_numbers)

    av_error: Optional[Exception] = None
    try:
        return _video_frames_av(path, frame_numbers)
    except Exception as exc:  # pragma: no cover - backend dependent
        av_error = exc
    try:
        return _video_frames_cv2(path, frame_numbers)
    except Exception as cv2_error:  # pragma: no cover - backend dependent
        raise RuntimeError(f"PyAV failed: {av_error}; OpenCV failed: {cv2_error}") from cv2_error


def load_first_frame_samples(
    dataset_root: Path,
    image_keys: Sequence[str],
    max_episodes: Optional[int],
    seed: int,
    video_backend: str,
) -> Tuple[List[ImageSample], List[str]]:
    info = read_dataset_info(dataset_root)
    features = info.get("features", {})
    parquet_files = list_data_files(dataset_root)
    samples: List[ImageSample] = []
    warnings_out: List[str] = []

    for parquet_file in parquet_files:
        episode_ids, frame_ids, local_rows = _first_rows_by_episode(
            parquet_file, max_episodes=max_episodes, seed=seed
        )
        if episode_ids.size == 0:
            continue

        for image_key in image_keys:
            feature = features.get(image_key, {})
            dtype = feature.get("dtype")
            if dtype == "image":
                table = pq.read_table(parquet_file, columns=[image_key], use_threads=False)
                column = table[image_key]
                for episode, frame, local_row in zip(episode_ids, frame_ids, local_rows):
                    try:
                        image = _image_from_cell(column[int(local_row)].as_py(), dataset_root)
                    except Exception as exc:
                        warnings_out.append(
                            f"{dataset_root.name}:{image_key}: skipped episode {episode}: {exc}"
                        )
                        continue
                    samples.append(
                        ImageSample(
                            dataset=dataset_root.name,
                            dataset_root=dataset_root,
                            image_key=image_key,
                            episode_index=int(episode),
                            frame_index=int(frame),
                            parquet_file=parquet_file,
                            local_row=int(local_row),
                            image=image,
                        )
                    )
            elif dtype == "video":
                path = _video_path(dataset_root, image_key, parquet_file)
                try:
                    images_by_row = _video_frames(path, local_rows.tolist(), video_backend)
                except Exception as exc:
                    for episode in episode_ids:
                        warnings_out.append(
                            f"{dataset_root.name}:{image_key}: skipped episode {episode}: {exc}"
                        )
                    continue
                for episode, frame, local_row in zip(episode_ids, frame_ids, local_rows):
                    try:
                        image = images_by_row[int(local_row)]
                    except Exception as exc:
                        warnings_out.append(
                            f"{dataset_root.name}:{image_key}: skipped episode {episode}: {exc}"
                        )
                        continue
                    samples.append(
                        ImageSample(
                            dataset=dataset_root.name,
                            dataset_root=dataset_root,
                            image_key=image_key,
                            episode_index=int(episode),
                            frame_index=int(frame),
                            parquet_file=parquet_file,
                            local_row=int(local_row),
                            image=image,
                        )
                    )
    return samples, warnings_out


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        return "cpu"


def compute_dinov3_embeddings(
    samples: Sequence[ImageSample],
    model_name: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "DINOv3 embedding requires torch, torchvision, and transformers. "
            "Install torch/torchvision with the command that matches your CUDA driver, "
            "then run: python3 -m pip install -r requirements.txt"
        ) from exc
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is installed but cannot be imported. This usually means the "
            "torch/torchvision/MKL packages in the active env are ABI-incompatible. "
            "Remove mixed pip/conda torch installs and reinstall a matched torch stack."
        ) from exc

    resolved_device = _resolve_device(device)
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(resolved_device)
    except ImportError as exc:
        raise SystemExit(
            "DINOv3 image preprocessing requires torchvision. Install a torchvision "
            "version that matches your PyTorch build."
        ) from exc
    model.eval()

    chunks: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = [sample.image for sample in samples[start:start + batch_size]]
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(resolved_device) for key, value in inputs.items()}
            outputs = model(**inputs)
            if getattr(outputs, "pooler_output", None) is not None:
                embedding = outputs.pooler_output
            else:
                hidden = outputs.last_hidden_state
                embedding = hidden[:, 0] if hidden.ndim == 3 else hidden.mean(dim=1)
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
            chunks.append(embedding.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def reduce_embeddings(embeddings: np.ndarray, seed: int) -> Tuple[np.ndarray, str]:
    if embeddings.shape[0] < 2:
        return np.zeros((embeddings.shape[0], 2), dtype=np.float64), "none"
    if embeddings.shape[0] < 12:
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(embeddings), "pca"
    try:
        import umap

        n_neighbors = min(15, max(2, embeddings.shape[0] - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.03,
            spread=0.85,
            metric="cosine",
            random_state=seed,
        )
        return reducer.fit_transform(embeddings), "umap"
    except ModuleNotFoundError:
        from sklearn.decomposition import PCA

        warnings.warn("umap-learn is not installed; using PCA fallback.", RuntimeWarning)
        return PCA(n_components=2, random_state=seed).fit_transform(embeddings), "pca"


def cluster_embeddings(
    embeddings: np.ndarray,
    cluster_count: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, int]:
    if embeddings.shape[0] < 2:
        return np.zeros((embeddings.shape[0],), dtype=np.int64), 1

    from sklearn.cluster import KMeans

    if cluster_count is None:
        cluster_count = max(2, min(8, int(round(math.sqrt(embeddings.shape[0] / 2)))))
    cluster_count = max(1, min(int(cluster_count), embeddings.shape[0]))
    labels = KMeans(n_clusters=cluster_count, n_init="auto", random_state=seed).fit_predict(
        embeddings
    )
    return labels.astype(np.int64), cluster_count


def _color_values(items: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    unique = sorted(set(items))
    lookup = {value: idx for idx, value in enumerate(unique)}
    return np.asarray([lookup[item] for item in items], dtype=np.int64), unique


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Return a 2D convex hull using Andrew's monotonic chain algorithm."""
    if points.shape[0] <= 2:
        return points
    ordered = sorted(set((float(x), float(y)) for x, y in points))
    if len(ordered) <= 2:
        return np.asarray(ordered, dtype=np.float64)

    def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: List[Tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _padded_limits(coords: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = span * 0.12
    return (float(mins[0] - pad[0]), float(maxs[0] + pad[0])), (
        float(mins[1] - pad[1]),
        float(maxs[1] + pad[1]),
    )


def plot_embedding(
    coords: np.ndarray,
    samples: Sequence[ImageSample],
    labels: np.ndarray,
    out_path: Path,
    title: str,
    color_mode: str,
) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(8.8, 6.8), facecolor="white")
    ax.set_facecolor("white")
    if color_mode == "cluster":
        values = labels
        legend_names = [f"cluster {i}" for i in sorted(set(labels.tolist()))]
        color_indices = sorted(set(labels.tolist()))
    elif color_mode == "dataset":
        values, legend_names = _color_values([sample.dataset for sample in samples])
        color_indices = list(range(len(legend_names)))
    else:
        values, legend_names = _color_values([sample.image_key for sample in samples])
        color_indices = list(range(len(legend_names)))

    cmap = plt.get_cmap("tab20" if len(legend_names) > 10 else "tab10")
    palette = {
        int(value): cmap(i % cmap.N)
        for i, value in enumerate(color_indices)
    }

    handles: List[Line2D] = []
    for legend_idx, legend_name in enumerate(legend_names):
        group_value = color_indices[legend_idx]
        mask = values == group_value
        group_coords = coords[mask]
        color = palette[int(group_value)]
        if group_coords.shape[0] >= 3:
            hull = _convex_hull(group_coords)
            ax.add_patch(
                Polygon(
                    hull,
                    closed=True,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.10,
                    linewidth=1.15,
                    zorder=1,
                )
            )
        elif group_coords.shape[0] == 2:
            ax.plot(
                group_coords[:, 0],
                group_coords[:, 1],
                color=color,
                alpha=0.22,
                linewidth=2.0,
                zorder=1,
            )

        ax.scatter(
            group_coords[:, 0],
            group_coords[:, 1],
            marker="D",
            s=30 if coords.shape[0] > 80 else 46,
            c=[color],
            alpha=0.88,
            edgecolors="#fbfaf7",
            linewidths=0.75,
            zorder=3,
        )
        if color_mode == "cluster" and group_coords.size:
            centroid = group_coords.mean(axis=0)
            ax.scatter(
                [centroid[0]],
                [centroid[1]],
                marker="+",
                s=120,
                c=["#262626"],
                linewidths=1.3,
                alpha=0.72,
                zorder=4,
            )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=color,
                markeredgecolor="#fbfaf7",
                markersize=7,
                label=legend_name,
            )
        )

    ax.set_title(title, fontsize=13, weight="semibold", pad=12)
    ax.set_xlabel("Embedding dim 1", fontsize=9)
    ax.set_ylabel("Embedding dim 2", fontsize=9)
    ax.grid(True, color="#d8d6cf", alpha=0.42, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c6c2b8")
    ax.tick_params(colors="#6b675f", labelsize=8)
    if coords.shape[0] > 0:
        xlim, ylim = _padded_limits(coords)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    ax.legend(
        handles=handles,
        loc="best",
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor="#d9d5ca",
        framealpha=0.92,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _make_thumb(image: Image.Image, size: int, caption: str) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size + 30), "#f7f7f4")
    x = (size - thumb.width) // 2
    y = (size - thumb.height) // 2
    canvas.paste(thumb, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, size, size, size + 30), fill="#262626")
    draw.text((6, size + 7), caption[:26], fill="#f7f7f4")
    return canvas


def save_contact_sheet(
    samples: Sequence[ImageSample],
    coords: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    thumb_size: int,
    max_per_cluster: int = 8,
) -> None:
    if not samples:
        return
    tiles: List[Image.Image] = []
    for cluster_id in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == cluster_id)
        if indices.size == 0:
            continue
        centroid = coords[indices].mean(axis=0)
        ranked = sorted(indices.tolist(), key=lambda i: float(np.linalg.norm(coords[i] - centroid)))
        for idx in ranked[:max_per_cluster]:
            tiles.append(
                _make_thumb(
                    samples[idx].image,
                    thumb_size,
                    f"c{cluster_id} {samples[idx].dataset} e{samples[idx].episode_index}",
                )
            )

    gap = 10
    tile_h = thumb_size + 30
    columns = min(8, max(1, len(tiles)))
    rows = int(math.ceil(len(tiles) / columns))
    sheet = Image.new(
        "RGB",
        (columns * thumb_size + (columns + 1) * gap, rows * tile_h + (rows + 1) * gap),
        "#efede7",
    )
    for idx, tile in enumerate(tiles):
        x = gap + (idx % columns) * (thumb_size + gap)
        y = gap + (idx // columns) * (tile_h + gap)
        sheet.paste(tile, (x, y))
    sheet.save(out_path)


def write_embedding_outputs(
    samples: Sequence[ImageSample],
    embeddings: np.ndarray,
    out_dir: Path,
    title_prefix: str,
    seed: int,
    cluster_count: Optional[int],
    thumbnail_size: int,
    include_dataset_plot: bool,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    coords, reducer_name = reduce_embeddings(embeddings, seed)
    labels, resolved_cluster_count = cluster_embeddings(embeddings, cluster_count, seed)

    np.savez_compressed(
        out_dir / "embeddings.npz",
        embeddings=embeddings,
        coords_2d=coords,
        cluster_labels=labels,
        dataset=np.asarray([sample.dataset for sample in samples]),
        image_key=np.asarray([sample.image_key for sample in samples]),
        episode_index=np.asarray([sample.episode_index for sample in samples]),
    )
    save_samples_json(samples, out_dir / "samples_embedding.json")
    plot_embedding(
        coords,
        samples,
        labels,
        out_dir / "embedding_clusters.png",
        f"{title_prefix} DINOv3 {reducer_name.upper()} by Cluster",
        "cluster",
    )
    plot_embedding(
        coords,
        samples,
        labels,
        out_dir / "embedding_cameras.png",
        f"{title_prefix} DINOv3 {reducer_name.upper()} by Camera",
        "camera",
    )
    outputs = {
        "embeddings": str(out_dir / "embeddings.npz"),
        "clusters": str(out_dir / "embedding_clusters.png"),
        "cameras": str(out_dir / "embedding_cameras.png"),
        "contact_sheet": str(out_dir / "cluster_contact_sheet.png"),
    }
    if include_dataset_plot:
        plot_embedding(
            coords,
            samples,
            labels,
            out_dir / "embedding_datasets.png",
            f"{title_prefix} DINOv3 {reducer_name.upper()} by Dataset",
            "dataset",
        )
        outputs["datasets"] = str(out_dir / "embedding_datasets.png")

    save_contact_sheet(
        samples,
        coords,
        labels,
        out_dir / "cluster_contact_sheet.png",
        thumbnail_size,
    )
    summary = {
        "reducer": reducer_name,
        "cluster_count": int(resolved_cluster_count),
        "sample_count": len(samples),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "outputs": outputs,
    }
    with (out_dir / "embedding_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _load_camera_pose_json(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--camera-pose-json must contain an object at the top level.")
    return data


def _heuristic_pose(key: str, ordinal: int, total: int) -> Tuple[np.ndarray, np.ndarray]:
    lowered = key.lower()
    if "top" in lowered:
        return np.array([0.0, 0.0, 1.25]), np.array([0.0, 0.0, 0.0])
    if "wrist" in lowered:
        return np.array([0.28, -0.28, 0.42]), np.array([0.05, 0.0, 0.08])
    if "exterior_image_2" in lowered or "side_2" in lowered:
        return np.array([-0.78, -0.58, 0.52]), np.array([0.0, 0.0, 0.18])
    if "exterior" in lowered or lowered.endswith(".image"):
        return np.array([0.78, -0.58, 0.52]), np.array([0.0, 0.0, 0.18])

    angle = 2 * math.pi * ordinal / max(total, 1)
    return (
        np.array([0.8 * math.cos(angle), 0.8 * math.sin(angle), 0.55]),
        np.array([0.0, 0.0, 0.1]),
    )


def resolve_camera_poses(
    image_keys: Sequence[str],
    pose_config: Dict[str, Dict[str, Any]],
) -> List[CameraPose]:
    poses: List[CameraPose] = []
    for ordinal, key in enumerate(image_keys):
        entry = pose_config.get(key)
        if entry:
            position = np.asarray(entry["position"], dtype=np.float64)
            target = np.asarray(entry.get("target", [0.0, 0.0, 0.0]), dtype=np.float64)
            source = "camera-pose-json"
        else:
            position, target = _heuristic_pose(key, ordinal, len(image_keys))
            source = "heuristic-layout"
        poses.append(CameraPose(key=key, position=position, target=target, source=source))
    return poses


def _basis_from_pose(position: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = target - position
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        forward = np.array([0.0, 0.0, -1.0])
    else:
        forward = forward / norm

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, world_up))) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right = right / max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, forward)
    up = up / max(np.linalg.norm(up), 1e-9)
    return forward, right, up


def plot_camera_pose_map(
    poses: Sequence[CameraPose],
    dataset_name: str,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter([0], [0], [0], c="#222222", s=80, marker="o", label="robot base")
    ax.quiver(0, 0, 0, 0.25, 0, 0, color="#d62728", arrow_length_ratio=0.18)
    ax.quiver(0, 0, 0, 0, 0.25, 0, color="#2ca02c", arrow_length_ratio=0.18)
    ax.quiver(0, 0, 0, 0, 0, 0.25, color="#1f77b4", arrow_length_ratio=0.18)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(poses), 1)))
    for idx, pose in enumerate(poses):
        p = pose.position
        target = pose.target
        forward, right, up = _basis_from_pose(p, target)
        scale = 0.16
        center = p + forward * scale * 1.8
        corners = [
            center + right * scale + up * scale * 0.7,
            center - right * scale + up * scale * 0.7,
            center - right * scale - up * scale * 0.7,
            center + right * scale - up * scale * 0.7,
        ]
        color = colors[idx]
        ax.scatter([p[0]], [p[1]], [p[2]], color=color, s=58)
        ax.text(p[0], p[1], p[2], " " + pose.key.split(".")[-1], fontsize=8)
        ax.plot([p[0], target[0]], [p[1], target[1]], [p[2], target[2]], color=color, alpha=0.45)
        for corner in corners:
            ax.plot([p[0], corner[0]], [p[1], corner[1]], [p[2], corner[2]], color=color, linewidth=1.0)
        closed = corners + [corners[0]]
        ax.plot(
            [c[0] for c in closed],
            [c[1] for c in closed],
            [c[2] for c in closed],
            color=color,
            linewidth=1.0,
        )

    all_points = np.vstack([[0.0, 0.0, 0.0], *[pose.position for pose in poses], *[pose.target for pose in poses]])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    centers = (mins + maxs) / 2
    radius = max(float(np.max(maxs - mins)) / 2, 0.35)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(max(0.0, centers[2] - radius), centers[2] + radius)
    ax.set_xlabel("X (robot base)")
    ax.set_ylabel("Y (robot base)")
    ax.set_zlabel("Z (robot base)")
    source_note = " / ".join(sorted(set(pose.source for pose in poses)))
    ax.set_title(f"{dataset_name} Camera Pose Map ({source_note})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_samples_json(samples: Sequence[ImageSample], out_path: Path) -> None:
    data = [
        {
            "dataset": sample.dataset,
            "image_key": sample.image_key,
            "episode_index": sample.episode_index,
            "frame_index": sample.frame_index,
            "parquet_file": str(sample.parquet_file),
            "local_row": sample.local_row,
            "cache_path": str(sample.cache_path) if sample.cache_path else None,
        }
        for sample in samples
    ]
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _safe_cache_stem(sample: ImageSample) -> str:
    key = sample.image_key.replace(".", "_").replace("/", "_")
    return f"{key}_ep{sample.episode_index:06d}_row{sample.local_row:06d}"


def cache_sample_images(samples: Sequence[ImageSample], dataset_output: Path) -> None:
    cache_dir = dataset_output / "first_frames"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        cache_path = cache_dir / f"{_safe_cache_stem(sample)}.jpg"
        if not cache_path.is_file():
            sample.image.save(cache_path, quality=92)
        sample.cache_path = cache_path


def load_cached_samples(
    dataset_root: Path,
    dataset_output: Path,
    image_keys: Sequence[str],
    max_episodes: Optional[int],
) -> Optional[List[ImageSample]]:
    manifest_path = dataset_output / "samples.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        return None

    key_set = set(image_keys)
    filtered = [row for row in rows if row.get("image_key") in key_set]
    if not filtered:
        return None

    if max_episodes is not None:
        episodes = sorted({int(row["episode_index"]) for row in filtered})[:max_episodes]
        allowed = set(episodes)
        filtered = [row for row in filtered if int(row["episode_index"]) in allowed]

    samples: List[ImageSample] = []
    for row in filtered:
        cache_value = row.get("cache_path")
        if not cache_value:
            return None
        cache_path = Path(cache_value)
        if not cache_path.is_absolute():
            cache_path = dataset_output / cache_path
        if not cache_path.is_file():
            return None
        try:
            image = Image.open(cache_path).convert("RGB")
        except Exception:
            return None
        samples.append(
            ImageSample(
                dataset=str(row["dataset"]),
                dataset_root=dataset_root,
                image_key=str(row["image_key"]),
                episode_index=int(row["episode_index"]),
                frame_index=int(row["frame_index"]),
                parquet_file=Path(str(row["parquet_file"])),
                local_row=int(row["local_row"]),
                image=image,
                cache_path=cache_path,
            )
        )
    return samples


def analyze() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_roots = discover_dataset_roots(root, args.all_datasets)
    pose_config = _load_camera_pose_json(args.camera_pose_json)

    all_samples: List[ImageSample] = []
    dataset_summaries: Dict[str, Dict[str, Any]] = {}
    all_warnings: List[str] = []

    for dataset_root in dataset_roots:
        info = read_dataset_info(dataset_root)
        keys = image_feature_keys(info, args.image_key)
        dataset_output = output_dir / dataset_root.name if len(dataset_roots) > 1 else output_dir
        dataset_output.mkdir(parents=True, exist_ok=True)

        poses = resolve_camera_poses(keys, pose_config)
        camera_map_path = dataset_output / "camera_pose_map_3d.png"
        if not camera_map_path.is_file():
            plot_camera_pose_map(poses, dataset_root.name, camera_map_path)

        cached_samples = None if args.no_image_cache else load_cached_samples(
            dataset_root=dataset_root,
            dataset_output=dataset_output,
            image_keys=keys,
            max_episodes=args.max_episodes,
        )
        load_warnings: List[str] = []
        if cached_samples is not None:
            samples = cached_samples
            print(f"{dataset_root.name}: reused {len(samples)} cached first-frame images")
        else:
            samples, load_warnings = load_first_frame_samples(
                dataset_root=dataset_root,
                image_keys=keys,
                max_episodes=args.max_episodes,
                seed=args.seed,
                video_backend=args.video_backend,
            )
            if not args.no_image_cache:
                cache_sample_images(samples, dataset_output)
            save_samples_json(samples, dataset_output / "samples.json")
        all_samples.extend(samples)
        all_warnings.extend(load_warnings)
        dataset_summaries[dataset_root.name] = {
            "image_keys": keys,
            "sample_count": len(samples),
            "camera_pose_sources": sorted(set(pose.source for pose in poses)),
            "camera_pose_map": str(camera_map_path),
            "image_cache_enabled": not args.no_image_cache,
            "reused_cached_images": cached_samples is not None,
            "warnings": load_warnings,
        }
        if cached_samples is None:
            print(f"{dataset_root.name}: {len(samples)} first-frame image samples")
        for warning in load_warnings[:5]:
            print(f"  warning: {warning}")
        if len(load_warnings) > 5:
            print(f"  warning: ... {len(load_warnings) - 5} more")

    if args.skip_embeddings:
        summary = {
            "model_name": None,
            "embedding_skipped": True,
            "datasets": dataset_summaries,
            "sample_count": len(all_samples),
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved camera maps and sample manifests to: {output_dir}")
        return

    if len(all_samples) == 0:
        raise SystemExit("No first-frame images were loaded; cannot compute embeddings.")

    print(f"Embedding {len(all_samples)} images with {args.model_name}")
    embeddings = compute_dinov3_embeddings(
        all_samples,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )

    global_embedding_summary = write_embedding_outputs(
        samples=all_samples,
        embeddings=embeddings,
        out_dir=output_dir,
        title_prefix="Global",
        seed=args.seed,
        cluster_count=args.cluster_count,
        thumbnail_size=args.thumbnail_size,
        include_dataset_plot=True,
    )

    per_dataset_embedding_summaries: Dict[str, Any] = {}
    dataset_names = [sample.dataset for sample in all_samples]
    for dataset_name in sorted(set(dataset_names)):
        indices = np.asarray(
            [idx for idx, name in enumerate(dataset_names) if name == dataset_name],
            dtype=np.int64,
        )
        dataset_samples = [all_samples[int(idx)] for idx in indices]
        dataset_output = (
            output_dir / dataset_name if len(dataset_roots) > 1 else output_dir
        )
        per_dataset_embedding_summaries[dataset_name] = write_embedding_outputs(
            samples=dataset_samples,
            embeddings=embeddings[indices],
            out_dir=dataset_output,
            title_prefix=dataset_name,
            seed=args.seed,
            cluster_count=args.cluster_count,
            thumbnail_size=args.thumbnail_size,
            include_dataset_plot=False,
        )
        dataset_summaries.setdefault(dataset_name, {})[
            "per_dataset_embedding"
        ] = per_dataset_embedding_summaries[dataset_name]

    summary = {
        "model_name": args.model_name,
        "sample_count": len(all_samples),
        "embedding_dim": int(embeddings.shape[1]),
        "global_embedding": global_embedding_summary,
        "per_dataset_embeddings": per_dataset_embedding_summaries,
        "datasets": dataset_summaries,
        "warnings": all_warnings,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved image distribution analysis to: {output_dir}")


if __name__ == "__main__":
    analyze()
