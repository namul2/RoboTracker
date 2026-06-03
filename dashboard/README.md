# RoboTracker Dashboard

A single-page, self-contained HTML dashboard that aggregates everything under
`../outputs/` into one clean view: trajectory distributions, first-frame image
embeddings, OOD detection signals, and per-episode OOD inspection.

## Generate / refresh

```bash
python3 dashboard/build_dashboard.py
```

This scans the local `outputs/` tree, reads the `summary*.json` files, and writes
`dashboard/index.html`. Re-run it whenever the analysis scripts produce new outputs.

> The generator ignores the absolute paths stored inside the JSON files (those
> point at the machine that produced them, e.g. `/hdd_1/...`). It links images by
> walking the real `outputs/` directory and emitting paths relative to this folder,
> so everything works entirely inside this repo with no external dependencies.

## View

Open `dashboard/index.html` directly in a browser, or serve it locally:

```bash
python3 -m http.server -d dashboard 8000   # then visit http://localhost:8000
```

## Sections

| Section | Source under `outputs/` |
|---------|-------------------------|
| Overview | aggregated dataset stats |
| Trajectory | `trajectory_distribution/` (single-arm, 2D, and bimanual FK) |
| Image Distribution | `image_distribution/` (DINOv3 + UMAP + KMeans) |
| OOD Detection | `ood_detection/` (train/test signal bars + plots) |
| Episode Inspector | `ood_episode_inspector/` (ranked top-OOD episodes) |

Click any plot to open it full-size; `Esc` closes the lightbox. The sidebar
switches sections, and `#overview` / `#trajectory` / `#image` / `#ood` /
`#episode` URL hashes deep-link straight to a section.

Standard library only — no `pip install` required to build or view.
