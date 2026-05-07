# LeRobot Dataset Analyze Tool

Visualizes the following items based on trajectory data.

- 2D/3D position distribution
- Position speed distribution

Automatically infers per-dataset configuration by reading the LeRobot `meta/info.json` and parquet schema.
Datasets with `x,y,z` column names use the corresponding indices; 2D state datasets are saved as 2D plots.
3D+ state datasets without coordinate names fall back to `observation.state[:3]`.
Speed is `||Δp|| / Δt` (state coordinate units per second), not a normalized value.

## 1) Installation

```bash
python3 -m pip install -r requirements.txt
```

## 2) Usage

```bash
python3 visualize.py --dataset-root dataset/droid_100 --output-dir outputs/droid_100
```

Run all LeRobot datasets under `dataset/` at once:

```bash
python3 visualize.py --dataset-root dataset --all-datasets --output-dir outputs
```

Print only the auto-inferred configuration:

```bash
python3 visualize.py --dataset-root dataset --all-datasets --print-config-only
```

## 3) Outputs

- 3D dataset:
  - `outputs/<DATASET>/position_distribution_3d.png`
  - `outputs/<DATASET>/position_projection_xy.png`
  - `outputs/<DATASET>/position_projection_yz.png`
  - `outputs/<DATASET>/position_projection_xz.png`
- 2D dataset:
  - `outputs/<DATASET>/position_distribution_2d.png`
- Common:
  - `outputs/<DATASET>/ee_speed_distribution_hist.png`
  - `outputs/<DATASET>/summary.json`
  - `outputs/summary_all.json` (when using `--all-datasets`)

Plot annotations:

  - Point color: frame step progress within an episode (0 → 1)
  - Colormap: blue → red
  - Green `^`: start of each episode
  - Red `x`: first point where the gripper value changes (shown only when gripper is inferred)

## Options

- Manually specify coordinate indices:

```bash
python3 visualize.py \
  --dataset-root dataset/droid_100 \
  --ee-indices 0 1 2
```

- Change the threshold for detecting gripper state changes:

```bash
python3 visualize.py \
  --dataset-root dataset/droid_100 \
  --gripper-index 6 \
  --gripper-change-threshold 1e-6
```

- Add jitter to spread out visualization points:

```bash
python3 visualize.py \
  --dataset-root dataset/droid_100 \
  --viz-jitter-std 0.003
```

- Use a non-standard LeRobot variant with different column names:

```bash
python3 visualize.py \
  --dataset-root <DATASET_ROOT> \
  --state-key <STATE_COLUMN> \
  --episode-key <EPISODE_COLUMN> \
  --timestamp-key <TIMESTAMP_COLUMN>
```
