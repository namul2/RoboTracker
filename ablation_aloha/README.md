# ALOHA Sim Insertion BC Ablation

This folder contains a minimal behavior cloning pipeline for comparing human demonstrations and scripted demonstrations on the ALOHA sim insertion task.

Both variants train the same image+state policy and are evaluated with the same offline action-prediction metrics.

## Dataset

The code expects LeRobot v3 parquet datasets at:

- `dataset/aloha_sim_insertion_human_image`
- `dataset/aloha_sim_insertion_scripted_image`

Each frame uses:

- `observation.images.top`
- `observation.state`
- `action`

## Install

```bash
pip install -r ablation_aloha/requirements.txt
```

`gymnasium` and `gym-aloha` are listed as optional dependencies for future simulator rollout evaluation. The current scripts only require offline dataset evaluation.

## Smoke Test

Run a tiny CPU training job first to verify that parquet loading, image decoding, forward/backward, and checkpoint saving work:

```bash
python ablation_aloha/train_bc.py \
  --config ablation_aloha/configs/aloha_sim_insertion_human.yaml \
  --epochs 1 \
  --sample-limit 256 \
  --batch-size 32 \
  --num-workers 0 \
  --device cpu
```

## Train

Train on human demonstrations:

```bash
python ablation_aloha/train_bc.py \
  --config ablation_aloha/configs/aloha_sim_insertion_human.yaml
```

Train on scripted demonstrations:

```bash
python ablation_aloha/train_bc.py \
  --config ablation_aloha/configs/aloha_sim_insertion_scripted.yaml
```

Checkpoints are saved to:

- `ablation_aloha/checkpoints/human/best.pt`
- `ablation_aloha/checkpoints/scripted/best.pt`

TensorBoard event files are saved under each checkpoint directory:

- `ablation_aloha/checkpoints/human/tensorboard`
- `ablation_aloha/checkpoints/scripted/tensorboard`

Open the loss dashboard with:

```bash
tensorboard --logdir ablation_aloha/checkpoints
```

Useful scalar groups:

- `objective/*`: normalized BC objective losses.
- `raw_action/*`: validation errors in the original action scale.
- `raw_action_dim/*`: per-action-dimension validation MAE.

## Compare

```bash
python ablation_aloha/eval_bc.py \
  ablation_aloha/checkpoints/human/best.pt \
  ablation_aloha/checkpoints/scripted/best.pt \
  --split val \
  --output ablation_aloha/checkpoints/compare_val.json
```

Reported metrics:

- `raw_mae`, `raw_rmse`: errors in the original action scale.
- `norm_mse`, `norm_mae`: errors after normalizing actions with train-split statistics.
- `raw_mae_per_action_dim`: per-dimension MAE for the 14 joint actions.

The current evaluation is offline action prediction because `gym_aloha` is not installed in this environment. To add simulator success-rate rollout later, install `gymnasium` and `gym-aloha`, then wrap the trained policy with the environment observation keys.
