# COMP3314 Project: ResNet Reproduction

## Paper Reference

- **Title**: *Deep Residual Learning for Image Recognition*
- **Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **Venue**: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016
- **Link**: https://arxiv.org/abs/1512.03385

Our goal is to reproduce the CIFAR-10 experiments from Section 4.2 of the paper, comparing plain convolutional networks (PlainNet) against residual counterparts (ResNet) at depths 20, 32, 56, and 110 layers.

## Repository Layout

```
resnet-reproduction/
├── models/                   # CIFAR-10 PlainNet/ResNet implementations
├── datasets/                 # CIFAR-10 data pipeline
├── utils/                    # Training, scheduler, plotting helpers
├── examples/                 # Scripts for plotting/analysis
├── experiments/              # Notebooks and analysis scripts
├── results/                  # Logs + checkpoints + generated figures
└── main.py                   # CLI entry point for training/evaluation
```

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

The project assumes PyTorch ≥ 1.9 with CUDA support if you plan to train on GPU.

## Running the Code

### 1. Quick sanity check

```bash
python - <<'PY'
import torch
from models.resnet_cifar import ResNet20, ResNet32, PlainNet20

x = torch.randn(2, 3, 32, 32)
for name, cls in [("PlainNet20", PlainNet20), ("ResNet20", ResNet20), ("ResNet32", ResNet32)]:
    y = cls()(x)
    print(name, "->", tuple(y.shape))
PY
```

You should see `(2, 10)` outputs for each model, confirming the classifier heads. Constructors for the deeper variants (`PlainNet56`, `PlainNet110`, `ResNet56`, `ResNet110`) follow the same pattern.

### 2. Train a model

```bash
python main.py \
  --model ResNet20 \
  --epochs 164 \
  --batch_size 128 \
  --lr 0.1 \
  --device cuda \
  --amp
```

Key arguments:

- `--model`: `PlainNet20`, `PlainNet32`, `PlainNet56`, `PlainNet110`, `ResNet20`, `ResNet32`, `ResNet56`, or `ResNet110`
- `--device`: `cuda`, `cuda:0`, or `cpu` (default tries CUDA and falls back to CPU)
- `--amp`: enables mixed precision on CUDA for faster training
- `--warmup-epochs`: linear warmup duration (e.g. `--warmup-epochs 5`)
- `--milestones`: override LR decay iterations (default `32000,48000`)
- `--checkpoint-freq`: checkpoint interval in epochs

The script downloads CIFAR-10 to `./data/` on first run. Checkpoints are saved under `results/checkpoints/`, while CSV/JSON logs and diagnostic plots live in `results/logs/`.

### 3. Resume or evaluate

```bash
# Resume
python main.py --model ResNet20 --resume results/resnet20/checkpoints/ResNet20_20251109-163847_epoch90.pth

# Evaluation only
python main.py --model ResNet20 --eval-only --resume results/resnet20/checkpoints/ResNet20_20251109-163847_epoch90.pth
```

### 4. Visualise training

After training you will find:

- Run-specific JSON logs under `results/<model>/logs/`
- Checkpoints under `results/<model>/checkpoints/`
- Aggregated CSV + quick-look plots under `results/logs/`
- Publication-style figures (Figure 6 comparisons, accuracy/error dashboards) under `results/plots/`

To regenerate the figures programmatically:

```bash
python examples/plot_training_results.py
```

For bespoke analysis or Figure 7 reproduction, use the notebooks in `experiments/`.

## Reproduced Results

Full 164-epoch training runs were executed for both plain and residual networks at depths 20, 32, 56, and 110. Best CIFAR-10 test-set checkpoints achieved the following metrics:

| Model        | Depth | Parameters | Error (%) | Accuracy (%) |
|--------------|------:|-----------:|----------:|-------------:|
| PlainNet20   | 20    | 2.70e5     | 8.65      | 91.35        |
| PlainNet32   | 32    | 4.64e5     | 9.38      | 90.62        |
| PlainNet56   | 56    | 8.53e5     | 11.83     | 88.17        |
| PlainNet110  | 110   | 1.73e6     | 77.52     | 22.48        |
| ResNet20     | 20    | 2.72e5     | 7.65      | 92.35        |
| ResNet32     | 32    | 4.67e5     | 7.01      | 92.99        |
| ResNet56     | 56    | 8.56e5     | 6.54      | 93.46        |
| ResNet110    | 110   | 1.73e6     | 5.82      | 94.18        |

See `results/performance_summary.txt` for run identifiers, epoch numbers, and references to the raw log files (\*.json) used to compute the table.

Observations:

- PlainNets suffer from the degradation problem: deeper models (56/110) underperform shallower ones.
- Residual networks continue to improve with depth, matching the trend reported in Table 6 of the original paper.
- Generated figures in `results/plots/` reproduce Figure 6 (training/test error curves) and extend them with accuracy/loss dashboards for all depths.

## Current Status & Next Steps

- ✅ Plain/Residual CIFAR models implemented for depths 20, 32, 56, 110 (6n+2 rule).
- ✅ Training pipeline replicates paper hyper-parameters and learning-rate schedule (32k/48k decays, optional warmup).
- ✅ Result tables and plots generated from actual runs (`results/performance_summary.txt`, `results/plots/`).


## Troubleshooting

- **Dataset download issues**: remove the corrupted archive under `./data/` and rerun.
- **CUDA out of memory**: reduce `--batch-size` or disable AMP (`--no-amp` by default).
- **Slow dataloading**: increase `--num-workers` if your system supports it.

Feel free to open issues or PRs to extend the reproduction further (e.g., ResNet164/1001 or ImageNet experiments).