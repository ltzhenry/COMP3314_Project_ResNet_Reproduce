# COMP3314 Project: ResNet Reproduction

## Overview

This project reproduces key experiments from "Deep Residual Learning for Image Recognition" (He et al., CVPR 2016).

## Project Structure

```
resnet-reproduction/
├── models/                   # Model architectures
│   ├── resnet_cifar.py      # ResNet and PlainNet for CIFAR-10
│   └── resnet_imagenet.py   # (Optional) ResNet for ImageNet
├── datasets/                # Data loading utilities
│   └── cifar10_loader.py    # CIFAR-10 dataset and augmentation
├── utils/                   # Training and utility functions
│   ├── train_eval.py        # Training loop and evaluation
│   ├── scheduler.py         # Learning rate scheduling
│   └── plot_curves.py       # Visualization utilities
├── experiments/             # Experiment notebooks and scripts
│   ├── plain_vs_resnet.ipynb         # Training curve comparison
│   └── layer_response_analysis.py    # Layer response analysis
├── results/                 # Output directory
│   ├── checkpoints/         # Trained model weights
│   └── logs/                # Training logs
└── main.py                  # Main training script
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python main.py --model ResNet20 --epochs 164 --batch_size 128 --lr 0.1
```

### Running Experiments

```bash
# Open Jupyter notebook for training comparison
jupyter notebook experiments/plain_vs_resnet.ipynb
```

## Paper Alignment

- **Section 3.1-3.3**: Residual block structure
- **Section 4.2**: CIFAR-10 experiment setup
- **Table 6**: Network depth configurations (20, 32, 56, 110)
- **Figure 6**: Training/test error curves
- **Figure 7**: Layer response analysis

## TODO

- [ ] Implement BasicBlock and ResNet architectures
- [ ] Implement CIFAR-10 data loading with augmentation
- [ ] Implement training loop and evaluation
- [ ] Implement learning rate scheduler
- [ ] Reproduce Figure 6 training curves
- [ ] Reproduce Figure 7 layer response analysis