"""
CIFAR-10 data loading and preprocessing.

Implements the data augmentation strategy from the ResNet paper:
- 4-pixel padding
- Random crop (32×32)
- Random horizontal flip
- Normalization to [-1, 1] per channel

TODO:
    - Implement CIFAR-10 dataset loader with torchvision
    - Apply 4-pixel padding transformation
    - Apply random crop and horizontal flip
    - Normalize to [-1, 1] range per channel
    - Return training and testing dataloaders
"""
