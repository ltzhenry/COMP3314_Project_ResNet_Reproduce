"""
Models package for ResNet reproduction project.

Contains implementations of ResNet architectures for CIFAR-10 and ImageNet datasets.
"""

from .resnet_cifar import (
    ResNet20,
    ResNet32,
    PlainNet20,
    PlainNet32,
    ResNet_CIFAR,
    PlainNet_CIFAR,
)
from .resnet_imagenet import (
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    ResNet_ImageNet,
)

__all__ = [
    # CIFAR models
    "ResNet20",
    "ResNet32",
    "PlainNet20",
    "PlainNet32",
    "ResNet_CIFAR",
    "PlainNet_CIFAR",
    # ImageNet models
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet_ImageNet",
]
