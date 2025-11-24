"""ResNet and PlainNet architectures for CIFAR-10.

This module implements the residual and plain convolutional neural network
architectures used in the CIFAR-10 experiments from
"Deep Residual Learning for Image Recognition" (He et al., CVPR 2016).

The CIFAR variants use only 3×3 convolutions and follow the depth formula
``depth = 6 * n + 2`` with three stages of feature maps (16, 32, 64 channels).

Models implemented here:

* ``ResNet20`` / ``PlainNet20`` (n=3)
* ``ResNet32`` / ``PlainNet32`` (n=5)
* ``ResNet56`` / ``PlainNet56`` (n=9)
* ``ResNet110`` / ``PlainNet110`` (n=18)
"""

from __future__ import annotations

from typing import List, Optional, Type

import torch
from torch import nn
from torch.nn import functional as F


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """3×3 convolution with padding used throughout the CIFAR models."""

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    """Standard residual block with two 3×3 convolutions."""

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PlainBlock(nn.Module):
    """Plain network block mirroring ``BasicBlock`` without residual shortcut."""

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            out = self.downsample(out)

        out = self.relu(out)
        return out


class _BaseCIFARNet(nn.Module):
    """Common scaffold shared by residual and plain CIFAR networks."""

    def __init__(
        self,
        block: Type[nn.Module],
        layers: List[int],
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if len(layers) != 3:
            raise ValueError("Expected three stage definitions for CIFAR ResNet/PlainNet")

        self.in_channels = 16
        self.conv1 = _conv3x3(3, self.in_channels)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self._initialize_parameters()

    def _make_layer(
        self,
        block: Type[nn.Module],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample: Optional[nn.Module] = None

        if issubclass(block, BasicBlock) and (
            stride != 1 or self.in_channels != planes * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers: List[nn.Module] = []
        layers.append(block(self.in_channels, planes, stride, downsample))
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes))

        return nn.Sequential(*layers)

    def _initialize_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class ResNet_CIFAR(_BaseCIFARNet):
    """Residual network for CIFAR-10."""

    def __init__(self, layers: List[int], num_classes: int = 10) -> None:
        super().__init__(BasicBlock, layers, num_classes=num_classes)


class PlainNet_CIFAR(_BaseCIFARNet):
    """Plain network (without residual connections) for CIFAR-10."""

    def __init__(self, layers: List[int], num_classes: int = 10) -> None:
        super().__init__(PlainBlock, layers, num_classes=num_classes)


def _validate_depth(depth: int) -> int:
    if (depth - 2) % 6 != 0:
        raise ValueError("CIFAR ResNet depth should satisfy depth = 6n + 2")
    return (depth - 2) // 6


def _make_layers_for_depth(depth: int) -> List[int]:
    n = _validate_depth(depth)
    return [n, n, n]


def resnet_cifar(depth: int, num_classes: int = 10) -> ResNet_CIFAR:
    """Factory function for CIFAR ResNet."""

    return ResNet_CIFAR(_make_layers_for_depth(depth), num_classes=num_classes)


def plainnet_cifar(depth: int, num_classes: int = 10) -> PlainNet_CIFAR:
    """Factory function for CIFAR PlainNet."""

    return PlainNet_CIFAR(_make_layers_for_depth(depth), num_classes=num_classes)


def ResNet20(num_classes: int = 10) -> ResNet_CIFAR:
    return resnet_cifar(20, num_classes=num_classes)


def ResNet32(num_classes: int = 10) -> ResNet_CIFAR:
    return resnet_cifar(32, num_classes=num_classes)


def ResNet56(num_classes: int = 10) -> ResNet_CIFAR:
    return resnet_cifar(56, num_classes=num_classes)


def ResNet110(num_classes: int = 10) -> ResNet_CIFAR:
    return resnet_cifar(110, num_classes=num_classes)


def PlainNet20(num_classes: int = 10) -> PlainNet_CIFAR:
    return plainnet_cifar(20, num_classes=num_classes)


def PlainNet32(num_classes: int = 10) -> PlainNet_CIFAR:
    return plainnet_cifar(32, num_classes=num_classes)


def PlainNet56(num_classes: int = 10) -> PlainNet_CIFAR:
    return plainnet_cifar(56, num_classes=num_classes)


def PlainNet110(num_classes: int = 10) -> PlainNet_CIFAR:
    return plainnet_cifar(110, num_classes=num_classes)


__all__ = [
    "BasicBlock",
    "PlainBlock",
    "ResNet_CIFAR",
    "PlainNet_CIFAR",
    "ResNet20",
    "ResNet32",
    "ResNet56",
    "ResNet110",
    "PlainNet20",
    "PlainNet32",
    "PlainNet56",
    "PlainNet110",
    "resnet_cifar",
    "plainnet_cifar",
]
