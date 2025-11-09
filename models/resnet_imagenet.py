"""ResNet architectures for ImageNet (optional extension).

This module implements the residual convolutional neural network
architectures used in the ImageNet experiments from
"Deep Residual Learning for Image Recognition" (He et al., CVPR 2016).

The ImageNet variants use 7×7 initial convolution, max pooling,
and four stages of feature maps (64, 128, 256, 512 channels).

Models implemented here:

* ``ResNet18`` (BasicBlock × [2, 2, 2, 2])
* ``ResNet34`` (BasicBlock × [3, 4, 6, 3])
* ``ResNet50`` (Bottleneck × [3, 4, 6, 3])
* ``ResNet101`` (Bottleneck × [3, 4, 23, 3])
* ``ResNet152`` (Bottleneck × [3, 8, 36, 3])
"""

from __future__ import annotations

from typing import List, Optional, Type

import torch
from torch import nn
from torch.nn import functional as F


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """3×3 convolution with padding."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """1×1 convolution."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    """Standard residual block with two 3×3 convolutions.
    
    Used in ResNet-18 and ResNet-34.
    """

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


class Bottleneck(nn.Module):
    """Bottleneck residual block with 1×1, 3×3, 1×1 convolutions.
    
    Used in ResNet-50, ResNet-101, and ResNet-152.
    The final 1×1 convolution expands the channels by a factor of 4.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = _conv3x3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = _conv1x1(out_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_ImageNet(nn.Module):
    """ResNet architecture for ImageNet classification."""

    def __init__(
        self,
        block: Type[nn.Module],
        layers: List[int],
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        if len(layers) != 4:
            raise ValueError("Expected four stage definitions for ImageNet ResNet")

        self.in_channels = 64
        
        # Initial 7×7 convolution with stride 2
        self.conv1 = nn.Conv2d(
            3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Four stages with increasing channel dimensions
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_parameters()

    def _make_layer(
        self,
        block: Type[nn.Module],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample: Optional[nn.Module] = None

        # Need downsampling if stride != 1 or channels don't match
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.in_channels, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers: List[nn.Module] = []
        layers.append(block(self.in_channels, planes, stride, downsample))
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes))

        return nn.Sequential(*layers)

    def _initialize_parameters(self) -> None:
        """Initialize parameters following the original paper."""
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
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def ResNet18(num_classes: int = 1000) -> ResNet_ImageNet:
    """ResNet-18 architecture for ImageNet.
    
    Args:
        num_classes: Number of output classes (default: 1000 for ImageNet)
    
    Returns:
        ResNet-18 model
    """
    return ResNet_ImageNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes: int = 1000) -> ResNet_ImageNet:
    """ResNet-34 architecture for ImageNet.
    
    Args:
        num_classes: Number of output classes (default: 1000 for ImageNet)
    
    Returns:
        ResNet-34 model
    """
    return ResNet_ImageNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes: int = 1000) -> ResNet_ImageNet:
    """ResNet-50 architecture for ImageNet.
    
    Args:
        num_classes: Number of output classes (default: 1000 for ImageNet)
    
    Returns:
        ResNet-50 model
    """
    return ResNet_ImageNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

__all__ = [
    "BasicBlock",
    "Bottleneck",
    "ResNet_ImageNet",
    "ResNet18",
    "ResNet34",
    "ResNet50",
]
