"""
ResNet and PlainNet architectures for CIFAR-10.

Implements deep residual learning as described in "Deep Residual Learning for Image Recognition"
(He et al., CVPR 2016).

Key classes:
    - BasicBlock: Standard residual block with two 3×3 convolutions
    - PlainBlock: Plain network block without residual connections
    - ResNet_CIFAR: Main ResNet class for CIFAR-10
    - PlainNet_CIFAR: Plain network for comparison studies

TODO: 
    - Implement BasicBlock with two 3×3 convs + BN + ReLU
    - Implement PlainBlock for comparison
    - Implement ResNet_CIFAR with 3 stages (16, 32, 64 filters)
    - Add constructors for ResNet20, ResNet32, ResNet56, ResNet110
    - Add PlainNet versions matching each depth
"""
