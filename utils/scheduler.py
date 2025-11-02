"""
Learning rate scheduling utilities.

Implements the learning rate decay strategy from the ResNet paper:
- Divide lr by 10 at 32k and 48k iterations

TODO:
    - Implement custom LR scheduler for decay at specific iterations
    - Optionally add warmup for first few epochs
    - Support configurable decay milestones
"""
