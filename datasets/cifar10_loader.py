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

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(root="./data", batch_size=128, num_workers=4):
    """
    Loads CIFAR-10 dataset with the standard ResNet-style preprocessing.

    Args:
        root (str): Path to store/load the dataset.
        batch_size (int): Batch size for both train and test.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        train_loader (DataLoader)
        test_loader (DataLoader)
    """
    cifar10_mean = (0.5, 0.5, 0.5)
    cifar10_std  = (0.5, 0.5, 0.5)

    # --- Data augmentation for training set ---
    train_transform = transforms.Compose([
        transforms.Pad(4),                          # 4-pixel padding
        transforms.RandomCrop(32),                  # Random 32x32 crop
        transforms.RandomHorizontalFlip(),          # Random flip
        transforms.ToTensor(),                      # Convert to [0,1] tensor
        transforms.Normalize(cifar10_mean, cifar10_std),  # Normalize to [-1,1]
    ])

    # --- Testing set: only normalize ---
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    # --- Dataset loading ---
    train_dataset = datasets.CIFAR10(root=root, train=True, download=True,
                                     transform=train_transform)
    test_dataset  = datasets.CIFAR10(root=root, train=False, download=True,
                                     transform=test_transform)

    # --- Data loaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    return train_loader, test_loader
