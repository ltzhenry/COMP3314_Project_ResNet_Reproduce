"""Visualization utilities for training curves.

Generates plots matching Figure 6 from the ResNet paper showing 
training/test error comparison between plain and residual networks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def load_training_log(log_path: Union[str, Path]) -> Dict[str, List[Dict[str, float]]]:
    """Load training log from JSON file.
    
    Args:
        log_path: Path to the JSON log file
        
    Returns:
        Dictionary with 'train' and 'test' keys containing lists of epoch metrics
    """
    with open(log_path, 'r') as f:
        return json.load(f)


def extract_metrics(
    log_data: Dict[str, List[Dict[str, float]]],
    metric: str = 'accuracy'
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract training and test metrics from log data.
    
    Args:
        log_data: Dictionary containing 'train' and 'test' lists
        metric: Metric name to extract ('accuracy' or 'loss')
        
    Returns:
        Tuple of (train_metrics, test_metrics) as numpy arrays
    """
    train_metrics = np.array([epoch[metric] for epoch in log_data['train']])
    test_metrics = np.array([epoch[metric] for epoch in log_data['test']])
    return train_metrics, test_metrics


def plot_training_curves(
    plain_log: Union[str, Path, Dict],
    resnet_log: Union[str, Path, Dict],
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
    plot_error: bool = True,
) -> plt.Figure:
    """Plot training curves comparing plain and residual networks.
    
    Creates a figure with two subplots showing training and test error/accuracy
    curves, similar to Figure 6 in the ResNet paper.
    
    Args:
        plain_log: Path to plain network log file or loaded log dict
        resnet_log: Path to ResNet log file or loaded log dict
        save_path: Optional path to save the figure
        title: Optional overall title for the figure
        figsize: Figure size (width, height)
        plot_error: If True, plot error rates; if False, plot accuracy
        
    Returns:
        matplotlib Figure object
    """
    # Load log data if paths are provided
    if isinstance(plain_log, (str, Path)):
        plain_data = load_training_log(plain_log)
    else:
        plain_data = plain_log
        
    if isinstance(resnet_log, (str, Path)):
        resnet_data = load_training_log(resnet_log)
    else:
        resnet_data = resnet_log
    
    # Extract metrics
    if plot_error:
        # Convert accuracy to error rate (%)
        plain_train, plain_test = extract_metrics(plain_data, 'accuracy')
        resnet_train, resnet_test = extract_metrics(resnet_data, 'accuracy')
        
        plain_train = (1 - plain_train) * 100
        plain_test = (1 - plain_test) * 100
        resnet_train = (1 - resnet_train) * 100
        resnet_test = (1 - resnet_test) * 100
        
        ylabel = 'Error (%)'
    else:
        plain_train, plain_test = extract_metrics(plain_data, 'accuracy')
        resnet_train, resnet_test = extract_metrics(resnet_data, 'accuracy')
        
        plain_train = plain_train * 100
        plain_test = plain_test * 100
        resnet_train = resnet_train * 100
        resnet_test = resnet_test * 100
        
        ylabel = 'Accuracy (%)'
    
    epochs = np.arange(1, len(plain_train) + 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot training curves
    ax1.plot(epochs, plain_train, 'b-', label='Plain', linewidth=2, alpha=0.8)
    ax1.plot(epochs, resnet_train, 'r-', label='ResNet', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel(f'Training {ylabel}', fontsize=12)
    ax1.set_title('Training', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot test curves
    ax2.plot(epochs, plain_test, 'b-', label='Plain', linewidth=2, alpha=0.8)
    ax2.plot(epochs, resnet_test, 'r-', label='ResNet', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel(f'Test {ylabel}', fontsize=12)
    ax2.set_title('Test', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_loss_curves(
    plain_log: Union[str, Path, Dict],
    resnet_log: Union[str, Path, Dict],
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Plot loss curves comparing plain and residual networks.
    
    Args:
        plain_log: Path to plain network log file or loaded log dict
        resnet_log: Path to ResNet log file or loaded log dict
        save_path: Optional path to save the figure
        title: Optional overall title for the figure
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    # Load log data if paths are provided
    if isinstance(plain_log, (str, Path)):
        plain_data = load_training_log(plain_log)
    else:
        plain_data = plain_log
        
    if isinstance(resnet_log, (str, Path)):
        resnet_data = load_training_log(resnet_log)
    else:
        resnet_data = resnet_log
    
    # Extract loss metrics
    plain_train, plain_test = extract_metrics(plain_data, 'loss')
    resnet_train, resnet_test = extract_metrics(resnet_data, 'loss')
    
    epochs = np.arange(1, len(plain_train) + 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot training loss
    ax1.plot(epochs, plain_train, 'b-', label='Plain', linewidth=2, alpha=0.8)
    ax1.plot(epochs, resnet_train, 'r-', label='ResNet', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot test loss
    ax2.plot(epochs, plain_test, 'b-', label='Plain', linewidth=2, alpha=0.8)
    ax2.plot(epochs, resnet_test, 'r-', label='ResNet', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Test Loss', fontsize=12)
    ax2.set_title('Test', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_single_model_curves(
    log_path: Union[str, Path, Dict],
    model_name: str = 'Model',
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Plot training curves for a single model.
    
    Args:
        log_path: Path to log file or loaded log dict
        model_name: Name of the model for labeling
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    # Load log data if path is provided
    if isinstance(log_path, (str, Path)):
        log_data = load_training_log(log_path)
    else:
        log_data = log_path
    
    # Extract metrics
    train_acc, test_acc = extract_metrics(log_data, 'accuracy')
    train_loss, test_loss = extract_metrics(log_data, 'loss')
    
    # Convert to percentages
    train_acc = train_acc * 100
    test_acc = test_acc * 100
    
    epochs = np.arange(1, len(train_acc) + 1)
    
    # Create figure with four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Plot training accuracy
    ax1.plot(epochs, train_acc, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Training Accuracy (%)', fontsize=11)
    ax1.set_title('Training Accuracy', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot test accuracy
    ax2.plot(epochs, test_acc, 'r-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax2.set_title('Test Accuracy', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot training loss
    ax3.plot(epochs, train_loss, 'b-', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Training Loss', fontsize=11)
    ax3.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot test loss
    ax4.plot(epochs, test_loss, 'r-', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Test Loss', fontsize=11)
    ax4.set_title('Test Loss', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(f'{model_name} Training Curves', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def compare_multiple_models(
    log_paths: Dict[str, Union[str, Path]],
    metric: str = 'accuracy',
    split: str = 'test',
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Compare multiple models on a single plot.
    
    Args:
        log_paths: Dictionary mapping model names to log file paths
        metric: Metric to plot ('accuracy' or 'loss')
        split: Data split to plot ('train' or 'test')
        save_path: Optional path to save the figure
        title: Optional title for the plot
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, log_path in log_paths.items():
        # Load log data
        log_data = load_training_log(log_path)
        
        # Extract metric
        if metric == 'accuracy':
            values = np.array([epoch[metric] for epoch in log_data[split]]) * 100
            ylabel = f'{split.capitalize()} Accuracy (%)'
        else:
            values = np.array([epoch[metric] for epoch in log_data[split]])
            ylabel = f'{split.capitalize()} Loss'
        
        epochs = np.arange(1, len(values) + 1)
        ax.plot(epochs, values, linewidth=2, label=model_name, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_paper_style_comparison(
    model_logs: Dict[str, Union[str, Path]],
    model_type: str = 'plain',
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 5),
    iterations_per_epoch: int = 391,  # CIFAR-10: 50000/128 ≈ 391
    max_iterations: Optional[int] = None,
) -> plt.Figure:
    """Plot training/test error curves in the style of ResNet paper Figure 6.
    
    Creates a figure matching the original paper's visualization with:
    - Dashed lines for training error
    - Solid lines for test error
    - X-axis in iterations (1e4)
    - Different colors for different network depths
    
    Args:
        model_logs: Dictionary mapping model names (e.g., 'plain-20', 'ResNet-20') to log paths
        model_type: Type of models being plotted ('plain' or 'resnet')
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
        iterations_per_epoch: Number of iterations per epoch (batches per epoch)
        max_iterations: Maximum iterations to plot (in actual iterations, not 1e4)
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> plain_logs = {
        ...     'plain-20': 'results/plain20/logs/plain20.json',
        ...     'plain-32': 'results/plain32/logs/plain32.json',
        ... }
        >>> fig = plot_paper_style_comparison(plain_logs, model_type='plain')
    """
    # Define colors for different depths (matching paper style)
    colors = {
        '20': '#B8A800',  # Yellow-green
        '32': '#00C8C8',  # Cyan
        '44': '#00C800',  # Green
        '56': '#C80000',  # Red
        '110': '#000000',  # Black
    }
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Sort models by depth for consistent plotting
    sorted_models = sorted(model_logs.items(), key=lambda x: int(x[0].split('-')[-1]))
    
    for model_name, log_path in sorted_models:
        # Extract depth from model name (e.g., 'plain-20' -> '20')
        depth = model_name.split('-')[-1]
        color = colors.get(depth, None)
        
        # Load log data
        log_data = load_training_log(log_path)
        
        # Extract accuracy and convert to error rate (%)
        train_acc = np.array([epoch['accuracy'] for epoch in log_data['train']])
        test_acc = np.array([epoch['accuracy'] for epoch in log_data['test']])
        
        train_error = (1 - train_acc) * 100
        test_error = (1 - test_acc) * 100
        
        # Convert epochs to iterations
        epochs = np.arange(1, len(train_error) + 1)
        iterations = epochs * iterations_per_epoch
        iterations_1e4 = iterations / 1e4  # Scale to 1e4
        
        # Apply max iterations filter if specified
        if max_iterations:
            mask = iterations <= max_iterations
            iterations_1e4 = iterations_1e4[mask]
            train_error = train_error[mask]
            test_error = test_error[mask]
        
        # Determine label based on model type
        if model_type.lower() == 'plain':
            label = f'plain-{depth}'
        else:
            label = f'ResNet-{depth}'
        
        # Plot training error (dashed line)
        ax.plot(iterations_1e4, train_error, '--', 
                color=color, linewidth=1.5, alpha=0.6, label=f'{label} (train)')
        
        # Plot test error (solid line)
        ax.plot(iterations_1e4, test_error, '-', 
                color=color, linewidth=2, alpha=0.9, label=label)
    
    # Formatting
    ax.set_xlabel('iter. (1e4)', fontsize=13)
    ax.set_ylabel('error (%)', fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 20)
    
    # Legend
    ax.legend(fontsize=10, loc='upper right', ncol=1)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_paper_style_side_by_side(
    plain_logs: Dict[str, Union[str, Path]],
    resnet_logs: Dict[str, Union[str, Path]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 5),
    iterations_per_epoch: int = 391,
    max_iterations: Optional[int] = None,
) -> plt.Figure:
    """Plot plain vs ResNet comparison side-by-side in paper style.
    
    Creates a figure with two subplots matching Figure 6 from the ResNet paper:
    - Left: Plain networks of various depths
    - Right: Residual networks of various depths
    - Dashed lines for training error, solid lines for test error
    
    Args:
        plain_logs: Dictionary mapping plain model names to log paths
        resnet_logs: Dictionary mapping ResNet model names to log paths
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
        iterations_per_epoch: Number of iterations per epoch
        max_iterations: Maximum iterations to plot
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> plain_logs = {
        ...     'plain-20': 'results/plain20/logs/plain20.json',
        ...     'plain-32': 'results/plain32/logs/plain32.json',
        ... }
        >>> resnet_logs = {
        ...     'ResNet-20': 'results/resnet20/logs/resnet20.json',
        ...     'ResNet-32': 'results/resnet32/logs/resnet32.json',
        ... }
        >>> fig = plot_paper_style_side_by_side(plain_logs, resnet_logs)
    """
    # Define colors for different depths
    colors = {
        '20': '#B8A800',  # Yellow-green
        '32': '#00C8C8',  # Cyan
        '44': '#00C800',  # Green
        '56': '#C80000',  # Red
        '110': '#000000',  # Black
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot plain networks (left subplot)
    sorted_plain = sorted(plain_logs.items(), key=lambda x: int(x[0].split('-')[-1]))
    
    for model_name, log_path in sorted_plain:
        depth = model_name.split('-')[-1]
        color = colors.get(depth, None)
        
        log_data = load_training_log(log_path)
        train_acc = np.array([epoch['accuracy'] for epoch in log_data['train']])
        test_acc = np.array([epoch['accuracy'] for epoch in log_data['test']])
        
        train_error = (1 - train_acc) * 100
        test_error = (1 - test_acc) * 100
        
        epochs = np.arange(1, len(train_error) + 1)
        iterations = epochs * iterations_per_epoch
        iterations_1e4 = iterations / 1e4
        
        if max_iterations:
            mask = iterations <= max_iterations
            iterations_1e4 = iterations_1e4[mask]
            train_error = train_error[mask]
            test_error = test_error[mask]
        
        # Training error (dashed)
        ax1.plot(iterations_1e4, train_error, '--', 
                color=color, linewidth=1.5, alpha=0.6)
        
        # Test error (solid)
        ax1.plot(iterations_1e4, test_error, '-', 
                color=color, linewidth=2, alpha=0.9, label=f'plain-{depth}')
    
    ax1.set_xlabel('iter. (1e4)', fontsize=13)
    ax1.set_ylabel('error (%)', fontsize=13)
    ax1.set_title('Plain Networks', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 20)
    ax1.legend(fontsize=10, loc='upper right')
    
    # Plot ResNets (right subplot)
    sorted_resnet = sorted(resnet_logs.items(), key=lambda x: int(x[0].split('-')[-1]))
    
    for model_name, log_path in sorted_resnet:
        depth = model_name.split('-')[-1]
        color = colors.get(depth, None)
        
        log_data = load_training_log(log_path)
        train_acc = np.array([epoch['accuracy'] for epoch in log_data['train']])
        test_acc = np.array([epoch['accuracy'] for epoch in log_data['test']])
        
        train_error = (1 - train_acc) * 100
        test_error = (1 - test_acc) * 100
        
        epochs = np.arange(1, len(train_error) + 1)
        iterations = epochs * iterations_per_epoch
        iterations_1e4 = iterations / 1e4
        
        if max_iterations:
            mask = iterations <= max_iterations
            iterations_1e4 = iterations_1e4[mask]
            train_error = train_error[mask]
            test_error = test_error[mask]
        
        # Training error (dashed)
        ax2.plot(iterations_1e4, train_error, '--', 
                color=color, linewidth=1.5, alpha=0.6)
        
        # Test error (solid)
        ax2.plot(iterations_1e4, test_error, '-', 
                color=color, linewidth=2, alpha=0.9, label=f'ResNet-{depth}')
        
        # Add text annotation for specific layers (like in the paper)
        if depth in ['20', '56', '110']:
            last_idx = -1
            ax2.text(iterations_1e4[last_idx] + 0.1, test_error[last_idx], 
                    f'{depth}-layer', fontsize=9, va='center')
    
    ax2.set_xlabel('iter. (1e4)', fontsize=13)
    ax2.set_ylabel('error (%)', fontsize=13)
    ax2.set_title('Residual Networks', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 20)
    ax2.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


__all__ = [
    'load_training_log',
    'extract_metrics',
    'plot_training_curves',
    'plot_loss_curves',
    'plot_single_model_curves',
    'compare_multiple_models',
    'plot_paper_style_comparison',
    'plot_paper_style_side_by_side',
]
