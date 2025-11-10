#!/usr/bin/env python3
"""Example script demonstrating how to use plot_curves utilities.

This script shows various ways to visualize training results from
the ResNet reproduction experiments.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt

from utils.plot_curves import (
    plot_training_curves,
    plot_loss_curves,
    plot_single_model_curves,
    compare_multiple_models,
)


def main():
    """Generate various training curve visualizations."""
    
    # Define paths to log files
    results_dir = Path('results')
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    print("Generating training curve visualizations...")
    print("=" * 70)
    
    # Example 1: Compare PlainNet-20 vs ResNet-20 (Error Rate)
    # This reproduces Figure 6 from the ResNet paper
    print("\n1. Generating PlainNet-20 vs ResNet-20 comparison (Error Rate)...")
    plot_training_curves(
        plain_log='results/plain20/logs/PlainNet20_20251106-161914.json',
        resnet_log='results/logs/ResNet20_20251106-024420.json',
        save_path=plots_dir / 'figure6_plain20_vs_resnet20_error.png',
        title='CIFAR-10: 20-layer Plain vs Residual Network (Error Rate)',
        plot_error=True,
    )
    print("   ✓ Saved to: figure6_plain20_vs_resnet20_error.png")
    
    # Example 2: Compare PlainNet-20 vs ResNet-20 (Accuracy)
    print("\n2. Generating PlainNet-20 vs ResNet-20 comparison (Accuracy)...")
    plot_training_curves(
        plain_log='results/plain20/logs/PlainNet20_20251106-161914.json',
        resnet_log='results/logs/ResNet20_20251106-024420.json',
        save_path=plots_dir / 'plain20_vs_resnet20_accuracy.png',
        title='CIFAR-10: 20-layer Plain vs Residual Network (Accuracy)',
        plot_error=False,
    )
    print("   ✓ Saved to: plain20_vs_resnet20_accuracy.png")
    
    # Example 3: Compare PlainNet-32 vs ResNet-32
    plain32_log = results_dir / 'plain32' / 'logs' / 'PlainNet32_20251106-203115.json'
    resnet32_log = results_dir / 'resnet32' / 'logs' / 'ResNet32_20251106-215354.json'
    
    if plain32_log.exists() and resnet32_log.exists():
        print("\n3. Generating PlainNet-32 vs ResNet-32 comparison...")
        plot_training_curves(
            plain_log=plain32_log,
            resnet_log=resnet32_log,
            save_path=plots_dir / 'plain32_vs_resnet32_error.png',
            title='CIFAR-10: 32-layer Plain vs Residual Network (Error Rate)',
            plot_error=True,
        )
        print("   ✓ Saved to: plain32_vs_resnet32_error.png")
    
    # Example 4: Loss curves for PlainNet-20 vs ResNet-20
    print("\n4. Generating loss curve comparison...")
    plot_loss_curves(
        plain_log='results/plain20/logs/PlainNet20_20251106-161914.json',
        resnet_log='results/logs/ResNet20_20251106-024420.json',
        save_path=plots_dir / 'plain20_vs_resnet20_loss.png',
        title='CIFAR-10: Training and Test Loss Comparison',
    )
    print("   ✓ Saved to: plain20_vs_resnet20_loss.png")
    
    # Example 5: Individual model training curves
    print("\n5. Generating individual model curves...")
    plot_single_model_curves(
        log_path='results/logs/ResNet20_20251106-024420.json',
        model_name='ResNet-20',
        save_path=plots_dir / 'resnet20_detailed.png',
    )
    print("   ✓ Saved to: resnet20_detailed.png")
    
    # Example 6: Compare all available models
    print("\n6. Generating multi-model comparison...")
    log_paths = {
        'PlainNet-20': 'results/plain20/logs/PlainNet20_20251106-161914.json',
        'ResNet-20': 'results/logs/ResNet20_20251106-024420.json',
    }
    
    if plain32_log.exists():
        log_paths['PlainNet-32'] = str(plain32_log)
    if resnet32_log.exists():
        log_paths['ResNet-32'] = str(resnet32_log)
    
    compare_multiple_models(
        log_paths=log_paths,
        metric='accuracy',
        split='test',
        save_path=plots_dir / 'all_models_test_accuracy.png',
        title='CIFAR-10: Test Accuracy Comparison Across All Models',
        figsize=(12, 7),
    )
    print(f"   ✓ Saved to: all_models_test_accuracy.png ({len(log_paths)} models)")
    
    # Example 7: Training accuracy comparison
    print("\n7. Generating training accuracy comparison...")
    compare_multiple_models(
        log_paths=log_paths,
        metric='accuracy',
        split='train',
        save_path=plots_dir / 'all_models_train_accuracy.png',
        title='CIFAR-10: Training Accuracy Comparison',
        figsize=(12, 7),
    )
    print("   ✓ Saved to: all_models_train_accuracy.png")
    
    print("\n" + "=" * 70)
    print(f"✓ All plots successfully generated and saved to: {plots_dir}")
    print("\nYou can now view the plots to analyze:")
    print("  - Training degradation in plain networks")
    print("  - Benefits of residual connections")
    print("  - Convergence behavior across different depths")


if __name__ == '__main__':
    main()
