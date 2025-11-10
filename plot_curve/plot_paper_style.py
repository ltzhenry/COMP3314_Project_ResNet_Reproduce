#!/usr/bin/env python3
"""Generate paper-style comparison plots for plain vs ResNet models.

This script creates visualizations matching Figure 6 from the ResNet paper,
showing training and test error curves for different network depths.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.plot_curves import (
    plot_paper_style_comparison,
    plot_paper_style_side_by_side,
)


def main():
    """Generate paper-style plots."""
    
    # Get the project root directory (parent of plot_curve)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_dir = project_root / 'results'
    
    # Example: Plot plain networks
    plain_logs = {
        'plain-20': results_dir / 'plain20/logs/PlainNet20_20251106-161914.json',
        'plain-32': results_dir / 'plain32/logs/PlainNet32_20251106-203115.json',
        'plain-56': results_dir / 'plain56/logs/PlainNet56_20251109-190438.json',
    }
    
    # Check which plain logs exist
    existing_plain_logs = {
        name: path for name, path in plain_logs.items() if path.exists()
    }
    
    if existing_plain_logs:
        print("Plotting plain networks...")
        fig = plot_paper_style_comparison(
            existing_plain_logs,
            model_type='plain',
            save_path=str(results_dir / 'figures/plain_networks_paper_style.png'),
            iterations_per_epoch=391,  # CIFAR-10: 50000/128
        )
        print("✓ Plain networks plot saved")
    else:
        print("No plain network logs found")
    
    # Example: Plot ResNets
    resnet_logs = {
        'ResNet-20': results_dir / 'logs/ResNet20_20251106-024420.json',
        'ResNet-32': results_dir / 'resnet32/logs/ResNet32_20251106-215354.json',
        'ResNet-56': results_dir / 'resnet56/logs/ResNet56_20251109-192408.json',
        'ResNet-110': results_dir / 'logs/ResNet110_20251109-165753.json',
    }
    
    # Check which ResNet logs exist
    existing_resnet_logs = {
        name: path for name, path in resnet_logs.items() if path.exists()
    }
    
    if existing_resnet_logs:
        print("\nPlotting ResNets...")
        fig = plot_paper_style_comparison(
            existing_resnet_logs,
            model_type='resnet',
            save_path=str(results_dir / 'figures/resnet_paper_style.png'),
            iterations_per_epoch=391,
        )
        print("✓ ResNet plot saved")
    else:
        print("No ResNet logs found")
    
    # Example: Side-by-side comparison
    if existing_plain_logs and existing_resnet_logs:
        print("\nPlotting side-by-side comparison...")
        fig = plot_paper_style_side_by_side(
            existing_plain_logs,
            existing_resnet_logs,
            save_path=str(results_dir / 'figures/plain_vs_resnet_paper_style.png'),
            iterations_per_epoch=391,
        )
        print("✓ Side-by-side comparison saved")
    else:
        print(f"Skipping side-by-side (plain: {len(existing_plain_logs)}, resnet: {len(existing_resnet_logs)})")
    
    print("\n✓ All paper-style plots generated successfully!")


if __name__ == '__main__':
    main()
