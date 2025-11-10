"""
Generate Performance Summary Table for ResNet Models on CIFAR-10

This script scans experiment result files and generates a formatted table
showing the performance of different ResNet models on the CIFAR-10 test set.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import torch


def count_parameters(model) -> int:
    """Count the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def get_model_params(model_name: str, depth: int, model_type: str = 'resnet') -> int:
    """
    Calculate the number of parameters for a ResNet or PlainNet model.
    
    Args:
        model_name: Name of the model (e.g., 'ResNet20', 'PlainNet20')
        depth: Depth of the network
        model_type: Type of model ('resnet' or 'plain')
        
    Returns:
        Total number of parameters
    """
    # Import directly from the specific module file
    import importlib.util
    from pathlib import Path
    
    # Load the resnet_cifar module directly
    module_path = Path(__file__).parent / 'models' / 'resnet_cifar.py'
    spec = importlib.util.spec_from_file_location("resnet_cifar", module_path)
    resnet_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resnet_module)
    
    # Create model and count parameters
    if model_type == 'plain':
        model = resnet_module.plainnet_cifar(depth, num_classes=10)
    else:
        model = resnet_module.resnet_cifar(depth, num_classes=10)
    return count_parameters(model)


def get_best_test_accuracy(log_file: Path) -> Tuple[float, int]:
    """
    Extract the best test accuracy from a log file.
    
    Args:
        log_file: Path to the JSON log file
        
    Returns:
        Tuple of (best_accuracy, epoch_number)
    """
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    test_results = data.get('test', [])
    if not test_results:
        return 0.0, 0
    
    # Find the best accuracy
    best_acc = 0.0
    best_epoch = 0
    for epoch, result in enumerate(test_results, start=1):
        acc = result.get('accuracy', 0.0)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
    
    return best_acc, best_epoch


def find_latest_log(results_dir: Path, model_pattern: str) -> Path | None:
    """
    Find the most recent log file for a given model pattern.
    
    Args:
        results_dir: Base results directory
        model_pattern: Model name pattern (e.g., 'ResNet20', 'PlainNet20')
        
    Returns:
        Path to the latest log file, or None if not found
    """
    log_files = []
    
    # Extract base model name for directory search
    # PlainNet20 -> plain20, ResNet32 -> resnet32
    if 'PlainNet' in model_pattern:
        dir_name = model_pattern.replace('PlainNet', 'plain').lower()
    else:
        dir_name = model_pattern.lower()
    
    # Search in results/logs/ and results/model_dir/logs/
    search_dirs = [
        results_dir / 'logs',
        results_dir / dir_name / 'logs'
    ]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            pattern = f"{model_pattern}_*.json"
            log_files.extend(search_dir.glob(pattern))
    
    if not log_files:
        return None
    
    # Return the most recent file (based on timestamp in filename)
    return max(log_files, key=lambda p: p.stem.split('_')[-1])


def extract_depth_from_name(model_name: str) -> int:
    """Extract the depth number from model name (e.g., 'ResNet20' -> 20)."""
    import re
    match = re.search(r'\d+', model_name)
    return int(match.group()) if match else 0


def generate_performance_table(results_dir: str = 'results') -> None:
    """
    Generate and print the performance summary table.
    
    Args:
        results_dir: Directory containing experiment results
    """
    results_path = Path(results_dir)
    
    # Define the models we want to analyze (PlainNet and ResNet)
    models = [
        ('PlainNet20', 20, 'plain'),
        ('PlainNet32', 32, 'plain'),
        ('PlainNet56', 56, 'plain'),
        ('ResNet20', 20, 'resnet'),
        ('ResNet32', 32, 'resnet'),
        ('ResNet56', 56, 'resnet'),
        ('ResNet110', 110, 'resnet'),
    ]
    
    # Collect results
    results: List[Dict] = []
    
    for model_name, depth, model_type in models:
        print(f"Processing {model_name}...", end=' ')
        
        # Find the latest log file
        log_file = find_latest_log(results_path, model_name)
        
        if log_file is None:
            print(f"❌ No log file found")
            continue
        
        # Get best test accuracy
        best_acc, best_epoch = get_best_test_accuracy(log_file)
        
        # Calculate error rate (%)
        error_rate = (1 - best_acc) * 100
        
        # Get number of parameters
        try:
            num_params = get_model_params(model_name, depth, model_type)
            params_str = f"{num_params:,}"
        except Exception as e:
            print(f"⚠️  Could not calculate parameters: {e}")
            params_str = "N/A"
            num_params = 0
        
        results.append({
            'method': model_name,
            'layers': depth,
            'params': num_params,
            'params_str': params_str,
            'error': error_rate,
            'accuracy': best_acc * 100,
            'epoch': best_epoch,
            'log_file': log_file.name
        })
        
        print(f"✓ (Best at epoch {best_epoch}: {best_acc*100:.2f}% accuracy)")
    
    # Print the table
    print("\n" + "="*85)
    print("Performance Summary - CIFAR-10 Test Set (Best Results)")
    print("="*85)
    print()
    
    # Table header
    header = f"{'Method':<15} {'# Layers':>10} {'# Params':>15} {'Error (%)':>12} {'Accuracy (%)':>14}"
    print(header)
    print("-" * 85)
    
    # Table rows
    for result in results:
        row = (f"{result['method']:<15} "
               f"{result['layers']:>10} "
               f"{result['params_str']:>15} "
               f"{result['error']:>12.2f} "
               f"{result['accuracy']:>14.2f}")
        print(row)
    
    print("-" * 85)
    print()
    
    # Additional details
    print("Details:")
    for result in results:
        print(f"  • {result['method']}: Best accuracy at epoch {result['epoch']} "
              f"(from {result['log_file']})")
    print()
    
    # Save to file
    output_file = Path(results_dir) / 'performance_summary.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*85 + "\n")
        f.write("Performance Summary - CIFAR-10 Test Set (Best Results)\n")
        f.write("="*85 + "\n\n")
        f.write(header + "\n")
        f.write("-" * 85 + "\n")
        for result in results:
            row = (f"{result['method']:<15} "
                   f"{result['layers']:>10} "
                   f"{result['params_str']:>15} "
                   f"{result['error']:>12.2f} "
                   f"{result['accuracy']:>14.2f}")
            f.write(row + "\n")
        f.write("-" * 85 + "\n\n")
        f.write("Details:\n")
        for result in results:
            f.write(f"  • {result['method']}: Best accuracy at epoch {result['epoch']} "
                   f"(from {result['log_file']})\n")
        f.write("\n")
    
    print(f"✓ Summary saved to: {output_file}")
    print()


if __name__ == '__main__':
    import sys
    
    # Allow custom results directory as command line argument
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'
    
    print("="*80)
    print("ResNet and PlainNet Performance Summary Generator")
    print("="*80)
    print(f"Scanning directory: {results_dir}")
    print()
    
    generate_performance_table(results_dir)
