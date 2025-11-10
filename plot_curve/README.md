# Examples

This directory contains example scripts demonstrating how to use the ResNet reproduction project.

## plot_training_results.py

Generates comprehensive training curve visualizations from experiment logs.

### Usage

```bash
python examples/plot_training_results.py
```

This will generate the following plots in `results/plots/`:

1. **figure6_plain20_vs_resnet20_error.png** - Reproduces Figure 6 from the paper showing error rates
2. **plain20_vs_resnet20_accuracy.png** - Accuracy comparison between Plain and ResNet
3. **plain32_vs_resnet32_error.png** - 32-layer network comparison
4. **plain20_vs_resnet20_loss.png** - Training and test loss curves
5. **resnet20_detailed.png** - Detailed metrics for a single model
6. **all_models_test_accuracy.png** - Test accuracy comparison across all models
7. **all_models_train_accuracy.png** - Training accuracy comparison

## Custom Plotting

You can also create custom plots using the utilities directly:

```python
from utils.plot_curves import plot_training_curves

# Compare two models
plot_training_curves(
    plain_log='path/to/plain_log.json',
    resnet_log='path/to/resnet_log.json',
    save_path='my_comparison.png',
    title='My Custom Comparison',
    plot_error=True  # Show error rates instead of accuracy
)
```

See `utils/plot_curves.py` for all available functions and parameters.
