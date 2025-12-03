#!/usr/bin/env python3
"""
Plot Training Curves

This script plots training loss and validation accuracy curves from
the training_curves.json file generated during training.

Usage:
    python scripts/plot_curves.py outputs/longrun/training_curves.json
    python scripts/plot_curves.py outputs/longrun/training_curves.json --save curves.png
"""

import json
import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_curves(curves_path: str, save_path: str = None):
    """
    Plot training curves from JSON file.

    Args:
        curves_path: Path to training_curves.json
        save_path: Optional path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Install with: pip install matplotlib")
        print("\nCurve Summary:")
        with open(curves_path) as f:
            curves = json.load(f)
        print(f"  Total steps: {curves['steps'][-1] if curves['steps'] else 0}")
        print(f"  Final loss: {curves['train_loss'][-1] if curves['train_loss'] else 'N/A'}")
        print(f"  Best accuracy: {curves['best_accuracy']:.2%}")
        print(f"  Best step: {curves['best_step']}")
        return

    # Load curves
    with open(curves_path) as f:
        curves = json.load(f)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot training loss
    ax1.plot(curves['steps'], curves['train_loss'], alpha=0.3, color='blue', label='Loss')
    if curves['train_loss_smooth']:
        ax1.plot(curves['steps'], curves['train_loss_smooth'], color='blue', linewidth=2, label='Loss (smoothed)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot validation accuracy
    if curves['valid_steps'] and curves['valid_accuracy']:
        ax2.plot(curves['valid_steps'], curves['valid_accuracy'], 'o-', color='green', linewidth=2, markersize=4)
        ax2.axhline(y=curves['best_accuracy'], color='red', linestyle='--', label=f'Best: {curves["best_accuracy"]:.2%}')
        ax2.axvline(x=curves['best_step'], color='red', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Add overall title
    fig.suptitle(f'Training Curves (Best: {curves["best_accuracy"]:.2%} @ step {curves["best_step"]})', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument(
        "curves_file",
        type=str,
        help="Path to training_curves.json",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save the figure (optional)",
    )

    args = parser.parse_args()

    if not Path(args.curves_file).exists():
        print(f"Error: File not found: {args.curves_file}")
        return

    plot_curves(args.curves_file, args.save)


if __name__ == "__main__":
    main()
