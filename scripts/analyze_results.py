#!/usr/bin/env python3
"""
Analyze OKADFA training results and generate visualizations.

This script parses training logs and creates plots showing:
- Training loss over time
- Validation perplexity over time
- Orthogonality loss trends
- Learning rate schedule

Usage:
    python scripts/analyze_results.py --log_file wikitext_quick_test.log
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
import json


def parse_training_log(log_path: Path) -> Dict[str, List]:
    """
    Parse a training log file and extract metrics.
    
    Args:
        log_path: Path to the log file
        
    Returns:
        Dictionary containing lists of metrics:
        - steps: List of step numbers
        - train_loss: List of training losses
        - lm_loss: List of language modeling losses
        - ortho_loss: List of orthogonality losses
        - val_loss: List of validation losses
        - val_ppl: List of validation perplexities
        - learning_rate: List of learning rates
        - grad_norm: List of gradient norms
    """
    metrics = {
        'steps': [],
        'train_loss': [],
        'lm_loss': [],
        'ortho_loss': [],
        'val_loss': [],
        'val_ppl': [],
        'learning_rate': [],
        'grad_norm': []
    }
    
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return metrics
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Parse training steps
    # Format: "Step 100/100 | Loss: 10.4900 | LM: 10.4644 | Ortho: 0.0256 | LR: 3.00e-05 | Grad: 1.50"
    train_pattern = r'Step (\d+)/\d+ \| Loss: ([\d.]+) \| LM: ([\d.]+) \| Ortho: ([\d.]+) \| LR: ([\d.e+-]+) \| Grad: ([\d.]+)'
    for match in re.finditer(train_pattern, content):
        step, loss, lm_loss, ortho_loss, lr, grad = match.groups()
        metrics['steps'].append(int(step))
        metrics['train_loss'].append(float(loss))
        metrics['lm_loss'].append(float(lm_loss))
        metrics['ortho_loss'].append(float(ortho_loss))
        metrics['learning_rate'].append(float(lr))
        metrics['grad_norm'].append(float(grad))
    
    # Parse evaluation results
    # Format: "Evaluation at step 50:\n  Val Loss: 10.8407\n  Val PPL: 51055.58"
    eval_pattern = r'Evaluation at step (\d+):\s+Val Loss: ([\d.]+)\s+Val PPL: ([\d.]+)'
    eval_steps = []
    for match in re.finditer(eval_pattern, content):
        step, val_loss, val_ppl = match.groups()
        eval_steps.append((int(step), float(val_loss), float(val_ppl)))
    
    # Add evaluation metrics at the correct steps
    for step, val_loss, val_ppl in eval_steps:
        # Find where to insert (keep sorted by step)
        insert_idx = 0
        for i, s in enumerate(metrics['steps']):
            if s == step:
                insert_idx = i
                break
        
        # Extend val metrics with None up to this point if needed
        while len(metrics['val_loss']) < insert_idx:
            metrics['val_loss'].append(None)
            metrics['val_ppl'].append(None)
        
        metrics['val_loss'].append(val_loss)
        metrics['val_ppl'].append(val_ppl)
    
    # Fill remaining with None
    while len(metrics['val_loss']) < len(metrics['steps']):
        metrics['val_loss'].append(None)
        metrics['val_ppl'].append(None)
    
    return metrics


def print_summary(metrics: Dict[str, List], log_path: Path):
    """Print a summary of the training run."""
    
    print(f"\n{'='*80}")
    print(f"OKADFA Training Analysis: {log_path.name}")
    print(f"{'='*80}\n")
    
    if not metrics['steps']:
        print("No training data found in log file.")
        return
    
    # Basic info
    total_steps = max(metrics['steps']) if metrics['steps'] else 0
    print(f"Training Steps: {total_steps}")
    print(f"Log entries: {len(metrics['steps'])}")
    
    # Training loss
    if metrics['train_loss']:
        print(f"\nTraining Loss:")
        print(f"  Initial: {metrics['train_loss'][0]:.4f}")
        print(f"  Final:   {metrics['train_loss'][-1]:.4f}")
        print(f"  Change:  {metrics['train_loss'][-1] - metrics['train_loss'][0]:+.4f}")
    
    # LM Loss
    if metrics['lm_loss']:
        print(f"\nLanguage Modeling Loss:")
        print(f"  Initial: {metrics['lm_loss'][0]:.4f}")
        print(f"  Final:   {metrics['lm_loss'][-1]:.4f}")
        print(f"  Change:  {metrics['lm_loss'][-1] - metrics['lm_loss'][0]:+.4f}")
    
    # Orthogonality Loss
    if metrics['ortho_loss']:
        print(f"\nOrthogonality Loss:")
        print(f"  Initial: {metrics['ortho_loss'][0]:.6f}")
        print(f"  Final:   {metrics['ortho_loss'][-1]:.6f}")
        print(f"  Min:     {min(metrics['ortho_loss']):.6f}")
        print(f"  Max:     {max(metrics['ortho_loss']):.6f}")
    
    # Validation metrics
    val_losses = [v for v in metrics['val_loss'] if v is not None]
    val_ppls = [v for v in metrics['val_ppl'] if v is not None]
    
    if val_losses:
        print(f"\nValidation Loss:")
        print(f"  Initial: {val_losses[0]:.4f}")
        print(f"  Final:   {val_losses[-1]:.4f}")
        print(f"  Best:    {min(val_losses):.4f}")
        print(f"  Change:  {val_losses[-1] - val_losses[0]:+.4f}")
    
    if val_ppls:
        print(f"\nValidation Perplexity:")
        print(f"  Initial: {val_ppls[0]:,.2f}")
        print(f"  Final:   {val_ppls[-1]:,.2f}")
        print(f"  Best:    {min(val_ppls):,.2f}")
        improvement = (1 - val_ppls[-1] / val_ppls[0]) * 100
        print(f"  Improvement: {improvement:.1f}%")
    
    # Learning rate
    if metrics['learning_rate']:
        print(f"\nLearning Rate:")
        print(f"  Initial: {metrics['learning_rate'][0]:.2e}")
        print(f"  Final:   {metrics['learning_rate'][-1]:.2e}")
    
    # Gradient norm
    if metrics['grad_norm']:
        print(f"\nGradient Norm:")
        print(f"  Mean: {sum(metrics['grad_norm']) / len(metrics['grad_norm']):.2f}")
        print(f"  Min:  {min(metrics['grad_norm']):.2f}")
        print(f"  Max:  {max(metrics['grad_norm']):.2f}")
    
    print(f"\n{'='*80}\n")


def save_metrics_json(metrics: Dict[str, List], output_path: Path):
    """Save metrics to JSON file for further analysis."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {output_path}")


def create_plots(metrics: Dict[str, List], output_dir: Path):
    """
    Create matplotlib plots of training metrics.
    
    Note: Only creates plots if matplotlib is available.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\nMatplotlib not available. Skipping plots.")
        print("Install with: pip install matplotlib")
        return
    
    if not metrics['steps']:
        print("No data to plot.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    # Plot 1: Training losses
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('OKADFA Training Metrics', fontsize=16, fontweight='bold')
    
    # LM Loss
    ax = axes[0, 0]
    ax.plot(metrics['steps'], metrics['lm_loss'], 'b-', linewidth=2, label='Training LM Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Language Modeling Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Orthogonality Loss
    ax = axes[0, 1]
    ax.plot(metrics['steps'], metrics['ortho_loss'], 'r-', linewidth=2, label='Orthogonality Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Orthogonality Regularization Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Validation Perplexity
    ax = axes[1, 0]
    val_steps = [s for s, v in zip(metrics['steps'], metrics['val_ppl']) if v is not None]
    val_ppls = [v for v in metrics['val_ppl'] if v is not None]
    if val_ppls:
        ax.plot(val_steps, val_ppls, 'g-o', linewidth=2, markersize=8, label='Val Perplexity')
        ax.set_xlabel('Step')
        ax.set_ylabel('Perplexity')
        ax.set_title('Validation Perplexity')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Learning Rate
    ax = axes[1, 1]
    ax.plot(metrics['steps'], metrics['learning_rate'], 'm-', linewidth=2, label='Learning Rate')
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    plot_path = output_dir / 'training_metrics.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    # Plot 2: Combined losses
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(metrics['steps'], metrics['lm_loss'], 'b-', linewidth=2, label='LM Loss', alpha=0.8)
    
    # Add validation loss on same plot
    val_steps = [s for s, v in zip(metrics['steps'], metrics['val_loss']) if v is not None]
    val_losses = [v for v in metrics['val_loss'] if v is not None]
    if val_losses:
        ax.plot(val_steps, val_losses, 'go-', linewidth=2, markersize=10, 
                label='Val Loss', alpha=0.8)
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('OKADFA Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    plot_path = output_dir / 'loss_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    plt.close('all')


def main():
    parser = argparse.ArgumentParser(
        description='Analyze OKADFA training results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default='wikitext_quick_test.log',
        help='Path to training log file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='analysis',
        help='Directory to save plots and metrics'
    )
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Skip creating plots'
    )
    
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    output_dir = Path(args.output_dir)
    
    # Parse log file
    print(f"Parsing log file: {log_path}")
    metrics = parse_training_log(log_path)
    
    # Print summary
    print_summary(metrics, log_path)
    
    # Save metrics as JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{log_path.stem}_metrics.json"
    save_metrics_json(metrics, json_path)
    
    # Create plots
    if not args.no_plots:
        print("\nGenerating plots...")
        create_plots(metrics, output_dir)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
