#!/usr/bin/env python3
"""
Model Size Calculator for OKADFA

Calculate parameter counts and memory requirements for different model configurations.
Helps plan experiments and choose appropriate hardware.
"""

import argparse
from typing import Dict, Tuple


def calculate_model_size(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int = 1024
) -> Dict[str, any]:
    """
    Calculate model parameters and memory requirements.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        max_seq_len: Maximum sequence length
        
    Returns:
        Dictionary with parameter counts and memory estimates
    """
    
    # Embedding layer
    token_embedding = vocab_size * d_model
    pos_embedding = max_seq_len * d_model
    embedding_params = token_embedding + pos_embedding
    
    # Attention layer (per layer)
    # Q, K, V projections
    qkv_params = 3 * (d_model * d_model)
    # Output projection
    out_proj = d_model * d_model
    # Layer norms (2 per layer: after attention + after FF)
    ln_params = 2 * (d_model + d_model)  # gamma + beta
    
    attention_params_per_layer = qkv_params + out_proj
    
    # Feed-forward layer (per layer)
    ff_params_per_layer = (d_model * d_ff) + (d_ff * d_model) + d_ff + d_model  # W1, W2, biases
    
    # Total per layer
    params_per_layer = attention_params_per_layer + ff_params_per_layer + ln_params
    
    # All transformer layers
    transformer_params = params_per_layer * num_layers
    
    # Output projection (LM head)
    output_proj = vocab_size * d_model
    
    # Total parameters
    total_params = embedding_params + transformer_params + output_proj
    
    # Memory calculations (assuming float32 = 4 bytes)
    bytes_per_param = 4
    
    # Model weights
    model_memory_mb = (total_params * bytes_per_param) / (1024 ** 2)
    
    # Activations during training (rough estimate)
    # Batch size * seq_len * d_model * num_layers * 4 (forward + backward)
    batch_size = 4  # default
    activation_memory_mb = (batch_size * max_seq_len * d_model * num_layers * 4 * bytes_per_param) / (1024 ** 2)
    
    # Optimizer state (AdamW needs 2x parameters for momentum)
    optimizer_memory_mb = 2 * model_memory_mb
    
    # Total training memory
    total_train_memory_mb = model_memory_mb + activation_memory_mb + optimizer_memory_mb
    
    return {
        'parameters': {
            'embeddings': embedding_params,
            'transformer': transformer_params,
            'output': output_proj,
            'total': total_params,
            'per_layer': params_per_layer
        },
        'memory_mb': {
            'model': model_memory_mb,
            'activations': activation_memory_mb,
            'optimizer': optimizer_memory_mb,
            'total_training': total_train_memory_mb
        },
        'config': {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'd_ff': d_ff,
            'max_seq_len': max_seq_len
        }
    }


def print_model_info(name: str, info: Dict):
    """Print formatted model information."""
    print(f"\n{'=' * 80}")
    print(f"  {name}")
    print(f"{'=' * 80}")
    
    # Configuration
    print(f"\n📐 CONFIGURATION")
    print(f"{'─' * 80}")
    cfg = info['config']
    print(f"  Vocabulary:      {cfg['vocab_size']:,}")
    print(f"  Model Dim:       {cfg['d_model']}")
    print(f"  Layers:          {cfg['num_layers']}")
    print(f"  Attention Heads: {cfg['num_heads']}")
    print(f"  FF Dim:          {cfg['d_ff']}")
    print(f"  Max Seq Length:  {cfg['max_seq_len']}")
    
    # Parameters
    print(f"\n🔢 PARAMETERS")
    print(f"{'─' * 80}")
    params = info['parameters']
    print(f"  Embeddings:      {params['embeddings']:>15,} ({params['embeddings']/params['total']*100:5.1f}%)")
    print(f"  Transformer:     {params['transformer']:>15,} ({params['transformer']/params['total']*100:5.1f}%)")
    print(f"  Output Head:     {params['output']:>15,} ({params['output']/params['total']*100:5.1f}%)")
    print(f"  ────────────────────────────────────")
    print(f"  TOTAL:           {params['total']:>15,}")
    print(f"  Per Layer:       {params['per_layer']:>15,}")
    
    # Memory
    print(f"\n💾 MEMORY REQUIREMENTS (Batch Size 4)")
    print(f"{'─' * 80}")
    mem = info['memory_mb']
    print(f"  Model Weights:   {mem['model']:>10.1f} MB  ({mem['model']/1024:6.2f} GB)")
    print(f"  Activations:     {mem['activations']:>10.1f} MB  ({mem['activations']/1024:6.2f} GB)")
    print(f"  Optimizer State: {mem['optimizer']:>10.1f} MB  ({mem['optimizer']/1024:6.2f} GB)")
    print(f"  ────────────────────────────────────────────────────")
    print(f"  TOTAL TRAINING:  {mem['total_training']:>10.1f} MB  ({mem['total_training']/1024:6.2f} GB)")
    
    # Hardware recommendations
    print(f"\n🖥️  HARDWARE RECOMMENDATIONS")
    print(f"{'─' * 80}")
    total_gb = mem['total_training'] / 1024
    
    if total_gb < 4:
        print(f"  ✅ Can run on CPU")
        print(f"  ✅ Can run on most GPUs (>4GB)")
    elif total_gb < 8:
        print(f"  ⚠️  CPU training will be slow")
        print(f"  ✅ Recommended: GPU with 8GB+ VRAM")
    elif total_gb < 16:
        print(f"  ❌ CPU training not recommended")
        print(f"  ✅ Recommended: GPU with 16GB+ VRAM (e.g., RTX 3090, V100)")
    else:
        print(f"  ❌ Requires high-end GPU")
        print(f"  ✅ Recommended: A100 (40GB+) or multi-GPU setup")
    
    # Training time estimate (very rough)
    print(f"\n⏱️  ESTIMATED TRAINING TIME (1000 steps)")
    print(f"{'─' * 80}")
    
    # Base on parameter count (rough heuristic)
    params_millions = params['total'] / 1_000_000
    if params_millions < 50:
        print(f"  CPU (16 cores):  ~30-60 minutes")
        print(f"  GPU (RTX 3070):  ~5-10 minutes")
    elif params_millions < 150:
        print(f"  CPU (16 cores):  ~2-4 hours")
        print(f"  GPU (RTX 3070):  ~15-30 minutes")
    else:
        print(f"  CPU (16 cores):  ~8-16 hours")
        print(f"  GPU (A100):      ~30-60 minutes")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate OKADFA model size and memory requirements',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--preset',
        choices=['tiny', 'small', 'base', 'gpt2-small', 'gpt2-medium', 'custom'],
        default=None,
        help='Use a preset configuration'
    )
    
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feed-forward dimension')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--compare', action='store_true', help='Compare multiple configurations')
    
    args = parser.parse_args()
    
    # Define presets
    presets = {
        'tiny': {
            'name': 'OKADFA Tiny (Quick Testing)',
            'vocab_size': 50257,
            'd_model': 128,
            'num_layers': 2,
            'num_heads': 2,
            'd_ff': 512,
            'max_seq_len': 256
        },
        'small': {
            'name': 'OKADFA Small (Default)',
            'vocab_size': 50257,
            'd_model': 256,
            'num_layers': 2,
            'num_heads': 4,
            'd_ff': 1024,
            'max_seq_len': 256
        },
        'base': {
            'name': 'OKADFA Base',
            'vocab_size': 50257,
            'd_model': 512,
            'num_layers': 6,
            'num_heads': 8,
            'd_ff': 2048,
            'max_seq_len': 512
        },
        'gpt2-small': {
            'name': 'OKADFA GPT-2 Small',
            'vocab_size': 50257,
            'd_model': 768,
            'num_layers': 12,
            'num_heads': 12,
            'd_ff': 3072,
            'max_seq_len': 1024
        },
        'gpt2-medium': {
            'name': 'OKADFA GPT-2 Medium',
            'vocab_size': 50257,
            'd_model': 1024,
            'num_layers': 24,
            'num_heads': 16,
            'd_ff': 4096,
            'max_seq_len': 1024
        }
    }
    
    if args.compare:
        # Show all presets
        print("\n" + "=" * 80)
        print("  OKADFA MODEL SIZE COMPARISON")
        print("=" * 80)
        
        for preset_name in ['tiny', 'small', 'base', 'gpt2-small', 'gpt2-medium']:
            preset = presets[preset_name]
            config = {k: v for k, v in preset.items() if k != 'name'}
            info = calculate_model_size(**config)
            print_model_info(preset['name'], info)
        
        print("\n" + "=" * 80 + "\n")
    
    elif args.preset:
        # Show specific preset
        if args.preset == 'custom':
            config = {
                'vocab_size': args.vocab_size,
                'd_model': args.d_model,
                'num_layers': args.num_layers,
                'num_heads': args.num_heads,
                'd_ff': args.d_ff,
                'max_seq_len': args.max_seq_len
            }
            info = calculate_model_size(**config)
            print_model_info('Custom OKADFA Configuration', info)
        else:
            preset = presets[args.preset]
            config = {k: v for k, v in preset.items() if k != 'name'}
            info = calculate_model_size(**config)
            print_model_info(preset['name'], info)
        
        print()
    
    else:
        # Show default (small)
        preset = presets['small']
        config = {k: v for k, v in preset.items() if k != 'name'}
        info = calculate_model_size(**config)
        print_model_info(preset['name'], info)
        
        print(f"\n💡 TIP: Use --compare to see all configurations")
        print(f"   Or:  --preset [tiny|small|base|gpt2-small|gpt2-medium]")
        print()


if __name__ == '__main__':
    main()
