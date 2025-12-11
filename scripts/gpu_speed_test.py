#!/usr/bin/env python3
"""
Quick GPU Speed Test for OKADFA

Compare training speed on CPU vs GPU with a small model.
"""

import sys
sys.path.insert(0, 'src')

import torch
import time
from models.okadfa_model import OKADFAModel
from data.tokenizer import get_tokenizer

def create_dummy_batch(batch_size, seq_len, vocab_size):
    """Create a dummy batch for testing."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    return input_ids, targets

def benchmark_device(device_name, num_steps=50):
    """Benchmark training on a specific device."""
    
    device = torch.device(device_name)
    print(f"\n{'='*70}")
    print(f"  Benchmarking on {device_name.upper()}")
    print(f"{'='*70}\n")
    
    # Model config (small for quick test)
    vocab_size = 50257
    d_model = 256
    num_layers = 2
    num_heads = 4
    d_ff = 1024
    seq_len = 128  # Shorter for speed
    batch_size = 4
    
    print(f"Configuration:")
    print(f"  Model: {d_model}d, {num_layers} layers, {num_heads} heads")
    print(f"  Batch: {batch_size} x {seq_len} tokens")
    print(f"  Steps: {num_steps}")
    print()
    
    # Create model
    print("Creating model...", end=" ", flush=True)
    model = OKADFAModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=seq_len,
        num_random_features=256
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ ({num_params:,} parameters)")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Warmup (not timed)
    print("Warming up...", end=" ", flush=True)
    for _ in range(3):
        input_ids, targets = create_dummy_batch(batch_size, seq_len, vocab_size)
        input_ids, targets = input_ids.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, vocab_size), 
            targets.view(-1)
        )
        loss.backward()
        optimizer.step()
    
    if device_name == 'cuda':
        torch.cuda.synchronize()
    print("✓")
    
    # Actual benchmark
    print(f"\nRunning {num_steps} training steps...")
    
    start_time = time.time()
    step_times = []
    
    for step in range(num_steps):
        step_start = time.time()
        
        # Forward pass
        input_ids, targets = create_dummy_batch(batch_size, seq_len, vocab_size)
        input_ids, targets = input_ids.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, vocab_size), 
            targets.view(-1)
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if device_name == 'cuda':
            torch.cuda.synchronize()
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        if (step + 1) % 10 == 0:
            avg_time = sum(step_times[-10:]) / len(step_times[-10:])
            print(f"  Step {step+1}/{num_steps} | {avg_time*1000:.1f} ms/step | "
                  f"{(batch_size * seq_len) / avg_time:.0f} tokens/sec")
    
    total_time = time.time() - start_time
    avg_step_time = sum(step_times) / len(step_times)
    
    print(f"\n{'─'*70}")
    print(f"Results:")
    print(f"  Total time:      {total_time:.2f} seconds")
    print(f"  Avg step time:   {avg_step_time*1000:.1f} ms")
    print(f"  Throughput:      {(batch_size * seq_len) / avg_step_time:.0f} tokens/sec")
    print(f"  Steps/second:    {1/avg_step_time:.2f}")
    
    if device_name == 'cuda':
        max_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak GPU memory: {max_memory:.1f} MB")
        torch.cuda.reset_peak_memory_stats()
    
    return {
        'device': device_name,
        'total_time': total_time,
        'avg_step_time': avg_step_time,
        'tokens_per_sec': (batch_size * seq_len) / avg_step_time,
        'steps_per_sec': 1 / avg_step_time
    }

def main():
    print("\n" + "="*70)
    print("  OKADFA GPU Speed Test")
    print("="*70)
    
    results = {}
    
    # Test CPU
    print("\n🖥️  Testing CPU performance...")
    try:
        results['cpu'] = benchmark_device('cpu', num_steps=50)
    except Exception as e:
        print(f"❌ CPU test failed: {e}")
        return
    
    # Test GPU
    if torch.cuda.is_available():
        print("\n🎮 Testing GPU performance...")
        try:
            results['gpu'] = benchmark_device('cuda', num_steps=50)
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
            # Clean up
            torch.cuda.empty_cache()
            return
    else:
        print("\n⚠️  No GPU available, skipping GPU test")
    
    # Comparison
    if 'cpu' in results and 'gpu' in results:
        print(f"\n{'='*70}")
        print("  COMPARISON")
        print(f"{'='*70}\n")
        
        cpu_time = results['cpu']['avg_step_time']
        gpu_time = results['gpu']['avg_step_time']
        speedup = cpu_time / gpu_time
        
        print(f"CPU:")
        print(f"  Step time:    {cpu_time*1000:.1f} ms")
        print(f"  Throughput:   {results['cpu']['tokens_per_sec']:.0f} tokens/sec")
        print()
        print(f"GPU:")
        print(f"  Step time:    {gpu_time*1000:.1f} ms")
        print(f"  Throughput:   {results['gpu']['tokens_per_sec']:.0f} tokens/sec")
        print()
        print(f"🚀 GPU Speedup:  {speedup:.1f}x faster!")
        print()
        
        if speedup > 5:
            print("✅ Excellent speedup! GPU is well-utilized.")
        elif speedup > 2:
            print("✅ Good speedup. GPU training recommended.")
        else:
            print("⚠️  Modest speedup. Model may be too small to benefit from GPU.")
    
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
