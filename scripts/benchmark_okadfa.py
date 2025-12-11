"""
Benchmark OKADFA vs Standard Transformer on WikiText-2.

This script trains both models side-by-side and compares:
- Training speed (samples/sec, tokens/sec)
- Memory usage (peak GPU/CPU memory)
- Model quality (perplexity, loss convergence)
- Computational efficiency (FLOPs, attention complexity)

Copyright (c) 2025 Gregory Ward - SmartLedger.Technology
Licensed under the MIT License. See LICENSE file in the project root.

Usage:
    # Quick benchmark (short training)
    python scripts/benchmark_okadfa.py --quick_test
    
    # Full benchmark (1000 steps)
    python scripts/benchmark_okadfa.py --max_steps 1000 --device cuda
    
    # Extended benchmark (10K steps)
    python scripts/benchmark_okadfa.py --max_steps 10000 --device cuda --save_results
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import psutil
import gc

from models.okadfa_model import OKADFAModel
from models.dfa_transformer_block import DFATransformerBlock, FeedForward
from training.dfa_feedback import DFAFeedbackMatrix
from training.dfa_backward import DFABackwardHook
from training.orthogonality_loss import OrthogonalityLoss
from data.wikitext_loader import create_wikitext_dataloaders
from data.tokenizer import get_tokenizer


@dataclass
class BenchmarkMetrics:
    """Metrics for a single training run."""
    model_name: str
    num_parameters: int
    device: str
    
    # Training metrics
    training_time: float
    samples_per_sec: float
    tokens_per_sec: float
    steps_completed: int
    
    # Loss metrics
    initial_train_loss: float
    final_train_loss: float
    initial_val_loss: float
    final_val_loss: float
    initial_val_ppl: float
    final_val_ppl: float
    
    # Memory metrics
    peak_memory_mb: float
    avg_memory_mb: float
    
    # Computational metrics
    avg_step_time: float
    avg_forward_time: float
    avg_backward_time: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: str):
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class StandardTransformer(nn.Module):
    """
    Standard transformer for comparison.
    
    Uses regular softmax attention (no kernelization) and
    standard backpropagation (no DFA).
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        use_causal: bool = True,
        tie_weights: bool = True,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding (learned)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        if tie_weights:
            self.output_projection.weight = self.token_embedding.weight
        
        # Create causal mask
        self.use_causal = use_causal
        if use_causal:
            mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
            self.register_buffer('causal_mask', mask.bool())
        
        # Initialize
        self._init_weights()
        self.num_parameters = sum(p.numel() for p in self.parameters())
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        x = self.token_embedding(input_ids)  # [batch, seq_len, d_model]
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Transformer with causal mask
        if self.use_causal:
            mask = self.causal_mask[:seq_len, :seq_len]
            x = self.transformer(x, mask=mask, is_causal=True)
        else:
            x = self.transformer(x)
        
        # Output
        x = self.output_norm(x)
        logits = self.output_projection(x)
        
        return logits


class ModelBenchmarker:
    """Benchmark a single model."""
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 3e-4,
        max_steps: int = 1000,
        warmup_steps: int = 100,
        use_dfa: bool = False,
        use_ortho: bool = False,
    ):
        self.model = model.to(device)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_steps = max_steps
        self.use_dfa = use_dfa
        self.use_ortho = use_ortho
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_steps - warmup_steps,
            eta_min=learning_rate * 0.1,
        )
        
        self.warmup_steps = warmup_steps
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Setup DFA if needed
        if use_dfa and hasattr(model, 'get_dfa_modules'):
            self._setup_dfa()
        
        # Setup orthogonality loss if needed
        if use_ortho:
            self.ortho_loss_fn = OrthogonalityLoss(
                lambda_max=1e-4,
                warmup_steps=int(0.1 * max_steps),
            )
        
        # Metrics
        self.step = 0
        self.memory_samples = []
        self.step_times = []
        self.forward_times = []
        self.backward_times = []
    
    def _setup_dfa(self):
        """Setup DFA hooks."""
        dfa_modules = self.model.get_dfa_modules()
        if not dfa_modules:
            return
        
        output_dim = self.model.output_projection.out_features
        layer_dims = [
            m.out_features if hasattr(m, 'out_features')
            else m.embed_dim if hasattr(m, 'embed_dim')
            else m.weight.shape[0]
            for m in dfa_modules
        ]
        
        self.feedback_matrix = DFAFeedbackMatrix(
            layer_dims=layer_dims,
            output_dim=output_dim,
        ).to(self.device)
        
        self.dfa_hook = DFABackwardHook(self.feedback_matrix)
        self.dfa_hook.register_hooks(dfa_modules, list(range(len(dfa_modules))))
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.device == 'cuda':
            return torch.cuda.max_memory_allocated() / 1024**2
        else:
            return psutil.Process().memory_info().rss / 1024**2
    
    def train_step(self, batch) -> Tuple[float, float, float]:
        """
        Execute one training step.
        
        Returns:
            loss, forward_time, backward_time
        """
        self.model.train()
        
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        
        # Forward pass
        t0 = time.time()
        logits = self.model(input_ids)
        forward_time = time.time() - t0
        
        # Loss
        loss = self.loss_fn(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1)
        )
        
        # Add orthogonality loss if enabled
        if self.use_ortho and hasattr(self.model, 'get_orthogonality_loss'):
            ortho_loss = self.model.get_orthogonality_loss(self.ortho_loss_fn)
            loss = loss + ortho_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        t1 = time.time()
        loss.backward()
        backward_time = time.time() - t1
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Warmup
        if self.step < self.warmup_steps:
            warmup_factor = (self.step + 1) / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * warmup_factor
        
        self.optimizer.step()
        
        if self.step >= self.warmup_steps:
            self.scheduler.step()
        
        self.step += 1
        
        return loss.item(), forward_time, backward_time
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        for batch in self.val_loader:
            input_ids, target_ids = batch
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits = self.model(input_ids)
            loss = self.loss_fn(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )
            
            total_loss += loss.item() * target_ids.numel()
            total_tokens += target_ids.numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(min(avg_loss, 20))
        
        return avg_loss, perplexity
    
    def benchmark(self) -> BenchmarkMetrics:
        """Run full benchmark."""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {self.model_name}")
        print(f"Parameters: {self.model.num_parameters:,}")
        print(f"Device: {self.device}")
        print(f"Steps: {self.max_steps}")
        print(f"{'='*60}\n")
        
        # Initial evaluation
        print("Initial evaluation...")
        initial_val_loss, initial_val_ppl = self.evaluate()
        print(f"  Val Loss: {initial_val_loss:.4f}, PPL: {initial_val_ppl:.2f}")
        
        # Training
        print(f"\nTraining for {self.max_steps} steps...")
        train_iter = iter(self.train_loader)
        
        start_time = time.time()
        total_tokens = 0
        total_samples = 0
        initial_train_loss = None
        final_train_loss = None
        
        for step in range(self.max_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            step_start = time.time()
            loss, fwd_time, bwd_time = self.train_step(batch)
            step_time = time.time() - step_start
            
            if initial_train_loss is None:
                initial_train_loss = loss
            final_train_loss = loss
            
            # Track metrics
            self.step_times.append(step_time)
            self.forward_times.append(fwd_time)
            self.backward_times.append(bwd_time)
            self.memory_samples.append(self._get_memory_usage())
            
            # Count tokens/samples
            batch_size, seq_len = batch[0].shape
            total_samples += batch_size
            total_tokens += batch_size * seq_len
            
            # Log progress
            if (step + 1) % 100 == 0:
                avg_time = sum(self.step_times[-100:]) / 100
                samples_sec = batch_size / avg_time
                tokens_sec = (batch_size * seq_len) / avg_time
                print(
                    f"  Step {step+1}/{self.max_steps} | "
                    f"Loss: {loss:.4f} | "
                    f"Speed: {samples_sec:.1f} samp/s, {tokens_sec:.0f} tok/s | "
                    f"Time: {avg_time:.3f}s"
                )
        
        training_time = time.time() - start_time
        
        # Final evaluation
        print("\nFinal evaluation...")
        final_val_loss, final_val_ppl = self.evaluate()
        print(f"  Val Loss: {final_val_loss:.4f}, PPL: {final_val_ppl:.2f}")
        
        # Compute metrics
        metrics = BenchmarkMetrics(
            model_name=self.model_name,
            num_parameters=self.model.num_parameters,
            device=self.device,
            training_time=training_time,
            samples_per_sec=total_samples / training_time,
            tokens_per_sec=total_tokens / training_time,
            steps_completed=self.max_steps,
            initial_train_loss=initial_train_loss,
            final_train_loss=final_train_loss,
            initial_val_loss=initial_val_loss,
            final_val_loss=final_val_loss,
            initial_val_ppl=initial_val_ppl,
            final_val_ppl=final_val_ppl,
            peak_memory_mb=max(self.memory_samples),
            avg_memory_mb=sum(self.memory_samples) / len(self.memory_samples),
            avg_step_time=sum(self.step_times) / len(self.step_times),
            avg_forward_time=sum(self.forward_times) / len(self.forward_times),
            avg_backward_time=sum(self.backward_times) / len(self.backward_times),
        )
        
        print(f"\n{self.model_name} Results:")
        print(f"  Training Time: {training_time:.2f}s")
        print(f"  Speed: {metrics.samples_per_sec:.1f} samples/sec")
        print(f"  Memory: {metrics.peak_memory_mb:.1f} MB (peak)")
        print(f"  Val PPL: {initial_val_ppl:.2f} → {final_val_ppl:.2f}")
        
        return metrics


def compare_metrics(okadfa: BenchmarkMetrics, baseline: BenchmarkMetrics):
    """Print comparison between OKADFA and baseline."""
    print(f"\n{'='*60}")
    print("BENCHMARK COMPARISON")
    print(f"{'='*60}")
    
    print(f"\n{'Metric':<30} {'OKADFA':>12} {'Baseline':>12} {'Speedup':>10}")
    print("-" * 66)
    
    # Speed metrics
    print(f"{'Samples/sec':<30} {okadfa.samples_per_sec:>12.1f} {baseline.samples_per_sec:>12.1f} {okadfa.samples_per_sec/baseline.samples_per_sec:>9.2f}x")
    print(f"{'Tokens/sec':<30} {okadfa.tokens_per_sec:>12.0f} {baseline.tokens_per_sec:>12.0f} {okadfa.tokens_per_sec/baseline.tokens_per_sec:>9.2f}x")
    print(f"{'Avg step time (s)':<30} {okadfa.avg_step_time:>12.3f} {baseline.avg_step_time:>12.3f} {baseline.avg_step_time/okadfa.avg_step_time:>9.2f}x")
    
    # Memory metrics
    print(f"{'Peak memory (MB)':<30} {okadfa.peak_memory_mb:>12.1f} {baseline.peak_memory_mb:>12.1f} {baseline.peak_memory_mb/okadfa.peak_memory_mb:>9.2f}x")
    
    # Quality metrics
    print(f"{'Final val PPL':<30} {okadfa.final_val_ppl:>12.2f} {baseline.final_val_ppl:>12.2f} {baseline.final_val_ppl/okadfa.final_val_ppl:>9.2f}x")
    print(f"{'PPL improvement':<30} {(okadfa.initial_val_ppl - okadfa.final_val_ppl):>12.2f} {(baseline.initial_val_ppl - baseline.final_val_ppl):>12.2f}")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    speedup = okadfa.samples_per_sec / baseline.samples_per_sec
    memory_ratio = baseline.peak_memory_mb / okadfa.peak_memory_mb
    ppl_ratio = okadfa.final_val_ppl / baseline.final_val_ppl
    
    print(f"OKADFA is {speedup:.2f}x {'FASTER' if speedup > 1 else 'SLOWER'} than baseline")
    print(f"OKADFA uses {memory_ratio:.2f}x {'LESS' if memory_ratio > 1 else 'MORE'} memory than baseline")
    print(f"OKADFA achieves {ppl_ratio:.2f}x {'BETTER' if ppl_ratio < 1 else 'WORSE'} perplexity than baseline")


def main():
    parser = argparse.ArgumentParser(description='Benchmark OKADFA vs Baseline')
    
    # Model configuration
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--num_random_features', type=int, default=128)
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_length', type=int, default=128)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--warmup_steps', type=int, default=100)
    
    # Dataset
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_val_samples', type=int, default=None)
    
    # System
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    
    # Output
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--results_dir', type=str, default='./benchmark_results')
    
    # Quick test
    parser.add_argument('--quick_test', action='store_true')
    
    args = parser.parse_args()
    
    # Quick test configuration
    if args.quick_test:
        print("Running quick benchmark test...")
        args.max_steps = 100
        args.max_train_samples = 500
        args.max_val_samples = 100
        args.device = 'cpu'
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Load tokenizer and data
    print("Loading tokenizer and dataset...")
    tokenizer = get_tokenizer('gpt2')
    train_loader, val_loader = create_wikitext_dataloaders(
        dataset_name='wikitext-2-raw-v1',
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    
    vocab_size = tokenizer.vocab_size
    
    # Create models
    print("\nCreating models...")
    
    # OKADFA model
    okadfa_model = OKADFAModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.seq_length,
        num_random_features=args.num_random_features,
    )
    
    # Baseline model
    baseline_model = StandardTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.seq_length,
    )
    
    # Benchmark OKADFA
    okadfa_benchmarker = ModelBenchmarker(
        model=okadfa_model,
        model_name="OKADFA",
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        use_dfa=True,
        use_ortho=True,
    )
    okadfa_metrics = okadfa_benchmarker.benchmark()
    
    # Clear memory
    del okadfa_model, okadfa_benchmarker
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    
    # Benchmark baseline
    baseline_benchmarker = ModelBenchmarker(
        model=baseline_model,
        model_name="Standard Transformer",
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        use_dfa=False,
        use_ortho=False,
    )
    baseline_metrics = baseline_benchmarker.benchmark()
    
    # Compare results
    compare_metrics(okadfa_metrics, baseline_metrics)
    
    # Save results
    if args.save_results:
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        okadfa_metrics.to_json(results_dir / 'okadfa_metrics.json')
        baseline_metrics.to_json(results_dir / 'baseline_metrics.json')
        
        print(f"\nResults saved to {results_dir}/")


if __name__ == '__main__':
    main()
