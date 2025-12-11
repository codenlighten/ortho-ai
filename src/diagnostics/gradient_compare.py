"""
Gradient Comparison and Diagnostic Tools for OKADFA

Copyright (c) 2025 Gregory Ward - SmartLedger.Technology
Licensed under the MIT License (see LICENSE file)

Provides comprehensive diagnostics for monitoring:
1. DFA vs BP gradient alignment (GradError, GradCosine)
2. Attention approximation quality (AttnSim)
3. Per-layer orthogonality violations
4. Training metrics and statistics
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict


@dataclass
class GradientMetrics:
    """Container for gradient comparison metrics."""
    
    # Gradient alignment metrics
    grad_error: float = 0.0  # ||∇_DFA - ∇_BP|| / ||∇_BP||
    grad_cosine: float = 0.0  # cos(∇_DFA, ∇_BP)
    dfa_norm: float = 0.0
    bp_norm: float = 0.0
    
    # Per-layer metrics
    layer_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Orthogonality metrics
    ortho_violations: Dict[str, float] = field(default_factory=dict)
    avg_ortho_violation: float = 0.0
    
    # Attention metrics
    attn_sim: float = 0.0  # Similarity between Favor+ and softmax attention
    
    # Statistics
    num_layers: int = 0
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            'grad_error': self.grad_error,
            'grad_cosine': self.grad_cosine,
            'dfa_norm': self.dfa_norm,
            'bp_norm': self.bp_norm,
            'avg_ortho_violation': self.avg_ortho_violation,
            'attn_sim': self.attn_sim,
            'num_layers': self.num_layers,
            **{f'layer_{k}': v for k, v in self.layer_metrics.items()},
            **{f'ortho_{k}': v for k, v in self.ortho_violations.items()},
        }


class GradientComparator:
    """
    Compares DFA and BP gradients for diagnostic purposes.
    
    Usage:
        comparator = GradientComparator(model)
        
        # During training:
        metrics = comparator.compare_gradients(
            dfa_grads=model.get_dfa_gradients(),
            bp_grads=model.get_bp_gradients()
        )
        
        print(f"Gradient Error: {metrics.grad_error:.4f}")
        print(f"Gradient Cosine: {metrics.grad_cosine:.4f}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        eps: float = 1e-8,
        track_per_layer: bool = True
    ):
        """
        Initialize gradient comparator.
        
        Args:
            model: PyTorch model to monitor
            eps: Small constant for numerical stability
            track_per_layer: Whether to compute per-layer metrics
        """
        self.model = model
        self.eps = eps
        self.track_per_layer = track_per_layer
        self.history: List[GradientMetrics] = []
        
    def compare_gradients(
        self,
        dfa_grads: Dict[str, torch.Tensor],
        bp_grads: Dict[str, torch.Tensor],
        ortho_violations: Optional[Dict[str, float]] = None
    ) -> GradientMetrics:
        """
        Compare DFA and BP gradients.
        
        Args:
            dfa_grads: Dictionary of DFA gradients {param_name: grad_tensor}
            bp_grads: Dictionary of BP gradients {param_name: grad_tensor}
            ortho_violations: Optional orthogonality violation metrics
            
        Returns:
            GradientMetrics with comparison results
        """
        metrics = GradientMetrics()
        
        # Ensure same parameters
        common_params = set(dfa_grads.keys()) & set(bp_grads.keys())
        if not common_params:
            return metrics
        
        # Flatten all gradients
        dfa_flat = torch.cat([dfa_grads[k].flatten() for k in sorted(common_params)])
        bp_flat = torch.cat([bp_grads[k].flatten() for k in sorted(common_params)])
        
        # Compute global metrics
        metrics.dfa_norm = torch.norm(dfa_flat).item()
        metrics.bp_norm = torch.norm(bp_flat).item()
        
        # Gradient error: ||∇_DFA - ∇_BP|| / ||∇_BP||
        diff = dfa_flat - bp_flat
        diff_norm = torch.norm(diff).item()
        metrics.grad_error = diff_norm / (metrics.bp_norm + self.eps)
        
        # Gradient cosine similarity: cos(∇_DFA, ∇_BP)
        dot_product = torch.dot(dfa_flat, bp_flat).item()
        metrics.grad_cosine = dot_product / (metrics.dfa_norm * metrics.bp_norm + self.eps)
        
        # Per-layer metrics
        if self.track_per_layer:
            for param_name in sorted(common_params):
                dfa_param = dfa_grads[param_name]
                bp_param = bp_grads[param_name]
                
                dfa_norm_layer = torch.norm(dfa_param).item()
                bp_norm_layer = torch.norm(bp_param).item()
                diff_layer = torch.norm(dfa_param - bp_param).item()
                
                layer_error = diff_layer / (bp_norm_layer + self.eps)
                
                # Cosine similarity
                dfa_flat_layer = dfa_param.flatten()
                bp_flat_layer = bp_param.flatten()
                dot_layer = torch.dot(dfa_flat_layer, bp_flat_layer).item()
                layer_cosine = dot_layer / (dfa_norm_layer * bp_norm_layer + self.eps)
                
                metrics.layer_metrics[param_name] = {
                    'error': layer_error,
                    'cosine': layer_cosine,
                    'dfa_norm': dfa_norm_layer,
                    'bp_norm': bp_norm_layer,
                }
        
        # Add orthogonality violations
        if ortho_violations:
            metrics.ortho_violations = ortho_violations
            metrics.avg_ortho_violation = np.mean(list(ortho_violations.values()))
        
        metrics.num_layers = len(common_params)
        metrics.timestamp = 0.0  # Placeholder for timestamp
        
        self.history.append(metrics)
        return metrics
    
    def get_statistics(self, window: int = 100) -> Dict[str, float]:
        """
        Get summary statistics over recent history.
        
        Args:
            window: Number of recent metrics to average
            
        Returns:
            Dictionary of averaged metrics
        """
        if not self.history:
            return {}
        
        recent = self.history[-window:]
        
        return {
            'avg_grad_error': np.mean([m.grad_error for m in recent]),
            'std_grad_error': np.std([m.grad_error for m in recent]),
            'avg_grad_cosine': np.mean([m.grad_cosine for m in recent]),
            'std_grad_cosine': np.std([m.grad_cosine for m in recent]),
            'avg_ortho_violation': np.mean([m.avg_ortho_violation for m in recent]),
            'std_ortho_violation': np.std([m.avg_ortho_violation for m in recent]),
            'num_samples': len(recent),
        }
    
    def reset_history(self):
        """Clear metric history."""
        self.history.clear()


class AttentionComparator:
    """
    Compare Favor+ kernelized attention with standard softmax attention.
    
    Computes attention similarity metrics to validate approximation quality.
    """
    
    def __init__(self, eps: float = 1e-8):
        """
        Initialize attention comparator.
        
        Args:
            eps: Small constant for numerical stability
        """
        self.eps = eps
        self.history: List[float] = []
    
    def compute_softmax_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute standard softmax attention.
        
        Args:
            Q: Queries [batch, heads, seq_len, d_k]
            K: Keys [batch, heads, seq_len, d_k]
            V: Values [batch, heads, seq_len, d_v]
            mask: Optional attention mask
            
        Returns:
            Attention output [batch, heads, seq_len, d_v]
        """
        d_k = Q.size(-1)
        
        # Attention scores: QK^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply to values
        output = torch.matmul(attn_weights, V)
        
        return output
    
    def compute_similarity(
        self,
        favor_output: torch.Tensor,
        softmax_output: torch.Tensor
    ) -> float:
        """
        Compute cosine similarity between Favor+ and softmax attention outputs.
        
        Args:
            favor_output: Favor+ attention output
            softmax_output: Softmax attention output
            
        Returns:
            Cosine similarity score [0, 1]
        """
        # Flatten outputs
        favor_flat = favor_output.flatten()
        softmax_flat = softmax_output.flatten()
        
        # Compute cosine similarity
        dot_product = torch.dot(favor_flat, softmax_flat).item()
        favor_norm = torch.norm(favor_flat).item()
        softmax_norm = torch.norm(softmax_flat).item()
        
        similarity = dot_product / (favor_norm * softmax_norm + self.eps)
        
        self.history.append(similarity)
        return similarity
    
    def compare_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        favor_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compare Favor+ and softmax attention.
        
        Args:
            Q: Queries
            K: Keys
            V: Values
            favor_output: Favor+ attention output
            mask: Optional attention mask
            
        Returns:
            Dictionary with comparison metrics
        """
        # Compute softmax attention
        softmax_output = self.compute_softmax_attention(Q, K, V, mask)
        
        # Compute similarity
        similarity = self.compute_similarity(favor_output, softmax_output)
        
        # Compute relative error
        diff = torch.norm(favor_output - softmax_output).item()
        softmax_norm = torch.norm(softmax_output).item()
        relative_error = diff / (softmax_norm + self.eps)
        
        return {
            'attn_sim': similarity,
            'attn_error': relative_error,
            'favor_norm': torch.norm(favor_output).item(),
            'softmax_norm': softmax_norm,
        }
    
    def get_avg_similarity(self, window: int = 100) -> float:
        """Get average similarity over recent history."""
        if not self.history:
            return 0.0
        recent = self.history[-window:]
        return np.mean(recent)


class DiagnosticLogger:
    """
    Unified logging interface for OKADFA diagnostics.
    
    Supports multiple backends: console, file, WandB, TensorBoard.
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        console_log: bool = True
    ):
        """
        Initialize diagnostic logger.
        
        Args:
            log_dir: Directory for log files
            use_wandb: Enable Weights & Biases logging
            use_tensorboard: Enable TensorBoard logging
            console_log: Enable console output
        """
        self.log_dir = log_dir
        self.console_log = console_log
        
        # WandB
        self.use_wandb = use_wandb
        self.wandb = None
        if use_wandb:
            try:
                import wandb as wandb_module
                self.wandb = wandb_module
            except ImportError:
                print("Warning: wandb not installed, disabling WandB logging")
                self.use_wandb = False
        
        # TensorBoard
        self.use_tensorboard = use_tensorboard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("Warning: tensorboard not installed, disabling TensorBoard logging")
                self.use_tensorboard = False
        
        self.step = 0
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to all enabled backends.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Training step (uses internal counter if None)
        """
        if step is None:
            step = self.step
            self.step += 1
        
        # Console
        if self.console_log:
            print(f"\n[Step {step}] Metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
        
        # WandB
        if self.use_wandb and self.wandb is not None:
            self.wandb.log(metrics, step=step)
        
        # TensorBoard
        if self.use_tensorboard and self.tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)
    
    def log_gradient_metrics(self, metrics: GradientMetrics, step: Optional[int] = None):
        """Log GradientMetrics object."""
        self.log_metrics(metrics.to_dict(), step)
    
    def close(self):
        """Close logger and cleanup resources."""
        if self.tb_writer is not None:
            self.tb_writer.close()


# Convenience function for quick diagnostics
def quick_diagnostic(
    model: nn.Module,
    dfa_grads: Dict[str, torch.Tensor],
    bp_grads: Dict[str, torch.Tensor],
    ortho_violations: Optional[Dict[str, float]] = None
) -> None:
    """
    Quick diagnostic printout for debugging.
    
    Args:
        model: Model being trained
        dfa_grads: DFA gradients
        bp_grads: BP gradients
        ortho_violations: Orthogonality violations
    """
    comparator = GradientComparator(model)
    metrics = comparator.compare_gradients(dfa_grads, bp_grads, ortho_violations)
    
    print("\n" + "="*60)
    print("OKADFA DIAGNOSTIC REPORT")
    print("="*60)
    print(f"Gradient Error:  {metrics.grad_error:.4f}")
    print(f"Gradient Cosine: {metrics.grad_cosine:.4f}")
    print(f"DFA Norm:        {metrics.dfa_norm:.4f}")
    print(f"BP Norm:         {metrics.bp_norm:.4f}")
    
    if ortho_violations:
        print(f"\nAvg Ortho Violation: {metrics.avg_ortho_violation:.6f}")
        print("Per-layer violations:")
        for layer, violation in sorted(ortho_violations.items()):
            print(f"  {layer}: {violation:.6f}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    print("Testing diagnostic tools...")
    
    # Test GradientComparator
    print("\n1. Testing GradientComparator:")
    
    # Create dummy gradients
    dfa_grads = {
        'layer1.weight': torch.randn(128, 256),
        'layer2.weight': torch.randn(256, 512),
    }
    bp_grads = {
        'layer1.weight': dfa_grads['layer1.weight'] + 0.1 * torch.randn(128, 256),
        'layer2.weight': dfa_grads['layer2.weight'] + 0.1 * torch.randn(256, 512),
    }
    
    comparator = GradientComparator(None)
    metrics = comparator.compare_gradients(dfa_grads, bp_grads)
    
    print(f"   Gradient Error: {metrics.grad_error:.4f}")
    print(f"   Gradient Cosine: {metrics.grad_cosine:.4f}")
    print(f"   Tracked {metrics.num_layers} layers")
    
    # Test AttentionComparator
    print("\n2. Testing AttentionComparator:")
    
    batch, heads, seq_len, d_k = 2, 4, 16, 32
    Q = torch.randn(batch, heads, seq_len, d_k)
    K = torch.randn(batch, heads, seq_len, d_k)
    V = torch.randn(batch, heads, seq_len, d_k)
    
    attn_comp = AttentionComparator()
    softmax_out = attn_comp.compute_softmax_attention(Q, K, V)
    favor_out = softmax_out + 0.05 * torch.randn_like(softmax_out)  # Simulate Favor+
    
    comparison = attn_comp.compare_attention(Q, K, V, favor_out)
    print(f"   Attention Similarity: {comparison['attn_sim']:.4f}")
    print(f"   Attention Error: {comparison['attn_error']:.4f}")
    
    # Test quick_diagnostic
    print("\n3. Testing quick_diagnostic:")
    ortho_violations = {
        'attn.q_proj': 0.0012,
        'attn.k_proj': 0.0008,
        'attn.v_proj': 0.0015,
    }
    quick_diagnostic(None, dfa_grads, bp_grads, ortho_violations)
    
    print("✅ All diagnostic tools working correctly!")
