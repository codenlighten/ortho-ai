"""
Orthogonality Loss for enforcing orthogonal weight matrices.

Copyright (c) 2025 Gregory Ward - SmartLedger.Technology
Licensed under the MIT License (see LICENSE file)

Computes: L_ortho = λ(t) * Σ_l ||W_l^T W_l - I||²_F

with warmup scheduler: λ(t) = λ_max * min(1, t / t_warmup)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional


class OrthogonalityLoss(nn.Module):
    """
    Orthogonality regularization loss with warmup scheduler.
    
    Enforces that weight matrices W_l ∈ R^{d_out × d_in} satisfy W_l^T W_l ≈ I,
    ensuring that columns of W_l are approximately orthonormal.
    
    Args:
        lambda_max: Maximum regularization coefficient
        warmup_steps: Number of steps to linearly warmup lambda
        reduction: How to reduce per-layer losses ('mean', 'sum', or 'none')
        track_per_head: Whether to track orthogonality violation per attention head
    """
    
    def __init__(
        self,
        lambda_max: float = 0.01,
        warmup_steps: int = 1000,
        reduction: str = "mean",
        track_per_head: bool = True
    ):
        super().__init__()
        
        self.lambda_max = lambda_max
        self.warmup_steps = warmup_steps
        self.reduction = reduction
        self.track_per_head = track_per_head
        
        # Register step counter as buffer (for checkpointing)
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))
        
        # Cache for statistics
        self._last_violations: Dict[str, float] = {}
        self._last_lambda: float = 0.0
    
    def get_lambda(self, step: Optional[int] = None) -> float:
        """
        Get current lambda value with linear warmup.
        
        λ(t) = λ_max * min(1, t / t_warmup)
        
        Args:
            step: Current training step (if None, uses internal counter)
        
        Returns:
            Current lambda value
        """
        if step is None:
            step = self.step_count.item()
        
        warmup_factor = min(1.0, step / max(1, self.warmup_steps))
        return self.lambda_max * warmup_factor
    
    def compute_violation(
        self, 
        weight: torch.Tensor,
        name: str = "weight"
    ) -> torch.Tensor:
        """
        Compute orthogonality violation: ||W^T W - I||²_F
        
        Args:
            weight: Weight matrix W ∈ R^{d_out × d_in}
            name: Name for tracking (e.g., "layer_0_query")
        
        Returns:
            Scalar violation metric
        """
        # W^T W ∈ R^{d_in × d_in}
        gram = torch.matmul(weight.T, weight)
        
        # Identity matrix
        identity = torch.eye(
            gram.shape[0], 
            dtype=gram.dtype, 
            device=gram.device
        )
        
        # ||W^T W - I||²_F
        violation = torch.norm(gram - identity, p="fro") ** 2
        
        # Track for diagnostics
        self._last_violations[name] = violation.item()
        
        return violation
    
    def forward(
        self, 
        weights: List[torch.Tensor],
        weight_names: Optional[List[str]] = None,
        step: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute weighted orthogonality loss across multiple weight matrices.
        
        L_ortho = λ(t) * Σ_l ||W_l^T W_l - I||²_F
        
        Args:
            weights: List of weight matrices to regularize
            weight_names: Optional names for tracking (e.g., ["layer_0_Q", "layer_0_K"])
            step: Current training step (if None, uses internal counter)
        
        Returns:
            Orthogonality loss scalar
        """
        if len(weights) == 0:
            return torch.tensor(0.0, device=self.step_count.device)
        
        # Get current lambda
        current_lambda = self.get_lambda(step)
        self._last_lambda = current_lambda
        
        # Compute violations for each weight matrix
        violations = []
        for i, weight in enumerate(weights):
            name = weight_names[i] if weight_names else f"weight_{i}"
            violation = self.compute_violation(weight, name=name)
            violations.append(violation)
        
        # Stack and reduce
        violations_tensor = torch.stack(violations)
        
        if self.reduction == "mean":
            total_violation = violations_tensor.mean()
        elif self.reduction == "sum":
            total_violation = violations_tensor.sum()
        elif self.reduction == "none":
            total_violation = violations_tensor
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
        
        # Apply lambda scaling
        loss = current_lambda * total_violation
        
        # Increment step counter
        self.step_count += 1
        
        return loss
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get current statistics for logging.
        
        Returns:
            Dictionary with lambda, total violation, and per-weight violations
        """
        stats = {
            "lambda": self._last_lambda,
            "total_violation": sum(self._last_violations.values()),
            "num_weights": len(self._last_violations),
            "step": self.step_count.item()
        }
        
        # Add per-weight violations
        for name, violation in self._last_violations.items():
            stats[f"violation_{name}"] = violation
        
        return stats
    
    def reset_step(self, step: int = 0):
        """Reset step counter (e.g., when resuming from checkpoint)."""
        self.step_count.fill_(step)
    
    def __repr__(self) -> str:
        return (
            f"OrthogonalityLoss("
            f"lambda_max={self.lambda_max}, "
            f"warmup_steps={self.warmup_steps}, "
            f"reduction={self.reduction}, "
            f"current_step={self.step_count.item()}"
            f")"
        )


class PerHeadOrthogonalityLoss(OrthogonalityLoss):
    """
    Orthogonality loss with per-head tracking for multi-head attention.
    
    Automatically handles splitting weight matrices by attention head
    and tracks violations separately for each head.
    
    Args:
        num_heads: Number of attention heads
        lambda_max: Maximum regularization coefficient
        warmup_steps: Number of steps to linearly warmup lambda
        reduction: How to reduce per-head losses ('mean', 'sum', or 'none')
    """
    
    def __init__(
        self,
        num_heads: int,
        lambda_max: float = 0.01,
        warmup_steps: int = 1000,
        reduction: str = "mean"
    ):
        super().__init__(
            lambda_max=lambda_max,
            warmup_steps=warmup_steps,
            reduction=reduction,
            track_per_head=True
        )
        self.num_heads = num_heads
        self._per_head_violations: Dict[int, List[float]] = {
            i: [] for i in range(num_heads)
        }
    
    def compute_per_head_violations(
        self,
        weight: torch.Tensor,
        layer_name: str = "layer"
    ) -> torch.Tensor:
        """
        Compute orthogonality violation for each attention head separately.
        
        Args:
            weight: Weight matrix W ∈ R^{d_out × d_in} where d_out = num_heads * d_head
            layer_name: Name of the layer for tracking
        
        Returns:
            Tensor of violations per head [num_heads]
        """
        d_out, d_in = weight.shape
        
        # Ensure d_out is divisible by num_heads
        assert d_out % self.num_heads == 0, \
            f"d_out={d_out} must be divisible by num_heads={self.num_heads}"
        
        d_head = d_out // self.num_heads
        
        # Reshape to [num_heads, d_head, d_in]
        weight_per_head = weight.reshape(self.num_heads, d_head, d_in)
        
        violations = []
        for head_idx in range(self.num_heads):
            W_head = weight_per_head[head_idx]  # [d_head, d_in]
            
            # Compute violation for this head
            violation = self.compute_violation(
                W_head,
                name=f"{layer_name}_head_{head_idx}"
            )
            violations.append(violation)
            
            # Track per-head history
            self._per_head_violations[head_idx].append(violation.item())
        
        return torch.stack(violations)
    
    def forward(
        self,
        weights: List[torch.Tensor],
        weight_names: Optional[List[str]] = None,
        step: Optional[int] = None,
        per_head: bool = True
    ) -> torch.Tensor:
        """
        Compute orthogonality loss with optional per-head tracking.
        
        Args:
            weights: List of weight matrices
            weight_names: Optional names for tracking
            step: Current training step
            per_head: If True, compute per-head violations
        
        Returns:
            Orthogonality loss scalar
        """
        if len(weights) == 0:
            return torch.tensor(0.0, device=self.step_count.device)
        
        current_lambda = self.get_lambda(step)
        self._last_lambda = current_lambda
        
        all_violations = []
        
        for i, weight in enumerate(weights):
            name = weight_names[i] if weight_names else f"weight_{i}"
            
            if per_head:
                # Compute per-head violations
                head_violations = self.compute_per_head_violations(weight, name)
                all_violations.extend(head_violations)
            else:
                # Compute single violation
                violation = self.compute_violation(weight, name)
                all_violations.append(violation)
        
        # Stack and reduce
        violations_tensor = torch.stack(all_violations)
        
        if self.reduction == "mean":
            total_violation = violations_tensor.mean()
        elif self.reduction == "sum":
            total_violation = violations_tensor.sum()
        else:
            total_violation = violations_tensor
        
        loss = current_lambda * total_violation
        
        self.step_count += 1
        
        return loss
    
    def get_per_head_statistics(self) -> Dict[str, float]:
        """Get statistics per attention head."""
        stats = self.get_statistics()
        
        # Add per-head averages
        for head_idx in range(self.num_heads):
            violations = self._per_head_violations[head_idx]
            if violations:
                avg_violation = sum(violations) / len(violations)
                stats[f"head_{head_idx}_avg_violation"] = avg_violation
        
        return stats


# Testing code
if __name__ == "__main__":
    print("Testing OrthogonalityLoss...")
    
    # Test 1: Basic functionality
    print("\n=== Test 1: Basic Loss Computation ===")
    loss_fn = OrthogonalityLoss(lambda_max=0.01, warmup_steps=100)
    
    # Create some weight matrices
    W1 = torch.randn(256, 256)  # Random (non-orthogonal)
    W2, _ = torch.linalg.qr(torch.randn(256, 256))  # Orthogonal
    
    loss = loss_fn([W1, W2], weight_names=["W1", "W2"], step=0)
    print(f"Loss at step 0 (warmup): {loss.item():.6f}")
    
    stats = loss_fn.get_statistics()
    print(f"Statistics: {stats}")
    print(f"  - W1 violation: {stats['violation_W1']:.6f}")
    print(f"  - W2 violation: {stats['violation_W2']:.6f}")
    
    # Test 2: Warmup scheduler
    print("\n=== Test 2: Warmup Scheduler ===")
    loss_fn = OrthogonalityLoss(lambda_max=0.1, warmup_steps=100)
    
    W = torch.randn(128, 128)
    
    for step in [0, 25, 50, 100, 200]:
        loss = loss_fn([W], step=step)
        lambda_val = loss_fn.get_lambda(step)
        print(f"Step {step:3d}: λ={lambda_val:.6f}, loss={loss.item():.6f}")
    
    # Test 3: Per-head tracking
    print("\n=== Test 3: Per-Head Tracking ===")
    num_heads = 8
    loss_fn = PerHeadOrthogonalityLoss(
        num_heads=num_heads,
        lambda_max=0.01,
        warmup_steps=100
    )
    
    # Multi-head weight: [num_heads * d_head, d_in] = [512, 256]
    W_multihead = torch.randn(512, 256)
    
    loss = loss_fn([W_multihead], weight_names=["mh_query"], step=50)
    print(f"Loss with per-head tracking: {loss.item():.6f}")
    
    stats = loss_fn.get_per_head_statistics()
    print(f"Per-head violations:")
    for head_idx in range(num_heads):
        key = f"violation_mh_query_head_{head_idx}"
        if key in stats:
            print(f"  Head {head_idx}: {stats[key]:.6f}")
    
    # Test 4: Orthogonal vs Non-orthogonal
    print("\n=== Test 4: Orthogonal vs Non-orthogonal ===")
    loss_fn = OrthogonalityLoss(lambda_max=0.1, warmup_steps=1)
    
    # Perfect orthogonal matrix
    W_ortho, _ = torch.linalg.qr(torch.randn(128, 128))
    loss_ortho = loss_fn([W_ortho], step=100)
    
    # Random matrix
    W_random = torch.randn(128, 128)
    loss_random = loss_fn([W_random], step=100)
    
    print(f"Orthogonal matrix loss: {loss_ortho.item():.6f}")
    print(f"Random matrix loss: {loss_random.item():.6f}")
    print(f"Ratio: {loss_random.item() / max(loss_ortho.item(), 1e-8):.2f}x")
    
    print("\n✅ All tests completed!")
