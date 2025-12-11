"""
Kernelized Orthogonal Attention (KOA) with Multi-Head Architecture

Combines:
1. Favor+ kernelized attention (linear complexity)
2. Orthogonal weight matrices (for DFA compatibility)
3. Multi-head attention mechanism
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import math

try:
    from ..kernels.favor_plus import FavorPlusAttention
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from kernels.favor_plus import FavorPlusAttention


class KOAMultiHeadAttention(nn.Module):
    """
    Kernelized Orthogonal Attention with multiple heads.
    
    Uses Favor+ for linear complexity O(TMd_k) and enforces orthogonality
    on projection matrices for DFA compatibility.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        num_features: Number of random features for Favor+ (default: 2 * d_k)
        dropout: Dropout probability
        causal: Whether to use causal masking
        orthogonal_init: Initialize weights with orthogonal matrices
        track_orthogonality: Track per-head orthogonality violations
        device: Device to place tensors on
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_features: Optional[int] = None,
        dropout: float = 0.1,
        causal: bool = True,
        orthogonal_init: bool = True,
        track_orthogonality: bool = True,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal
        self.track_orthogonality = track_orthogonality
        self.device = device or torch.device("cpu")
        
        # Default num_features to 2 * d_k (as per Performer paper)
        self.num_features = num_features or (2 * self.d_k)
        
        # Q, K, V projections (separate for each head to enable orthogonality tracking)
        # Standard approach: single projection then split heads
        # Our approach: per-head projections for finer orthogonality control
        self.q_proj = nn.Linear(d_model, d_model, device=device)
        self.k_proj = nn.Linear(d_model, d_model, device=device)
        self.v_proj = nn.Linear(d_model, d_model, device=device)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, device=device)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Favor+ attention for each head (uses d_k as the model dimension)
        self.favor_attention = FavorPlusAttention(
            d_model=self.d_k,  # Each head operates on d_k dimensions
            num_features=self.num_features,
            causal=causal,
            device=device
        )
        
        # Initialize weights
        if orthogonal_init:
            self._init_orthogonal_weights()
        else:
            self._init_standard_weights()
        
        # Statistics tracking
        self._last_attn_weights: Optional[torch.Tensor] = None
        self._ortho_violations: List[float] = []
    
    def _init_orthogonal_weights(self):
        """Initialize projection matrices with orthogonal weights."""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            # Use orthogonal initialization
            nn.init.orthogonal_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
    
    def _init_standard_weights(self):
        """Initialize with standard Xavier/Glorot initialization."""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
    
    def get_projection_matrices(self) -> List[Tuple[str, torch.Tensor]]:
        """
        Get projection weight matrices for orthogonality loss computation.
        
        Returns:
            List of (name, weight) tuples
        """
        return [
            ("Q", self.q_proj.weight),
            ("K", self.k_proj.weight),
            ("V", self.v_proj.weight),
            ("out", self.out_proj.weight)
        ]
    
    def compute_orthogonality_violation(self, weight: torch.Tensor) -> float:
        """
        Compute ||W^T W - I||²_F for a weight matrix.
        
        Args:
            weight: Weight matrix W ∈ R^{d_out × d_in}
        
        Returns:
            Orthogonality violation (scalar)
        """
        gram = torch.matmul(weight.T, weight)
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        violation = torch.norm(gram - identity, p="fro") ** 2
        return violation.item()
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through multi-head KOA.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional mask [batch, seq_len] (1 = attend, 0 = mask)
            return_attention_weights: Whether to return attention weights
        
        Returns:
            output: Output tensor [batch, seq_len, d_model]
            attention_weights: Optional attention weights if requested
        """
        batch_size, seq_len, d_model = x.shape
        
        assert d_model == self.d_model, \
            f"Input d_model ({d_model}) doesn't match expected ({self.d_model})"
        
        # Project Q, K, V
        Q = self.q_proj(x)  # [batch, seq_len, d_model]
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head: [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply Favor+ attention to each head
        # Favor+ expects [batch, seq_len, d_k], so we process heads sequentially
        head_outputs = []
        attention_weights_list = []
        
        for head_idx in range(self.num_heads):
            Q_head = Q[:, head_idx, :, :]  # [batch, seq_len, d_k]
            K_head = K[:, head_idx, :, :]
            V_head = V[:, head_idx, :, :]
            
            # Apply Favor+ attention (returns tuple: (output, attn_weights))
            out_head, _ = self.favor_attention(Q_head, K_head, V_head)  # [batch, seq_len, d_k]
            head_outputs.append(out_head)
            
            # Track attention weights if requested (approximation)
            if return_attention_weights and self.track_orthogonality:
                # Compute approximate attention weights from kernel features
                # φ(Q) @ φ(K)^T ≈ softmax(Q K^T)
                with torch.no_grad():
                    attn_approx = torch.matmul(Q_head, K_head.transpose(-2, -1)) / math.sqrt(self.d_k)
                    if self.causal:
                        # Apply causal mask
                        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
                        attn_approx = attn_approx.masked_fill(mask, float('-inf'))
                    attn_weights = torch.softmax(attn_approx, dim=-1)
                    attention_weights_list.append(attn_weights)
        
        # Concatenate heads: [batch, seq_len, d_model]
        multi_head_output = torch.cat(head_outputs, dim=-1)
        
        # Output projection
        output = self.out_proj(multi_head_output)
        
        # Dropout
        output = self.dropout(output)
        
        # Track orthogonality violations if enabled
        if self.track_orthogonality:
            self._ortho_violations.clear()
            for name, weight in self.get_projection_matrices():
                violation = self.compute_orthogonality_violation(weight)
                self._ortho_violations.append(violation)
        
        # Aggregate attention weights if requested
        attn_weights_out = None
        if return_attention_weights and attention_weights_list:
            attn_weights_out = torch.stack(attention_weights_list, dim=1)  # [batch, num_heads, seq_len, seq_len]
        
        return output, attn_weights_out
    
    def get_orthogonality_statistics(self) -> dict:
        """Get orthogonality statistics for all projection matrices."""
        stats = {}
        
        for i, (name, weight) in enumerate(self.get_projection_matrices()):
            violation = self.compute_orthogonality_violation(weight)
            stats[f"ortho_violation_{name}"] = violation
            
            # Compute condition number (stability indicator)
            with torch.no_grad():
                try:
                    singular_values = torch.linalg.svdvals(weight)
                    cond = (singular_values.max() / singular_values.min()).item()
                    stats[f"condition_number_{name}"] = cond
                except:
                    stats[f"condition_number_{name}"] = float('inf')
        
        if self._ortho_violations:
            stats["total_violation"] = sum(self._ortho_violations)
            stats["mean_violation"] = sum(self._ortho_violations) / len(self._ortho_violations)
        
        return stats
    
    def redraw_projection_matrix(self):
        """Redraw random projection matrix for Favor+ (for training stability)."""
        self.favor_attention.feature_map.redraw_projection_matrix()
    
    def __repr__(self) -> str:
        return (
            f"KOAMultiHeadAttention("
            f"d_model={self.d_model}, "
            f"num_heads={self.num_heads}, "
            f"d_k={self.d_k}, "
            f"num_features={self.num_features}, "
            f"causal={self.causal}"
            f")"
        )


# Testing code
if __name__ == "__main__":
    print("Testing KOAMultiHeadAttention...")
    
    # Test 1: Basic forward pass
    print("\n=== Test 1: Basic Forward Pass ===")
    
    d_model = 512
    num_heads = 8
    batch_size = 4
    seq_len = 64
    
    koa = KOAMultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        causal=True,
        orthogonal_init=True
    )
    
    print(f"Model: {koa}")
    print(f"Parameters: {sum(p.numel() for p in koa.parameters()):,}")
    
    # Forward pass
    x = torch.randn(batch_size, seq_len, d_model)
    output, attn_weights = koa(x, return_attention_weights=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    if attn_weights is not None:
        print(f"Attention weights shape: {attn_weights.shape}")
    
    assert output.shape == x.shape, "Output shape mismatch"
    
    # Test 2: Orthogonality statistics
    print("\n=== Test 2: Orthogonality Statistics ===")
    
    stats = koa.get_orthogonality_statistics()
    print(f"Orthogonality statistics:")
    for key, val in sorted(stats.items()):
        if 'violation' in key:
            print(f"  {key}: {val:.6f}")
        elif 'condition' in key:
            print(f"  {key}: {val:.2f}")
    
    # Test 3: Gradient flow
    print("\n=== Test 3: Gradient Flow ===")
    
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    output, _ = koa(x)
    
    loss = output.sum()
    loss.backward()
    
    print(f"Input gradient: {x.grad.norm().item():.6f}")
    print(f"Q projection gradient: {koa.q_proj.weight.grad.norm().item():.6f}")
    print(f"K projection gradient: {koa.k_proj.weight.grad.norm().item():.6f}")
    print(f"V projection gradient: {koa.v_proj.weight.grad.norm().item():.6f}")
    print(f"Out projection gradient: {koa.out_proj.weight.grad.norm().item():.6f}")
    
    # Test 4: Non-causal attention
    print("\n=== Test 4: Non-Causal Attention ===")
    
    koa_non_causal = KOAMultiHeadAttention(
        d_model=256,
        num_heads=4,
        causal=False,
        orthogonal_init=False  # Test standard init
    )
    
    x = torch.randn(2, 32, 256)
    output, _ = koa_non_causal(x)
    
    print(f"Non-causal output shape: {output.shape}")
    assert output.shape == x.shape
    
    # Test 5: Different sequence lengths
    print("\n=== Test 5: Various Sequence Lengths ===")
    
    koa_test = KOAMultiHeadAttention(d_model=128, num_heads=4)
    
    for length in [16, 64, 256]:
        x = torch.randn(2, length, 128)
        output, _ = koa_test(x)
        print(f"  Seq length {length}: output shape {output.shape}")
        assert output.shape == x.shape
    
    # Test 6: Projection matrix redraw
    print("\n=== Test 6: Projection Matrix Redraw ===")
    
    x = torch.randn(2, 32, 256)
    output1, _ = koa_non_causal(x)
    
    # Redraw projection matrix
    koa_non_causal.redraw_projection_matrix()
    
    output2, _ = koa_non_causal(x)
    
    diff = (output1 - output2).norm() / output1.norm()
    print(f"Output difference after redraw: {diff.item():.4f}")
    print(f"(Should be non-zero, indicating projection changed)")
    
    # Test 7: Batch size variations
    print("\n=== Test 7: Various Batch Sizes ===")
    
    for batch in [1, 4, 16]:
        x = torch.randn(batch, 32, 128)
        output, _ = koa_test(x)
        print(f"  Batch size {batch}: output shape {output.shape}")
        assert output.shape == x.shape
    
    print("\n✅ All tests completed!")
