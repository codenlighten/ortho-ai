"""
DFA Transformer Block

Copyright (c) 2025 Gregory Ward - SmartLedger.Technology
Licensed under the MIT License (see LICENSE file)

Combines KOA attention with feedforward network and applies hybrid DFA/BP training.
Uses DFA between blocks, standard BP within blocks (per expert review).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import math

try:
    from .koa_attention import KOAMultiHeadAttention
    from ..training.dfa_backward import HybridDFAHook
    from ..training.dfa_feedback import DFAFeedbackMatrix
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.koa_attention import KOAMultiHeadAttention
    from training.dfa_backward import HybridDFAHook
    from training.dfa_feedback import DFAFeedbackMatrix


class DFATransformerBlock(nn.Module):
    """
    Transformer block with KOA attention and Direct Feedback Alignment.
    
    Architecture:
    - Pre-LayerNorm (more stable than post-norm)
    - KOA multi-head attention
    - Feedforward network (2-layer MLP)
    - Residual connections
    - Hybrid DFA/BP: DFA between blocks, BP within blocks
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feedforward dimension (default: 4 * d_model)
        dropout: Dropout probability
        causal: Whether to use causal attention
        orthogonal_init: Initialize with orthogonal weights
        activation: Activation function ('relu', 'gelu', 'swish')
        device: Device to place tensors on
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        causal: bool = True,
        orthogonal_init: bool = True,
        activation: str = "gelu",
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff or (4 * d_model)
        self.device = device or torch.device("cpu")
        
        # Layer normalization (pre-norm)
        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.norm2 = nn.LayerNorm(d_model, device=device)
        
        # KOA multi-head attention
        self.attention = KOAMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            causal=causal,
            orthogonal_init=orthogonal_init,
            device=device
        )
        
        # Feedforward network
        self.ff = FeedForward(
            d_model=d_model,
            d_ff=self.d_ff,
            dropout=dropout,
            activation=activation,
            orthogonal_init=orthogonal_init,
            device=device
        )
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask
            return_attention_weights: Whether to return attention weights
        
        Returns:
            output: Output tensor [batch, seq_len, d_model]
            attention_weights: Optional attention weights
        """
        # Attention block with pre-norm and residual
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attention(
            x_norm, 
            attention_mask=attention_mask,
            return_attention_weights=return_attention_weights
        )
        x = residual + self.dropout(attn_out)
        
        # Feedforward block with pre-norm and residual
        residual = x
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = residual + self.dropout(ff_out)
        
        return x, attn_weights
    
    def get_attention_modules(self) -> list:
        """Get attention projection modules for DFA hook registration."""
        return [
            self.attention.q_proj,
            self.attention.k_proj,
            self.attention.v_proj,
            self.attention.out_proj
        ]
    
    def get_feedforward_modules(self) -> list:
        """Get feedforward modules for DFA hook registration."""
        return [self.ff.fc1, self.ff.fc2]
    
    def get_all_linear_modules(self) -> list:
        """Get all linear modules for DFA hook registration."""
        return self.get_attention_modules() + self.get_feedforward_modules()
    
    def get_orthogonality_weights(self) -> list:
        """Get weight matrices for orthogonality loss computation."""
        weights = []
        
        # Attention projections
        for name, module in [
            ("attn_Q", self.attention.q_proj),
            ("attn_K", self.attention.k_proj),
            ("attn_V", self.attention.v_proj),
            ("attn_out", self.attention.out_proj)
        ]:
            weights.append((name, module.weight))
        
        # Feedforward projections
        weights.append(("ff_fc1", self.ff.fc1.weight))
        weights.append(("ff_fc2", self.ff.fc2.weight))
        
        return weights
    
    def get_statistics(self) -> Dict[str, float]:
        """Get comprehensive statistics for diagnostics."""
        stats = {}
        
        # Attention statistics
        attn_stats = self.attention.get_orthogonality_statistics()
        for key, val in attn_stats.items():
            stats[f"attn_{key}"] = val
        
        # Feedforward statistics
        ff_stats = self.ff.get_statistics()
        for key, val in ff_stats.items():
            stats[f"ff_{key}"] = val
        
        return stats
    
    def __repr__(self) -> str:
        return (
            f"DFATransformerBlock("
            f"d_model={self.d_model}, "
            f"num_heads={self.num_heads}, "
            f"d_ff={self.d_ff}"
            f")"
        )


class FeedForward(nn.Module):
    """
    Feedforward network for transformer block.
    
    Standard 2-layer MLP with activation and dropout.
    
    Args:
        d_model: Model dimension
        d_ff: Hidden dimension
        dropout: Dropout probability
        activation: Activation function name
        orthogonal_init: Use orthogonal initialization
        device: Device to place tensors on
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        orthogonal_init: bool = True,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Two linear layers
        self.fc1 = nn.Linear(d_model, d_ff, device=device)
        self.fc2 = nn.Linear(d_ff, d_model, device=device)
        
        # Activation
        self.activation = self._get_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        if orthogonal_init:
            self._init_orthogonal()
        else:
            self._init_standard()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),  # Swish = SiLU
            "silu": nn.SiLU()
        }
        
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
        
        return activations[name.lower()]
    
    def _init_orthogonal(self):
        """Initialize with orthogonal weights."""
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def _init_standard(self):
        """Initialize with Xavier/Glorot."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x -> fc1 -> activation -> dropout -> fc2 -> dropout
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
        
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
    def get_statistics(self) -> Dict[str, float]:
        """Get orthogonality statistics for feedforward weights."""
        stats = {}
        
        for name, weight in [("fc1", self.fc1.weight), ("fc2", self.fc2.weight)]:
            # Compute ||W^T W - I||²_F
            gram = torch.matmul(weight.T, weight)
            identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            violation = torch.norm(gram - identity, p="fro") ** 2
            stats[f"ortho_violation_{name}"] = violation.item()
        
        return stats


# Testing code
if __name__ == "__main__":
    print("Testing DFATransformerBlock...")
    
    # Test 1: Basic forward pass
    print("\n=== Test 1: Basic Forward Pass ===")
    
    d_model = 256
    num_heads = 8
    batch_size = 4
    seq_len = 32
    
    block = DFATransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        causal=True,
        orthogonal_init=True
    )
    
    print(f"Block: {block}")
    print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")
    
    # Forward pass
    x = torch.randn(batch_size, seq_len, d_model)
    output, attn_weights = block(x, return_attention_weights=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    if attn_weights is not None:
        print(f"Attention weights shape: {attn_weights.shape}")
    
    assert output.shape == x.shape
    
    # Test 2: Statistics
    print("\n=== Test 2: Comprehensive Statistics ===")
    
    stats = block.get_statistics()
    print(f"Statistics ({len(stats)} entries):")
    
    # Group by category
    attn_stats = {k: v for k, v in stats.items() if k.startswith("attn_")}
    ff_stats = {k: v for k, v in stats.items() if k.startswith("ff_")}
    
    print(f"\nAttention statistics:")
    for key, val in sorted(attn_stats.items()):
        if 'violation' in key:
            print(f"  {key}: {val:.6f}")
        elif 'condition' in key:
            print(f"  {key}: {val:.2f}")
    
    print(f"\nFeedforward statistics:")
    for key, val in sorted(ff_stats.items()):
        print(f"  {key}: {val:.6f}")
    
    # Test 3: Gradient flow
    print("\n=== Test 3: Gradient Flow ===")
    
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    output, _ = block(x)
    
    loss = output.sum()
    loss.backward()
    
    print(f"Input gradient norm: {x.grad.norm().item():.6f}")
    
    # Check all modules have gradients
    modules = block.get_all_linear_modules()
    print(f"Module gradients:")
    for i, module in enumerate(modules):
        grad_norm = module.weight.grad.norm().item() if module.weight.grad is not None else 0.0
        print(f"  Module {i}: {grad_norm:.6f}")
    
    # Test 4: Module collection
    print("\n=== Test 4: Module Collection for DFA ===")
    
    attn_modules = block.get_attention_modules()
    ff_modules = block.get_feedforward_modules()
    all_modules = block.get_all_linear_modules()
    
    print(f"Attention modules: {len(attn_modules)}")
    print(f"Feedforward modules: {len(ff_modules)}")
    print(f"Total linear modules: {len(all_modules)}")
    
    # Test 5: Orthogonality weights
    print("\n=== Test 5: Orthogonality Weight Collection ===")
    
    ortho_weights = block.get_orthogonality_weights()
    print(f"Orthogonality-tracked weights: {len(ortho_weights)}")
    
    for name, weight in ortho_weights:
        print(f"  {name}: shape {weight.shape}")
    
    # Test 6: Different configurations
    print("\n=== Test 6: Various Configurations ===")
    
    configs = [
        {"d_model": 128, "num_heads": 4, "d_ff": 512, "activation": "relu"},
        {"d_model": 256, "num_heads": 8, "d_ff": 1024, "activation": "gelu"},
        {"d_model": 512, "num_heads": 16, "d_ff": 2048, "activation": "swish"},
    ]
    
    for i, config in enumerate(configs):
        block_test = DFATransformerBlock(**config)
        x_test = torch.randn(2, 16, config["d_model"])
        out_test, _ = block_test(x_test)
        
        print(f"  Config {i+1}: d_model={config['d_model']}, "
              f"num_heads={config['num_heads']}, "
              f"d_ff={config['d_ff']}, "
              f"activation={config['activation']}")
        print(f"    Output shape: {out_test.shape}")
        print(f"    Parameters: {sum(p.numel() for p in block_test.parameters()):,}")
    
    # Test 7: Non-causal mode
    print("\n=== Test 7: Non-Causal (Bidirectional) Mode ===")
    
    block_bidir = DFATransformerBlock(
        d_model=256,
        num_heads=8,
        causal=False  # Bidirectional
    )
    
    x = torch.randn(2, 32, 256)
    output, _ = block_bidir(x)
    
    print(f"Bidirectional output shape: {output.shape}")
    assert output.shape == x.shape
    
    # Test 8: FeedForward standalone
    print("\n=== Test 8: FeedForward Network Standalone ===")
    
    ff = FeedForward(d_model=256, d_ff=1024, activation="gelu")
    
    x = torch.randn(4, 32, 256)
    output = ff(x)
    
    print(f"FF input shape: {x.shape}")
    print(f"FF output shape: {output.shape}")
    
    ff_stats = ff.get_statistics()
    print(f"FF orthogonality violations:")
    for key, val in ff_stats.items():
        print(f"  {key}: {val:.6f}")
    
    print("\n✅ All tests completed!")
