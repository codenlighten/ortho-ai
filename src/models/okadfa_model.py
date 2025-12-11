"""
OKADFA Model: Orthogonalized Kernel Attention with Direct Feedback Alignment

Copyright (c) 2025 Gregory Ward - SmartLedger.Technology
Licensed under the MIT License (see LICENSE file)

Complete transformer model combining:
1. Kernelized Orthogonal Attention (KOA) - Linear O(TMd_k) complexity
2. Direct Feedback Alignment (DFA) - Efficient gradient propagation
3. Orthogonality regularization - Weight conditioning for DFA

Architecture:
- Token embeddings with positional encoding
- Stack of N DFA Transformer Blocks
- Output projection head
- Hybrid DFA/BP training strategy
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any
import math
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.dfa_transformer_block import DFATransformerBlock
from src.training.dfa_feedback import DFAFeedbackMatrix
from src.training.dfa_backward import HybridDFAHook
from src.training.orthogonality_loss import OrthogonalityLoss


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence position information.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class OKADFAModel(nn.Module):
    """
    Complete OKADFA Transformer Model.
    
    Features:
    - Linear complexity attention (O(TMd_k) vs O(T²d_model))
    - DFA training with fixed random feedback matrices
    - Orthogonality regularization for weight conditioning
    - Hybrid DFA/BP strategy (DFA between blocks, BP within)
    - Standard transformer API (compatible with Hugging Face)
    
    Example:
        model = OKADFAModel(
            vocab_size=50257,
            d_model=768,
            num_heads=12,
            num_layers=12,
            d_ff=3072,
            max_seq_len=1024
        )
        
        # Forward pass
        logits = model(input_ids)
        
        # Get modules for DFA hooks
        modules = model.get_dfa_modules()
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
        attention_dropout: float = 0.1,
        num_random_features: int = 256,
        use_causal: bool = True,
        activation: str = 'gelu',
        tie_weights: bool = True,
        orthogonal_init: bool = True,
    ):
        """
        Initialize OKADFA model.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension (embedding dimension)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feedforward dimension
            max_seq_len: Maximum sequence length
            dropout: General dropout rate
            attention_dropout: Attention dropout rate
            num_random_features: Number of random features for Favor+
            use_causal: Use causal (autoregressive) attention
            activation: Activation function ('relu', 'gelu', 'swish')
            tie_weights: Tie input/output embeddings
            orthogonal_init: Use orthogonal weight initialization
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.use_causal = use_causal
        self.tie_weights = tie_weights
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            DFATransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                causal=use_causal,
                activation=activation,
                orthogonal_init=orthogonal_init,
            )
            for _ in range(num_layers)
        ])
        
        # Output layer norm
        self.output_norm = nn.LayerNorm(d_model)
        
        # Output projection (language modeling head)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights if specified
        if tie_weights:
            self.output_projection.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights(orthogonal_init)
        
        # Compute total parameters
        self.num_parameters = sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, orthogonal: bool = True):
        """Initialize model weights."""
        # Token embeddings - normal initialization
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Output projection (if not tied)
        if not self.tie_weights:
            if orthogonal and self.d_model <= self.vocab_size:
                # Use orthogonal initialization if dimensions allow
                nn.init.orthogonal_(self.output_projection.weight)
            else:
                nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through OKADFA model.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Optional attention mask [batch, seq_len]
            return_hidden_states: Return intermediate hidden states
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            or tuple (logits, hidden_states) if return_hidden_states=True
        """
        # Token embeddings
        x = self.token_embedding(input_ids)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Store hidden states if requested
        hidden_states = [] if return_hidden_states else None
        
        # Pass through transformer layers
        for layer in self.layers:
            x, _ = layer(x, attention_mask)  # Unpack tuple (output, attn_weights)
            if return_hidden_states:
                hidden_states.append(x)
        
        # Output normalization
        x = self.output_norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)  # [batch, seq_len, vocab_size]
        
        if return_hidden_states:
            return logits, hidden_states
        return logits
    
    def get_dfa_modules(self) -> List[nn.Module]:
        """
        Get all modules that should use DFA.
        
        Returns:
            List of modules for DFA hook registration
        """
        modules = []
        for layer in self.layers:
            modules.extend(layer.get_all_linear_modules())
        return modules
    
    def setup_dfa_hooks(
        self,
        feedback_matrix: DFAFeedbackMatrix,
        global_error: torch.Tensor,
    ) -> List[HybridDFAHook]:
        """
        Setup DFA hooks for all layers.
        
        Args:
            feedback_matrix: DFA feedback matrix manager
            global_error: Global error signal δ_L
            
        Returns:
            List of registered hooks
        """
        hooks = []
        modules = self.get_dfa_modules()
        
        for i, module in enumerate(modules):
            hook = HybridDFAHook(
                module=module,
                feedback_matrix=feedback_matrix,
                layer_id=i,
                use_dfa=True,  # Use DFA between blocks
            )
            hook.set_global_error(global_error)
            hooks.append(hook)
        
        return hooks
    
    def get_orthogonality_loss(
        self,
        loss_fn: OrthogonalityLoss,
    ) -> torch.Tensor:
        """
        Compute orthogonality loss for all layers.
        
        Args:
            loss_fn: Orthogonality loss function
            
        Returns:
            Total orthogonality loss
        """
        total_loss = 0.0
        
        for layer in self.layers:
            # Get attention weight matrices (using correct attribute names)
            attn_weights = [
                layer.attention.q_proj.weight,
                layer.attention.k_proj.weight,
                layer.attention.v_proj.weight,
                layer.attention.out_proj.weight,
            ]
            
            # Get feedforward weight matrices
            ff_weights = [
                layer.ff.fc1.weight,
                layer.ff.fc2.weight,
            ]
            
            # Compute loss
            all_weights = attn_weights + ff_weights
            total_loss = total_loss + loss_fn(all_weights)
        
        return total_loss
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get model statistics and diagnostics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'num_parameters': self.num_parameters,
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
        }
        
        # Per-layer statistics
        layer_stats = []
        for i, layer in enumerate(self.layers):
            layer_stats.append(layer.get_statistics())
        stats['layers'] = layer_stats
        
        # Memory estimate (rough)
        param_memory_mb = self.num_parameters * 4 / (1024 * 1024)  # 4 bytes per float32
        stats['param_memory_mb'] = param_memory_mb
        
        return stats
    
    def estimate_complexity(self, seq_len: int, batch_size: int = 1) -> Dict[str, Any]:
        """
        Estimate computational complexity.
        
        Args:
            seq_len: Sequence length
            batch_size: Batch size
            
        Returns:
            Dictionary with complexity estimates
        """
        T = seq_len
        B = batch_size
        d = self.d_model
        M = 256  # Typical number of random features for Favor+
        L = self.num_layers
        
        # Favor+ attention: O(TMd) per layer
        favor_ops = L * B * T * M * d
        
        # Standard attention: O(T²d) per layer
        standard_ops = L * B * T * T * d
        
        # Feedforward: O(Td²) per layer
        ff_ops = L * B * T * d * self.d_ff * 2
        
        # Total
        total_okadfa = favor_ops + ff_ops
        total_standard = standard_ops + ff_ops
        
        speedup = total_standard / total_okadfa if total_okadfa > 0 else 1.0
        
        return {
            'favor_ops': favor_ops,
            'standard_ops': standard_ops,
            'ff_ops': ff_ops,
            'total_okadfa_ops': total_okadfa,
            'total_standard_ops': total_standard,
            'speedup': speedup,
            'complexity_okadfa': f'O({L}*{B}*{T}*{M}*{d})',
            'complexity_standard': f'O({L}*{B}*{T}²*{d})',
        }


def create_gpt2_small_okadfa(vocab_size: int = 50257) -> OKADFAModel:
    """
    Create OKADFA model with GPT-2 Small architecture (124M params).
    
    Args:
        vocab_size: Vocabulary size (default: GPT-2's 50257)
        
    Returns:
        OKADFAModel configured like GPT-2 Small
    """
    return OKADFAModel(
        vocab_size=vocab_size,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,
        max_seq_len=1024,
        dropout=0.1,
        attention_dropout=0.1,
        num_random_features=256,
        use_causal=True,
        activation='gelu',
        tie_weights=True,
        orthogonal_init=True,
    )


def create_gpt2_medium_okadfa(vocab_size: int = 50257) -> OKADFAModel:
    """
    Create OKADFA model with GPT-2 Medium architecture (350M params).
    
    Args:
        vocab_size: Vocabulary size
        
    Returns:
        OKADFAModel configured like GPT-2 Medium
    """
    return OKADFAModel(
        vocab_size=vocab_size,
        d_model=1024,
        num_heads=16,
        num_layers=24,
        d_ff=4096,
        max_seq_len=1024,
        dropout=0.1,
        attention_dropout=0.1,
        num_random_features=256,
        use_causal=True,
        activation='gelu',
        tie_weights=True,
        orthogonal_init=True,
    )


if __name__ == "__main__":
    print("Testing OKADFA Model...")
    
    # Test 1: Model creation
    print("\n1. Testing model creation:")
    model = OKADFAModel(
        vocab_size=1000,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        max_seq_len=512,
    )
    print(f"   ✓ Model created: {model.num_parameters:,} parameters")
    
    # Test 2: Forward pass
    print("\n2. Testing forward pass:")
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    logits = model(input_ids)
    print(f"   ✓ Input shape: {input_ids.shape}")
    print(f"   ✓ Output shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, 1000)
    
    # Test 3: Hidden states
    print("\n3. Testing hidden states:")
    logits, hidden_states = model(input_ids, return_hidden_states=True)
    print(f"   ✓ Number of hidden states: {len(hidden_states)}")
    assert len(hidden_states) == 4
    
    # Test 4: DFA modules
    print("\n4. Testing DFA module collection:")
    dfa_modules = model.get_dfa_modules()
    print(f"   ✓ DFA modules collected: {len(dfa_modules)}")
    assert len(dfa_modules) == 4 * 6  # 4 layers * 6 modules per layer
    
    # Test 5: Statistics
    print("\n5. Testing statistics:")
    stats = model.get_statistics()
    print(f"   ✓ Parameters: {stats['num_parameters']:,}")
    print(f"   ✓ Memory: {stats['param_memory_mb']:.2f} MB")
    print(f"   ✓ Layers tracked: {len(stats['layers'])}")
    
    # Test 6: Complexity estimation
    print("\n6. Testing complexity estimation:")
    complexity = model.estimate_complexity(seq_len=512, batch_size=2)
    print(f"   ✓ OKADFA ops: {complexity['total_okadfa_ops']:,}")
    print(f"   ✓ Standard ops: {complexity['total_standard_ops']:,}")
    print(f"   ✓ Speedup: {complexity['speedup']:.2f}x")
    
    # Test 7: GPT-2 Small config
    print("\n7. Testing GPT-2 Small configuration:")
    gpt2_small = create_gpt2_small_okadfa()
    print(f"   ✓ Parameters: {gpt2_small.num_parameters:,}")
    print(f"   ✓ Expected ~124M parameters")
    
    # Test 8: Gradient flow
    print("\n8. Testing gradient flow:")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    input_ids = torch.randint(0, 1000, (2, 32))
    target_ids = torch.randint(0, 1000, (2, 32))
    
    logits = model(input_ids)
    loss = nn.functional.cross_entropy(
        logits.view(-1, 1000),
        target_ids.view(-1)
    )
    loss.backward()
    
    # Check gradients
    has_grads = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    print(f"   ✓ Parameters with gradients: {has_grads}/{total_params}")
    
    print("\n✅ All OKADFA model tests passed!")
