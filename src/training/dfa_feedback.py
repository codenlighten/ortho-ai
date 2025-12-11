"""
DFA Feedback Matrix Implementation

Copyright (c) 2025 Gregory Ward - SmartLedger.Technology
Licensed under the MIT License (see LICENSE file)

Manages fixed random feedback matrices B_l for Direct Feedback Alignment.

Based on:
- Nøkland, "Direct Feedback Alignment Provides Learning in Deep Neural Networks" (2016)
- OKADFA Mathematical Specification v1.0

Key properties:
- B_l ∈ R^{d_l × d_final}
- B_l(i,j) ~ N(0, 1/√d_final)
- Fixed during training (never updated)
- Reproducible with fixed seed
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import math


class DFAFeedbackMatrix:
    """
    Manages fixed random feedback matrices for DFA.
    
    Each layer l has a feedback matrix B_l that projects the global error
    signal from the final layer back to the layer's local error signal:
    
        e_l = B_l δ_L
    
    where δ_L is the global error from the final loss.
    
    Args:
        layer_dims: List of layer dimensions [d_1, d_2, ..., d_L]
        output_dim: Final output dimension d_final (e.g., vocabulary size)
        seed: Fixed random seed for reproducibility
        device: Device to place tensors on
        dtype: Data type for tensors
    
    Example:
        >>> layer_dims = [512, 512, 512, 512]  # 4 layers
        >>> output_dim = 50257  # GPT-2 vocab size
        >>> feedback = DFAFeedbackMatrix(layer_dims, output_dim, seed=42)
        >>> 
        >>> # During backward pass:
        >>> global_error = torch.randn(batch_size, output_dim)
        >>> local_error = feedback.compute_local_error(layer_idx=0, global_error)
    """
    
    def __init__(
        self,
        layer_dims: List[int],
        output_dim: int,
        seed: int = 42,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.layer_dims = layer_dims
        self.output_dim = output_dim
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        # Initialize all feedback matrices
        self.feedback_matrices = self._initialize_feedback_matrices()
        
    def _initialize_feedback_matrices(self) -> Dict[int, torch.Tensor]:
        """
        Initialize fixed random feedback matrices for all layers.
        
        Returns:
            Dict mapping layer_idx to B_l tensor
        """
        # Set seed for reproducibility
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        
        feedback_matrices = {}
        
        for layer_idx, layer_dim in enumerate(self.layer_dims):
            # Initialize B_l ~ N(0, 1/√d_final)
            std = 1.0 / math.sqrt(self.output_dim)
            
            B_l = torch.randn(
                layer_dim,
                self.output_dim,
                generator=generator,
                device=self.device,
                dtype=self.dtype
            ) * std
            
            # Make it non-trainable
            B_l.requires_grad = False
            
            feedback_matrices[layer_idx] = B_l
            
        return feedback_matrices
    
    def get_feedback_matrix(self, layer_idx: int) -> torch.Tensor:
        """
        Get feedback matrix B_l for a specific layer.
        
        Args:
            layer_idx: Index of the layer (0-indexed)
            
        Returns:
            B_l: Feedback matrix of shape (d_l, d_final)
        """
        if layer_idx not in self.feedback_matrices:
            raise ValueError(
                f"Layer index {layer_idx} not found. "
                f"Valid indices: 0 to {len(self.layer_dims) - 1}"
            )
        
        return self.feedback_matrices[layer_idx]
    
    def compute_local_error(
        self,
        layer_idx: int,
        global_error: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute local error signal for a layer using DFA.
        
        Computes: e_l = B_l δ_L
        
        Args:
            layer_idx: Index of the layer
            global_error: Global error signal δ_L of shape (batch, d_final)
            
        Returns:
            e_l: Local error signal of shape (batch, d_l)
        """
        B_l = self.get_feedback_matrix(layer_idx)
        
        # Ensure B_l is on same device as global_error
        if B_l.device != global_error.device:
            B_l = B_l.to(global_error.device)
            # Update stored matrix to new device
            self.feedback_matrices[layer_idx] = B_l
        
        # e_l = B_l @ δ_L^T -> (d_l, d_final) @ (d_final, ...) -> (d_l, ...)
        # Then transpose to (..., d_l)
        if global_error.dim() == 1:
            # Single sample: (d_final,)
            e_l = torch.matmul(B_l, global_error)
        elif global_error.dim() == 2:
            # Batched: (batch, d_final)
            # Transpose for matmul: (batch, d_final) -> (d_final, batch)
            # B_l @ global_error^T = (d_l, d_final) @ (d_final, batch) -> (d_l, batch)
            # Transpose back: (d_l, batch) -> (batch, d_l)
            e_l = torch.matmul(B_l, global_error.T).T
        elif global_error.dim() == 3:
            # Sequence: (batch, seq, d_final)
            # Reshape to (batch * seq, d_final), compute, reshape back
            batch, seq, d_final = global_error.shape
            global_error_flat = global_error.reshape(-1, d_final)
            e_l_flat = torch.matmul(B_l, global_error_flat.T).T  # (batch*seq, d_l)
            e_l = e_l_flat.reshape(batch, seq, -1)  # (batch, seq, d_l)
        else:
            raise ValueError(f"Unsupported global_error dimensions: {global_error.dim()}")
        
        return e_l
    
    def get_statistics(self, layer_idx: Optional[int] = None) -> Dict[str, float]:
        """
        Get statistics about feedback matrices for diagnostics.
        
        Args:
            layer_idx: If provided, get stats for specific layer only
            
        Returns:
            Dict with statistics (mean, std, norm, etc.)
        """
        if layer_idx is not None:
            B_l = self.get_feedback_matrix(layer_idx)
            return {
                f"layer_{layer_idx}_mean": B_l.mean().item(),
                f"layer_{layer_idx}_std": B_l.std().item(),
                f"layer_{layer_idx}_norm": B_l.norm().item(),
                f"layer_{layer_idx}_max": B_l.max().item(),
                f"layer_{layer_idx}_min": B_l.min().item(),
            }
        else:
            # Get stats for all layers
            stats = {}
            for idx in range(len(self.layer_dims)):
                stats.update(self.get_statistics(idx))
            return stats
    
    def to(self, device: torch.device) -> 'DFAFeedbackMatrix':
        """Move all feedback matrices to a different device."""
        self.device = device
        for layer_idx in self.feedback_matrices:
            self.feedback_matrices[layer_idx] = self.feedback_matrices[layer_idx].to(device)
        return self
    
    def __len__(self) -> int:
        """Number of layers with feedback matrices."""
        return len(self.layer_dims)
    
    def __repr__(self) -> str:
        return (
            f"DFAFeedbackMatrix(\n"
            f"  num_layers={len(self.layer_dims)},\n"
            f"  layer_dims={self.layer_dims},\n"
            f"  output_dim={self.output_dim},\n"
            f"  seed={self.seed},\n"
            f"  device={self.device}\n"
            f")"
        )


def test_dfa_feedback_basic():
    """Quick test of basic functionality."""
    print("Testing DFA Feedback Matrix...")
    
    # Setup
    layer_dims = [256, 512, 512, 256]
    output_dim = 1024
    batch_size = 4
    
    feedback = DFAFeedbackMatrix(
        layer_dims=layer_dims,
        output_dim=output_dim,
        seed=42,
        device=torch.device("cpu")
    )
    
    print(f"✓ Created feedback matrices for {len(feedback)} layers")
    
    # Test getting feedback matrix
    for layer_idx in range(len(layer_dims)):
        B_l = feedback.get_feedback_matrix(layer_idx)
        expected_shape = (layer_dims[layer_idx], output_dim)
        assert B_l.shape == expected_shape, f"Layer {layer_idx}: Expected {expected_shape}, got {B_l.shape}"
        print(f"✓ Layer {layer_idx}: B_l shape {B_l.shape}")
    
    # Test local error computation
    global_error = torch.randn(batch_size, output_dim)
    for layer_idx in range(len(layer_dims)):
        e_l = feedback.compute_local_error(layer_idx, global_error)
        expected_shape = (batch_size, layer_dims[layer_idx])
        assert e_l.shape == expected_shape, f"Expected {expected_shape}, got {e_l.shape}"
        print(f"✓ Layer {layer_idx}: e_l shape {e_l.shape}")
    
    # Test single sample (no batch dim)
    global_error_single = torch.randn(output_dim)
    e_l_single = feedback.compute_local_error(0, global_error_single)
    assert e_l_single.shape == (layer_dims[0],)
    print(f"✓ Single sample: e_l shape {e_l_single.shape}")
    
    # Test statistics
    stats = feedback.get_statistics(layer_idx=0)
    print(f"✓ Layer 0 statistics: {stats}")
    
    # Test initialization scale
    B_0 = feedback.get_feedback_matrix(0)
    actual_std = B_0.std().item()
    expected_std = 1.0 / math.sqrt(output_dim)
    # Allow 20% tolerance due to random sampling
    assert abs(actual_std - expected_std) / expected_std < 0.2, \
        f"Std {actual_std:.6f} too far from expected {expected_std:.6f}"
    print(f"✓ Initialization scale: std={actual_std:.6f}, expected={expected_std:.6f}")
    
    # Test reproducibility
    feedback2 = DFAFeedbackMatrix(
        layer_dims=layer_dims,
        output_dim=output_dim,
        seed=42,  # Same seed
        device=torch.device("cpu")
    )
    
    B_0_v1 = feedback.get_feedback_matrix(0)
    B_0_v2 = feedback2.get_feedback_matrix(0)
    assert torch.allclose(B_0_v1, B_0_v2), "Reproducibility failed!"
    print(f"✓ Reproducibility: identical matrices with same seed")
    
    # Test different seed gives different matrices
    feedback3 = DFAFeedbackMatrix(
        layer_dims=layer_dims,
        output_dim=output_dim,
        seed=123,  # Different seed
        device=torch.device("cpu")
    )
    
    B_0_v3 = feedback3.get_feedback_matrix(0)
    assert not torch.allclose(B_0_v1, B_0_v3), "Different seeds should give different matrices"
    print(f"✓ Different seed produces different matrices")
    
    print("\nAll basic tests passed! ✓")


if __name__ == "__main__":
    test_dfa_feedback_basic()
