"""
Favor+ Kernel Approximation (Performer's Positive Orthogonal Random Features)

Copyright (c) 2025 Gregory Ward - SmartLedger.Technology
Licensed under the MIT License (see LICENSE file)

Based on: Choromanski et al., "Rethinking Attention with Performers" (2020)
https://arxiv.org/abs/2009.14794

Key idea: Approximate softmax attention using positive orthogonal random features
to achieve linear complexity O(L) instead of quadratic O(L^2) in sequence length.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class FavorPlusFeatures(nn.Module):
    """
    Favor+ random feature map for approximating softmax kernel.
    
    Uses positive orthogonal random features (ORF) for better approximation
    quality and theoretical guarantees compared to standard random features.
    
    Args:
        d_model: Dimension of input features (d_k in attention)
        num_features: Number of random features M (typically 2*d_k to 4*d_k)
        orthogonal: Whether to use orthogonal random features (recommended)
        redraw_features: Whether to redraw features during training
        device: Device to place tensors on
    """
    
    def __init__(
        self,
        d_model: int,
        num_features: Optional[int] = None,
        orthogonal: bool = True,
        redraw_features: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_features = num_features or (2 * d_model)
        self.orthogonal = orthogonal
        self.redraw_features = redraw_features
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Normalization constant for feature map
        # sqrt(d_k) normalization ensures proper variance
        self.norm_constant = math.sqrt(self.d_model)
        
        # Initialize random projection matrix
        self.register_buffer(
            "projection_matrix",
            self._create_projection_matrix()
        )
        
    def _create_projection_matrix(self) -> torch.Tensor:
        """
        Create random projection matrix for feature map.
        
        Returns orthogonal random features if self.orthogonal=True,
        otherwise standard Gaussian random features.
        """
        if self.orthogonal:
            # Create orthogonal random matrix using QR decomposition
            # This provides better approximation quality
            num_blocks = math.ceil(self.num_features / self.d_model)
            blocks = []
            
            for _ in range(num_blocks):
                # Sample Gaussian matrix
                gaussian_block = torch.randn(self.d_model, self.d_model, device=self.device)
                # QR decomposition to get orthogonal matrix
                q, _ = torch.linalg.qr(gaussian_block)
                blocks.append(q)
            
            # Concatenate blocks and take first num_features columns
            projection = torch.cat(blocks, dim=1)[:, :self.num_features]
            
            # Apply random scaling to each row (improves isotropy)
            scaling = torch.randn(self.d_model, 1, device=self.device).abs().sqrt()
            projection = projection * scaling
            
        else:
            # Standard random features (Gaussian)
            projection = torch.randn(
                self.d_model, self.num_features, device=self.device
            ) / math.sqrt(self.d_model)
        
        return projection
    
    def redraw_projection_matrix(self):
        """Redraw the random projection matrix (useful for training stability)."""
        if self.redraw_features:
            self.projection_matrix = self._create_projection_matrix()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Favor+ feature map: φ(x) = exp(xW/√d - ||x||²/2) / √M
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Feature map φ(x) of shape (batch, seq_len, num_features)
        """
        # x shape: (B, L, d_model)
        batch_size, seq_len, d_model = x.shape
        
        # Ensure x is on the same device as projection matrix
        x = x.to(self.projection_matrix.device)
        
        # Project: xW where W is (d_model, num_features)
        # Result: (B, L, num_features)
        projected = torch.matmul(x, self.projection_matrix)
        
        # Compute squared norms: ||x||² for each position
        # Shape: (B, L, 1)
        x_squared_norm = torch.sum(x ** 2, dim=-1, keepdim=True) / 2.0
        
        # Apply exponential kernel with norm correction
        # φ(x) = exp(xW/√d_k - ||x||²/2)
        features = torch.exp(projected / self.norm_constant - x_squared_norm)
        
        # Normalize by √M for unbiased estimation
        features = features / math.sqrt(self.num_features)
        
        return features


class FavorPlusAttention(nn.Module):
    """
    Linear-complexity attention using Favor+ kernel approximation.
    
    Computes: Attention(Q, K, V) ≈ φ(Q) @ (φ(K)^T @ V)
    where φ is the Favor+ feature map.
    
    This reduces complexity from O(L²d) to O(Ld²) where L is sequence length.
    
    Args:
        d_model: Model dimension
        num_features: Number of random features (default: 2 * d_model)
        orthogonal: Use orthogonal random features
        causal: Whether to apply causal masking
        eps: Small constant for numerical stability
    """
    
    def __init__(
        self,
        d_model: int,
        num_features: Optional[int] = None,
        orthogonal: bool = True,
        causal: bool = False,
        eps: float = 1e-6,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.causal = causal
        self.eps = eps
        self.device = device or torch.device("cpu")
        
        # Feature map for Q and K
        self.feature_map = FavorPlusFeatures(
            d_model=d_model,
            num_features=num_features,
            orthogonal=orthogonal,
            device=self.device,
        )
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute kernelized attention.
        
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            attention_mask: Optional mask (not fully supported in kernel formulation)
            
        Returns:
            attention_output: (batch, seq_len, d_model)
            attention_weights: None (not computed for efficiency)
        """
        batch_size, seq_len, d_model = query.shape
        
        # Apply feature maps: φ(Q) and φ(K)
        q_prime = self.feature_map(query)  # (B, L, M)
        k_prime = self.feature_map(key)    # (B, L, M)
        
        if self.causal:
            # Causal attention: use cumulative sums for efficiency
            # This maintains O(L) complexity while enforcing causality
            attention_output = self._causal_attention(q_prime, k_prime, value)
        else:
            # Non-causal attention: φ(Q) @ (φ(K)^T @ V)
            # Compute in this order to maintain O(LMd) complexity
            
            # Step 1: φ(K)^T @ V -> (B, M, d_model)
            kv = torch.matmul(k_prime.transpose(1, 2), value)
            
            # Step 2: φ(Q) @ (φ(K)^T @ V) -> (B, L, d_model)
            attention_output = torch.matmul(q_prime, kv)
            
            # Normalization: divide by sum of attention weights
            # Attention weights sum = φ(Q) @ φ(K)^T @ 1 = φ(Q) @ (φ(K)^T @ 1)
            k_sum = torch.sum(k_prime, dim=1, keepdim=True)  # (B, 1, M)
            normalizer = torch.matmul(q_prime, k_sum.transpose(1, 2))  # (B, L, 1)
            normalizer = normalizer + self.eps  # Numerical stability
            
            attention_output = attention_output / normalizer
        
        # Note: We don't return attention weights as computing them would be O(L²)
        return attention_output, None
    
    def _causal_attention(
        self,
        q_prime: torch.Tensor,
        k_prime: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute causal (autoregressive) attention efficiently using cumulative sums.
        
        Maintains O(LMd) complexity while enforcing causality.
        """
        batch_size, seq_len, num_features = q_prime.shape
        _, _, d_model = value.shape
        
        # We'll compute attention autoregressively but vectorized
        # Using cumulative sums to avoid explicit loops
        
        # Expand for broadcasting: (B, L, M, d_model)
        k_prime_expanded = k_prime.unsqueeze(-1)  # (B, L, M, 1)
        value_expanded = value.unsqueeze(2)        # (B, L, 1, d_model)
        
        # Element-wise product: (B, L, M, d_model)
        kv_products = k_prime_expanded * value_expanded
        
        # Cumulative sum over sequence dimension for causality
        # cumsum[i] = sum of k_prime[j] * value[j] for all j <= i
        kv_cumsum = torch.cumsum(kv_products, dim=1)  # (B, L, M, d_model)
        
        # Similarly for normalization
        k_cumsum = torch.cumsum(k_prime, dim=1)  # (B, L, M)
        
        # Compute attention: sum over feature dimension
        # For each position i: φ(Q[i]) @ (sum_{j<=i} φ(K[j]) ⊗ V[j])
        attention_output = torch.einsum('blm,blmd->bld', q_prime, kv_cumsum)
        
        # Normalize
        normalizer = torch.einsum('blm,blm->bl', q_prime, k_cumsum)
        normalizer = normalizer.unsqueeze(-1) + self.eps
        
        attention_output = attention_output / normalizer
        
        return attention_output


def test_favor_plus():
    """Quick test to verify Favor+ implementation."""
    print("Testing Favor+ Implementation...")
    
    # Use CPU for testing to avoid device issues
    device = torch.device("cpu")
    
    batch_size = 2
    seq_len = 128
    d_model = 64
    num_features = 128
    
    # Create random Q, K, V on CPU
    Q = torch.randn(batch_size, seq_len, d_model, device=device)
    K = torch.randn(batch_size, seq_len, d_model, device=device)
    V = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # Test feature map
    feature_map = FavorPlusFeatures(d_model, num_features, device=device)
    q_prime = feature_map(Q)
    print(f"✓ Feature map output shape: {q_prime.shape}")
    assert q_prime.shape == (batch_size, seq_len, num_features)
    
    # Test attention
    attention = FavorPlusAttention(d_model, num_features, device=device)
    output, _ = attention(Q, K, V)
    print(f"✓ Attention output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model)
    
    # Test causal attention
    causal_attention = FavorPlusAttention(d_model, num_features, causal=True, device=device)
    causal_output, _ = causal_attention(Q, K, V)
    print(f"✓ Causal attention output shape: {causal_output.shape}")
    assert causal_output.shape == (batch_size, seq_len, d_model)
    
    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    test_favor_plus()
