"""
Unit tests for Favor+ kernel approximation.
"""

import pytest
import torch
import math
from kernels.favor_plus import FavorPlusFeatures, FavorPlusAttention


class TestFavorPlusFeatures:
    """Test suite for FavorPlusFeatures class."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cpu")
    
    @pytest.fixture
    def feature_map(self, device):
        """Create a standard feature map for testing."""
        return FavorPlusFeatures(
            d_model=64,
            num_features=128,
            orthogonal=True,
            device=device
        )
    
    def test_output_shape(self, feature_map, device):
        """Test that output has correct shape."""
        batch_size, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        output = feature_map(x)
        
        assert output.shape == (batch_size, seq_len, 128)
    
    def test_orthogonal_projection(self, device):
        """Test that orthogonal projection matrix has correct properties."""
        d_model = 64
        num_features = 128
        
        feature_map = FavorPlusFeatures(
            d_model=d_model,
            num_features=num_features,
            orthogonal=True,
            device=device
        )
        
        proj = feature_map.projection_matrix  # (64, 128)
        
        # Check shape
        assert proj.shape == (d_model, num_features)
        
        # For orthogonal random features, the matrix is constructed from
        # orthogonal blocks with random scaling, so we don't expect perfect
        # orthogonality of the full matrix, but rather that it's well-conditioned
        # Just check that the matrix is non-degenerate
        _, s, _ = torch.svd(proj)
        condition_number = s[0] / s[-1]
        
        # Well-conditioned matrix should have reasonable condition number
        assert condition_number < 100, f"Poor conditioning: {condition_number}"
    
    def test_gaussian_projection(self, device):
        """Test that Gaussian (non-orthogonal) projection works."""
        feature_map = FavorPlusFeatures(
            d_model=64,
            num_features=128,
            orthogonal=False,
            device=device
        )
        
        x = torch.randn(2, 32, 64, device=device)
        output = feature_map(x)
        
        assert output.shape == (2, 32, 128)
    
    def test_positive_features(self, feature_map, device):
        """Test that all features are positive (from exp function)."""
        x = torch.randn(2, 32, 64, device=device)
        output = feature_map(x)
        
        # All values should be positive due to exp
        assert torch.all(output > 0)
    
    def test_redraw_projection(self, feature_map):
        """Test that projection matrix can be redrawn."""
        old_proj = feature_map.projection_matrix.clone()
        
        feature_map.redraw_features = True
        feature_map.redraw_projection_matrix()
        new_proj = feature_map.projection_matrix
        
        # Matrices should be different after redrawing
        assert not torch.allclose(old_proj, new_proj)


class TestFavorPlusAttention:
    """Test suite for FavorPlusAttention class."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cpu")
    
    @pytest.fixture
    def attention(self, device):
        """Create standard attention module."""
        return FavorPlusAttention(
            d_model=64,
            num_features=128,
            orthogonal=True,
            causal=False,
            device=device
        )
    
    @pytest.fixture
    def causal_attention(self, device):
        """Create causal attention module."""
        return FavorPlusAttention(
            d_model=64,
            num_features=128,
            orthogonal=True,
            causal=True,
            device=device
        )
    
    def test_output_shape(self, attention, device):
        """Test that attention output has correct shape."""
        batch_size, seq_len, d_model = 2, 32, 64
        
        Q = torch.randn(batch_size, seq_len, d_model, device=device)
        K = torch.randn(batch_size, seq_len, d_model, device=device)
        V = torch.randn(batch_size, seq_len, d_model, device=device)
        
        output, weights = attention(Q, K, V)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights is None  # We don't compute explicit weights
    
    def test_causal_output_shape(self, causal_attention, device):
        """Test that causal attention output has correct shape."""
        batch_size, seq_len, d_model = 2, 32, 64
        
        Q = torch.randn(batch_size, seq_len, d_model, device=device)
        K = torch.randn(batch_size, seq_len, d_model, device=device)
        V = torch.randn(batch_size, seq_len, d_model, device=device)
        
        output, _ = causal_attention(Q, K, V)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_self_attention(self, attention, device):
        """Test self-attention (Q=K=V)."""
        batch_size, seq_len, d_model = 2, 16, 64
        
        X = torch.randn(batch_size, seq_len, d_model, device=device)
        output, _ = attention(X, X, X)
        
        assert output.shape == (batch_size, seq_len, d_model)
        # Output should not be identical to input (attention aggregates)
        assert not torch.allclose(output, X)
    
    def test_causal_mask_effect(self, causal_attention, device):
        """Test that causal masking prevents future positions from affecting past."""
        batch_size, seq_len, d_model = 1, 16, 64  # Longer sequence for better test
        
        # Create deterministic inputs
        torch.manual_seed(42)
        Q = torch.randn(batch_size, seq_len, d_model, device=device)
        K = torch.randn(batch_size, seq_len, d_model, device=device)
        
        # Create V with distinct early and late values
        V = torch.zeros(batch_size, seq_len, d_model, device=device)
        V[:, :seq_len//2, :] = 1.0  # Early positions
        V[:, seq_len//2:, :] = -1.0  # Late positions
        
        # Get output with causal attention
        causal_output, _ = causal_attention(Q, K, V)
        
        # First position should only see itself (all 1s)
        # and should be positive since V[0] = 1.0
        first_pos_output = causal_output[0, 0, :]
        assert torch.mean(first_pos_output) > 0, "First position should be positive"
        
        # Early positions (before midpoint) should also be mostly positive
        # since they only see early V values (which are 1.0)
        early_output = causal_output[0, :seq_len//4, :]
        assert torch.mean(early_output) > 0, "Early positions should be positive"
    
    def test_attention_approximation_quality(self, device):
        """
        Test that kernel attention approximates softmax attention.
        
        We can't expect exact match, but should see some correlation.
        """
        torch.manual_seed(42)  # For reproducibility
        batch_size, seq_len, d_model = 2, 16, 32
        num_features = 256  # More features for better approximation
        
        Q = torch.randn(batch_size, seq_len, d_model, device=device) * 0.1  # Smaller values
        K = torch.randn(batch_size, seq_len, d_model, device=device) * 0.1
        V = torch.randn(batch_size, seq_len, d_model, device=device)
        
        # Kernel attention
        kernel_attn = FavorPlusAttention(d_model, num_features, device=device)
        kernel_output, _ = kernel_attn(Q, K, V)
        
        # Softmax attention (for comparison)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        softmax_output = torch.matmul(attn_weights, V)
        
        # Compute cosine similarity
        kernel_flat = kernel_output.flatten()
        softmax_flat = softmax_output.flatten()
        similarity = torch.nn.functional.cosine_similarity(
            kernel_flat.unsqueeze(0),
            softmax_flat.unsqueeze(0)
        )
        
        # With proper setup and enough random features, should achieve decent similarity
        # This is a sanity check - in practice, approximation quality varies
        print(f"Approximation similarity: {similarity.item():.4f}")
        assert similarity > 0.5, f"Similarity {similarity.item():.4f} too low - possible implementation bug"
        
        # Also check that outputs are in similar ranges
        kernel_std = kernel_output.std()
        softmax_std = softmax_output.std()
        std_ratio = kernel_std / softmax_std
        assert 0.1 < std_ratio < 10, f"Output scales very different: {std_ratio:.4f}"
    
    def test_gradient_flow(self, attention, device):
        """Test that gradients flow through the attention mechanism."""
        batch_size, seq_len, d_model = 2, 8, 64
        
        Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        
        output, _ = attention(Q, K, V)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist and are non-zero
        assert Q.grad is not None
        assert K.grad is not None
        assert V.grad is not None
        assert torch.any(Q.grad != 0)
        assert torch.any(K.grad != 0)
        assert torch.any(V.grad != 0)


@pytest.mark.parametrize("seq_len", [16, 64, 256])
def test_different_sequence_lengths(seq_len):
    """Test that attention works with different sequence lengths."""
    device = torch.device("cpu")
    batch_size, d_model = 2, 64
    
    Q = torch.randn(batch_size, seq_len, d_model, device=device)
    K = torch.randn(batch_size, seq_len, d_model, device=device)
    V = torch.randn(batch_size, seq_len, d_model, device=device)
    
    attention = FavorPlusAttention(d_model, device=device)
    output, _ = attention(Q, K, V)
    
    assert output.shape == (batch_size, seq_len, d_model)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
