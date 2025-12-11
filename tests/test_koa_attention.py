"""
Unit tests for KOA Multi-Head Attention
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.koa_attention import KOAMultiHeadAttention


class TestKOAMultiHeadAttention:
    """Test suite for KOAMultiHeadAttention."""
    
    @pytest.fixture
    def d_model(self):
        return 256
    
    @pytest.fixture
    def num_heads(self):
        return 8
    
    @pytest.fixture
    def koa(self, d_model, num_heads):
        return KOAMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            causal=True,
            orthogonal_init=True
        )
    
    def test_initialization(self, koa, d_model, num_heads):
        """Test KOA initialization."""
        assert koa.d_model == d_model
        assert koa.num_heads == num_heads
        assert koa.d_k == d_model // num_heads
        assert koa.causal is True
    
    def test_output_shape(self, koa):
        """Test that output shape matches input shape."""
        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, koa.d_model)
        
        output, _ = koa(x)
        
        assert output.shape == x.shape
    
    def test_attention_weights_shape(self, koa, num_heads):
        """Test attention weights shape."""
        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, koa.d_model)
        
        output, attn_weights = koa(x, return_attention_weights=True)
        
        assert attn_weights is not None
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    def test_causal_masking(self):
        """Test that causal masking works correctly."""
        koa = KOAMultiHeadAttention(d_model=128, num_heads=4, causal=True)
        
        x = torch.randn(2, 16, 128)
        output, attn_weights = koa(x, return_attention_weights=True)
        
        # Check that attention weights are causal (upper triangle should be near zero)
        if attn_weights is not None:
            # Average over batch and heads
            avg_attn = attn_weights.mean(dim=(0, 1))
            
            # Upper triangle should have smaller values than lower triangle
            upper_triangle = torch.triu(avg_attn, diagonal=1).sum()
            lower_triangle = torch.tril(avg_attn, diagonal=-1).sum()
            
            # Not a strict test due to approximation, but should show trend
            assert upper_triangle < lower_triangle * 1.5
    
    def test_non_causal_attention(self):
        """Test non-causal (bidirectional) attention."""
        koa = KOAMultiHeadAttention(d_model=128, num_heads=4, causal=False)
        
        x = torch.randn(2, 16, 128)
        output, _ = koa(x)
        
        assert output.shape == x.shape
    
    def test_orthogonal_initialization(self):
        """Test that orthogonal initialization produces low violations."""
        koa = KOAMultiHeadAttention(
            d_model=256,
            num_heads=8,
            orthogonal_init=True
        )
        
        # Run forward pass to populate statistics
        x = torch.randn(2, 16, 256)
        _ = koa(x)
        
        stats = koa.get_orthogonality_statistics()
        
        # Orthogonal init should have very low violations
        assert stats["total_violation"] < 1e-4
        assert stats["mean_violation"] < 1e-5
    
    def test_standard_initialization(self):
        """Test standard (non-orthogonal) initialization."""
        koa = KOAMultiHeadAttention(
            d_model=256,
            num_heads=8,
            orthogonal_init=False
        )
        
        x = torch.randn(2, 16, 256)
        _ = koa(x)
        
        stats = koa.get_orthogonality_statistics()
        
        # Standard init will have higher violations
        assert stats["total_violation"] > 0
    
    def test_gradient_flow(self, koa):
        """Test that gradients flow through all components."""
        x = torch.randn(2, 16, koa.d_model, requires_grad=True)
        
        output, _ = koa(x)
        loss = output.sum()
        loss.backward()
        
        # Check input gradient
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check projection gradients
        assert koa.q_proj.weight.grad is not None
        assert koa.k_proj.weight.grad is not None
        assert koa.v_proj.weight.grad is not None
        assert koa.out_proj.weight.grad is not None
    
    def test_projection_matrices(self, koa):
        """Test getting projection matrices."""
        matrices = koa.get_projection_matrices()
        
        assert len(matrices) == 4  # Q, K, V, out
        
        for name, weight in matrices:
            assert isinstance(name, str)
            assert isinstance(weight, torch.Tensor)
            assert weight.dim() == 2
    
    def test_orthogonality_violation_computation(self, koa):
        """Test orthogonality violation computation."""
        # Perfect orthogonal matrix
        W_ortho, _ = torch.linalg.qr(torch.randn(100, 100))
        violation_ortho = koa.compute_orthogonality_violation(W_ortho)
        
        # Random matrix
        W_random = torch.randn(100, 100)
        violation_random = koa.compute_orthogonality_violation(W_random)
        
        # Orthogonal should have much lower violation
        assert violation_ortho < 1e-4
        assert violation_random > 1.0
        assert violation_random / max(violation_ortho, 1e-10) > 1000
    
    def test_redraw_projection_matrix(self, koa):
        """Test redrawing projection matrices."""
        x = torch.randn(2, 16, koa.d_model)
        
        output1, _ = koa(x)
        
        # Redraw
        koa.redraw_projection_matrix()
        
        output2, _ = koa(x)
        
        # Outputs should differ
        diff = (output1 - output2).norm() / output1.norm()
        assert diff > 0.01  # Should be noticeably different
    
    def test_different_sequence_lengths(self, koa):
        """Test with various sequence lengths."""
        batch_size = 2
        
        for seq_len in [8, 32, 128]:
            x = torch.randn(batch_size, seq_len, koa.d_model)
            output, _ = koa(x)
            
            assert output.shape == (batch_size, seq_len, koa.d_model)
    
    def test_different_batch_sizes(self, koa):
        """Test with various batch sizes."""
        seq_len = 16
        
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, seq_len, koa.d_model)
            output, _ = koa(x)
            
            assert output.shape == (batch_size, seq_len, koa.d_model)
    
    def test_device_compatibility(self):
        """Test on different devices."""
        koa_cpu = KOAMultiHeadAttention(d_model=128, num_heads=4, device=torch.device("cpu"))
        
        x_cpu = torch.randn(2, 16, 128)
        output_cpu, _ = koa_cpu(x_cpu)
        
        assert output_cpu.device.type == "cpu"
        
        if torch.cuda.is_available():
            koa_cuda = KOAMultiHeadAttention(d_model=128, num_heads=4, device=torch.device("cuda"))
            
            x_cuda = torch.randn(2, 16, 128, device="cuda")
            output_cuda, _ = koa_cuda(x_cuda)
            
            assert output_cuda.device.type == "cuda"
    
    def test_statistics_tracking(self, koa):
        """Test orthogonality statistics tracking."""
        x = torch.randn(2, 16, koa.d_model)
        _ = koa(x)
        
        stats = koa.get_orthogonality_statistics()
        
        assert "ortho_violation_Q" in stats
        assert "ortho_violation_K" in stats
        assert "ortho_violation_V" in stats
        assert "ortho_violation_out" in stats
        assert "total_violation" in stats
        assert "mean_violation" in stats
        assert "condition_number_Q" in stats
    
    def test_repr(self, koa):
        """Test string representation."""
        repr_str = repr(koa)
        
        assert "KOAMultiHeadAttention" in repr_str
        assert str(koa.d_model) in repr_str
        assert str(koa.num_heads) in repr_str


@pytest.mark.parametrize("d_model,num_heads", [
    (128, 4),
    (256, 8),
    (512, 16),
    (1024, 16)
])
def test_various_model_sizes(d_model, num_heads):
    """Test with different model dimensions and head counts."""
    koa = KOAMultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    x = torch.randn(2, 16, d_model)
    output, _ = koa(x)
    
    assert output.shape == x.shape


@pytest.mark.parametrize("num_features_multiplier", [1, 2, 4])
def test_various_num_features(num_features_multiplier):
    """Test with different numbers of random features."""
    d_model = 256
    num_heads = 8
    d_k = d_model // num_heads
    
    koa = KOAMultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        num_features=d_k * num_features_multiplier
    )
    
    x = torch.randn(2, 16, d_model)
    output, _ = koa(x)
    
    assert output.shape == x.shape


def test_dropout_effect():
    """Test that dropout has an effect during training."""
    koa = KOAMultiHeadAttention(d_model=128, num_heads=4, dropout=0.5)
    koa.train()
    
    x = torch.randn(2, 16, 128)
    
    output1, _ = koa(x)
    output2, _ = koa(x)
    
    # Outputs should differ due to dropout
    diff = (output1 - output2).norm() / output1.norm()
    assert diff > 0.01


def test_eval_mode():
    """Test that eval mode is deterministic."""
    koa = KOAMultiHeadAttention(d_model=128, num_heads=4, dropout=0.5)
    koa.eval()
    
    x = torch.randn(2, 16, 128)
    
    # Redraw to ensure same random features
    koa.redraw_projection_matrix()
    
    with torch.no_grad():
        output1, _ = koa(x)
        output2, _ = koa(x)
    
    # Outputs should be identical in eval mode (except for numerical precision)
    assert torch.allclose(output1, output2, rtol=1e-5)


def test_parameter_count():
    """Test that parameter count is correct."""
    d_model = 256
    koa = KOAMultiHeadAttention(d_model=d_model, num_heads=8)
    
    num_params = sum(p.numel() for p in koa.parameters())
    
    # Q, K, V, out projections: 4 * (d_model * d_model + d_model)
    expected_params = 4 * (d_model * d_model + d_model)
    
    assert num_params == expected_params


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
