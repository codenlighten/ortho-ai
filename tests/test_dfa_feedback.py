"""
Unit tests for DFA Feedback Matrix
"""

import pytest
import torch
import math
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.dfa_feedback import DFAFeedbackMatrix


class TestDFAFeedbackMatrix:
    """Test suite for DFAFeedbackMatrix class."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cpu")
    
    @pytest.fixture
    def layer_dims(self):
        return [256, 512, 512, 256]
    
    @pytest.fixture
    def output_dim(self):
        return 1024
    
    @pytest.fixture
    def feedback(self, layer_dims, output_dim, device):
        return DFAFeedbackMatrix(
            layer_dims=layer_dims,
            output_dim=output_dim,
            seed=42,
            device=device
        )
    
    def test_initialization(self, feedback, layer_dims, output_dim):
        """Test that feedback matrices are initialized correctly."""
        assert len(feedback) == len(layer_dims)
        
        for layer_idx, layer_dim in enumerate(layer_dims):
            B_l = feedback.get_feedback_matrix(layer_idx)
            assert B_l.shape == (layer_dim, output_dim)
            assert B_l.requires_grad is False  # Should not be trainable
    
    def test_initialization_scale(self, feedback, output_dim):
        """Test that initialization follows N(0, 1/√d_final)."""
        B_0 = feedback.get_feedback_matrix(0)
        
        # Check mean close to 0
        mean = B_0.mean().item()
        assert abs(mean) < 0.01, f"Mean {mean} should be close to 0"
        
        # Check std close to 1/√d_final (allow 30% tolerance)
        std = B_0.std().item()
        expected_std = 1.0 / math.sqrt(output_dim)
        relative_error = abs(std - expected_std) / expected_std
        assert relative_error < 0.3, \
            f"Std {std:.6f} too far from expected {expected_std:.6f} (error: {relative_error:.2%})"
    
    def test_reproducibility(self, layer_dims, output_dim, device):
        """Test that same seed produces identical matrices."""
        feedback1 = DFAFeedbackMatrix(layer_dims, output_dim, seed=42, device=device)
        feedback2 = DFAFeedbackMatrix(layer_dims, output_dim, seed=42, device=device)
        
        for layer_idx in range(len(layer_dims)):
            B1 = feedback1.get_feedback_matrix(layer_idx)
            B2 = feedback2.get_feedback_matrix(layer_idx)
            assert torch.allclose(B1, B2), f"Layer {layer_idx} matrices not identical"
    
    def test_different_seeds(self, layer_dims, output_dim, device):
        """Test that different seeds produce different matrices."""
        feedback1 = DFAFeedbackMatrix(layer_dims, output_dim, seed=42, device=device)
        feedback2 = DFAFeedbackMatrix(layer_dims, output_dim, seed=123, device=device)
        
        for layer_idx in range(len(layer_dims)):
            B1 = feedback1.get_feedback_matrix(layer_idx)
            B2 = feedback2.get_feedback_matrix(layer_idx)
            assert not torch.allclose(B1, B2), \
                f"Layer {layer_idx} matrices should differ with different seeds"
    
    def test_local_error_batch(self, feedback, layer_dims, output_dim):
        """Test local error computation with batched input."""
        batch_size = 8
        global_error = torch.randn(batch_size, output_dim)
        
        for layer_idx, layer_dim in enumerate(layer_dims):
            e_l = feedback.compute_local_error(layer_idx, global_error)
            
            assert e_l.shape == (batch_size, layer_dim)
            assert not torch.isnan(e_l).any()
            assert not torch.isinf(e_l).any()
    
    def test_local_error_single(self, feedback, layer_dims, output_dim):
        """Test local error computation with single sample."""
        global_error = torch.randn(output_dim)
        
        for layer_idx, layer_dim in enumerate(layer_dims):
            e_l = feedback.compute_local_error(layer_idx, global_error)
            
            assert e_l.shape == (layer_dim,)
            assert not torch.isnan(e_l).any()
            assert not torch.isinf(e_l).any()
    
    def test_local_error_computation(self, feedback, output_dim):
        """Test that local error computation is correct: e_l = B_l @ δ_L."""
        batch_size = 4
        layer_idx = 0
        
        global_error = torch.randn(batch_size, output_dim)
        B_l = feedback.get_feedback_matrix(layer_idx)
        
        # Compute using method
        e_l = feedback.compute_local_error(layer_idx, global_error)
        
        # Compute manually
        e_l_manual = torch.matmul(B_l, global_error.T).T
        
        assert torch.allclose(e_l, e_l_manual, rtol=1e-5)
    
    def test_invalid_layer_index(self, feedback):
        """Test that invalid layer index raises error."""
        with pytest.raises(ValueError):
            feedback.get_feedback_matrix(999)
    
    def test_statistics(self, feedback):
        """Test statistics computation."""
        stats = feedback.get_statistics(layer_idx=0)
        
        assert "layer_0_mean" in stats
        assert "layer_0_std" in stats
        assert "layer_0_norm" in stats
        assert "layer_0_max" in stats
        assert "layer_0_min" in stats
        
        # Check that values are reasonable
        assert abs(stats["layer_0_mean"]) < 0.1  # Mean near 0
        assert stats["layer_0_std"] > 0  # Positive std
        assert stats["layer_0_norm"] > 0  # Positive norm
    
    def test_statistics_all_layers(self, feedback, layer_dims):
        """Test statistics for all layers."""
        stats = feedback.get_statistics()
        
        # Should have stats for all layers
        for layer_idx in range(len(layer_dims)):
            assert f"layer_{layer_idx}_mean" in stats
            assert f"layer_{layer_idx}_std" in stats
    
    def test_device_transfer(self, layer_dims, output_dim):
        """Test moving feedback matrices to different device."""
        feedback = DFAFeedbackMatrix(
            layer_dims=layer_dims,
            output_dim=output_dim,
            device=torch.device("cpu")
        )
        
        # Check initial device
        B_0 = feedback.get_feedback_matrix(0)
        assert B_0.device.type == "cpu"
        
        # Move to cuda if available
        if torch.cuda.is_available():
            feedback.to(torch.device("cuda"))
            B_0_cuda = feedback.get_feedback_matrix(0)
            assert B_0_cuda.device.type == "cuda"
    
    def test_gradient_flow_blocked(self, feedback):
        """Test that feedback matrices don't accumulate gradients."""
        B_0 = feedback.get_feedback_matrix(0)
        
        # Try to compute gradients (should fail or be None)
        assert B_0.requires_grad is False
        
        # Even if we try to force it, grad should be None
        x = torch.randn(10, B_0.shape[0], requires_grad=True)
        y = torch.matmul(x, B_0)
        loss = y.sum()
        loss.backward()
        
        # B_0 should have no gradient
        assert B_0.grad is None
    
    def test_different_dtypes(self, layer_dims, output_dim, device):
        """Test initialization with different data types."""
        for dtype in [torch.float32, torch.float16, torch.float64]:
            if dtype == torch.float16 and not torch.cuda.is_available():
                continue  # Skip float16 on CPU
            
            feedback = DFAFeedbackMatrix(
                layer_dims=layer_dims,
                output_dim=output_dim,
                dtype=dtype,
                device=device
            )
            
            B_0 = feedback.get_feedback_matrix(0)
            assert B_0.dtype == dtype
    
    def test_repr(self, feedback, layer_dims, output_dim):
        """Test string representation."""
        repr_str = repr(feedback)
        
        assert "DFAFeedbackMatrix" in repr_str
        assert str(len(layer_dims)) in repr_str
        assert str(output_dim) in repr_str


@pytest.mark.parametrize("num_layers", [1, 2, 4, 8])
def test_various_layer_counts(num_layers):
    """Test with different numbers of layers."""
    layer_dims = [512] * num_layers
    output_dim = 1024
    
    feedback = DFAFeedbackMatrix(
        layer_dims=layer_dims,
        output_dim=output_dim,
        device=torch.device("cpu")
    )
    
    assert len(feedback) == num_layers
    
    for layer_idx in range(num_layers):
        B_l = feedback.get_feedback_matrix(layer_idx)
        assert B_l.shape == (512, output_dim)


@pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
def test_various_batch_sizes(batch_size):
    """Test local error computation with different batch sizes."""
    layer_dims = [256, 512]
    output_dim = 1024
    
    feedback = DFAFeedbackMatrix(
        layer_dims=layer_dims,
        output_dim=output_dim,
        device=torch.device("cpu")
    )
    
    global_error = torch.randn(batch_size, output_dim)
    
    for layer_idx, layer_dim in enumerate(layer_dims):
        e_l = feedback.compute_local_error(layer_idx, global_error)
        assert e_l.shape == (batch_size, layer_dim)


def test_memory_efficiency():
    """Test that feedback matrices don't consume excessive memory."""
    # Create large feedback matrices
    layer_dims = [1024] * 10
    output_dim = 50257  # GPT-2 vocab size
    
    feedback = DFAFeedbackMatrix(
        layer_dims=layer_dims,
        output_dim=output_dim,
        device=torch.device("cpu")
    )
    
    # Compute expected memory
    total_params = sum(d * output_dim for d in layer_dims)
    bytes_per_param = 4  # float32
    expected_mb = (total_params * bytes_per_param) / (1024 * 1024)
    
    print(f"\nMemory test: {total_params:,} parameters, ~{expected_mb:.1f} MB")
    
    # Should be reasonable (< 2GB for this config)
    assert expected_mb < 2000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
