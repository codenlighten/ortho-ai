"""
Unit tests for Orthogonality Loss
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.orthogonality_loss import OrthogonalityLoss, PerHeadOrthogonalityLoss


class TestOrthogonalityLoss:
    """Test suite for OrthogonalityLoss."""
    
    @pytest.fixture
    def loss_fn(self):
        return OrthogonalityLoss(
            lambda_max=0.01,
            warmup_steps=100,
            reduction="mean"
        )
    
    def test_initialization(self, loss_fn):
        """Test that loss function initializes correctly."""
        assert loss_fn.lambda_max == 0.01
        assert loss_fn.warmup_steps == 100
        assert loss_fn.reduction == "mean"
        assert loss_fn.step_count.item() == 0
    
    def test_lambda_warmup(self, loss_fn):
        """Test that lambda warmup works correctly."""
        # At step 0, lambda should be 0
        assert loss_fn.get_lambda(0) == 0.0
        
        # At warmup_steps/2, lambda should be lambda_max/2
        assert abs(loss_fn.get_lambda(50) - 0.005) < 1e-6
        
        # At warmup_steps, lambda should be lambda_max
        assert abs(loss_fn.get_lambda(100) - 0.01) < 1e-6
        
        # After warmup, lambda stays at lambda_max
        assert abs(loss_fn.get_lambda(200) - 0.01) < 1e-6
    
    def test_orthogonal_matrix_low_violation(self):
        """Test that orthogonal matrices have low violation."""
        loss_fn = OrthogonalityLoss(lambda_max=0.1, warmup_steps=1)
        
        # Create orthogonal matrix using QR decomposition
        W_ortho, _ = torch.linalg.qr(torch.randn(128, 128))
        
        loss = loss_fn([W_ortho], step=100)
        
        # Orthogonal matrix should have very low loss
        assert loss.item() < 1e-4
    
    def test_random_matrix_high_violation(self):
        """Test that random matrices have high violation."""
        loss_fn = OrthogonalityLoss(lambda_max=0.1, warmup_steps=1)
        
        # Random matrix (not orthogonal)
        W_random = torch.randn(128, 128)
        
        loss = loss_fn([W_random], step=100)
        
        # Random matrix should have significant loss
        assert loss.item() > 100
    
    def test_violation_computation(self, loss_fn):
        """Test that violation ||W^T W - I||²_F is computed correctly."""
        W = torch.randn(64, 64)
        
        # Manual computation
        gram = torch.matmul(W.T, W)
        identity = torch.eye(64)
        expected_violation = torch.norm(gram - identity, p="fro") ** 2
        
        # Using method
        computed_violation = loss_fn.compute_violation(W, name="test")
        
        assert abs(computed_violation.item() - expected_violation.item()) < 1e-3
    
    def test_multiple_weights(self, loss_fn):
        """Test loss computation with multiple weight matrices."""
        W1 = torch.randn(128, 128)
        W2 = torch.randn(64, 64)
        W3 = torch.randn(256, 128)
        
        loss = loss_fn([W1, W2, W3], step=50)
        
        # Loss should be positive
        assert loss.item() >= 0
        
        # Check statistics
        stats = loss_fn.get_statistics()
        assert stats["num_weights"] == 3
        assert "violation_weight_0" in stats
        assert "violation_weight_1" in stats
        assert "violation_weight_2" in stats
    
    def test_empty_weights(self, loss_fn):
        """Test that empty weight list returns zero loss."""
        loss = loss_fn([], step=0)
        assert loss.item() == 0.0
    
    def test_reduction_modes(self):
        """Test different reduction modes."""
        W1 = torch.randn(64, 64)
        W2 = torch.randn(64, 64)
        
        # Mean reduction
        loss_mean = OrthogonalityLoss(lambda_max=0.1, reduction="mean")
        l_mean = loss_mean([W1, W2], step=100)
        
        # Sum reduction
        loss_sum = OrthogonalityLoss(lambda_max=0.1, reduction="sum")
        l_sum = loss_sum([W1, W2], step=100)
        
        # Sum should be roughly 2x mean
        ratio = l_sum.item() / l_mean.item()
        assert 1.5 < ratio < 2.5  # Allow some tolerance
    
    def test_step_counter_increment(self, loss_fn):
        """Test that step counter increments after each forward pass."""
        W = torch.randn(64, 64)
        
        assert loss_fn.step_count.item() == 0
        
        loss_fn([W])
        assert loss_fn.step_count.item() == 1
        
        loss_fn([W])
        assert loss_fn.step_count.item() == 2
    
    def test_reset_step(self, loss_fn):
        """Test resetting step counter."""
        W = torch.randn(64, 64)
        
        # Run a few times
        for _ in range(5):
            loss_fn([W])
        
        assert loss_fn.step_count.item() == 5
        
        # Reset
        loss_fn.reset_step(10)
        assert loss_fn.step_count.item() == 10
    
    def test_gradient_flow(self, loss_fn):
        """Test that gradients flow through the loss."""
        W = torch.randn(64, 64, requires_grad=True)
        
        loss = loss_fn([W], step=100)
        loss.backward()
        
        # W should have gradients
        assert W.grad is not None
        assert not torch.isnan(W.grad).any()
    
    def test_statistics_tracking(self, loss_fn):
        """Test that statistics are tracked correctly."""
        W1 = torch.randn(64, 64)
        W2 = torch.randn(32, 32)
        
        loss = loss_fn([W1, W2], weight_names=["W1", "W2"], step=50)
        
        stats = loss_fn.get_statistics()
        
        assert "lambda" in stats
        assert "total_violation" in stats
        assert "num_weights" in stats
        assert "step" in stats
        assert "violation_W1" in stats
        assert "violation_W2" in stats
        
        # Lambda should be 0.5 * lambda_max at step 50
        assert abs(stats["lambda"] - 0.005) < 1e-6
    
    def test_repr(self, loss_fn):
        """Test string representation."""
        repr_str = repr(loss_fn)
        
        assert "OrthogonalityLoss" in repr_str
        assert "lambda_max=0.01" in repr_str
        assert "warmup_steps=100" in repr_str


class TestPerHeadOrthogonalityLoss:
    """Test suite for PerHeadOrthogonalityLoss."""
    
    @pytest.fixture
    def num_heads(self):
        return 8
    
    @pytest.fixture
    def loss_fn(self, num_heads):
        return PerHeadOrthogonalityLoss(
            num_heads=num_heads,
            lambda_max=0.01,
            warmup_steps=100
        )
    
    def test_initialization(self, loss_fn, num_heads):
        """Test per-head loss initialization."""
        assert loss_fn.num_heads == num_heads
        assert len(loss_fn._per_head_violations) == num_heads
    
    def test_per_head_violation_computation(self, loss_fn, num_heads):
        """Test that per-head violations are computed correctly."""
        d_head = 64
        d_in = 128
        d_out = num_heads * d_head  # 512
        
        W = torch.randn(d_out, d_in)
        
        violations = loss_fn.compute_per_head_violations(W, "test_layer")
        
        assert violations.shape == (num_heads,)
        assert not torch.isnan(violations).any()
        
        # Check that violations are tracked per head
        for head_idx in range(num_heads):
            assert len(loss_fn._per_head_violations[head_idx]) > 0
    
    def test_dimension_mismatch_error(self, loss_fn):
        """Test that dimension mismatch raises error."""
        # d_out not divisible by num_heads
        W_bad = torch.randn(100, 128)
        
        with pytest.raises(AssertionError):
            loss_fn.compute_per_head_violations(W_bad, "bad")
    
    def test_per_head_forward(self, loss_fn, num_heads):
        """Test forward pass with per-head tracking."""
        d_head = 64
        d_in = 128
        d_out = num_heads * d_head
        
        W = torch.randn(d_out, d_in)
        
        loss = loss_fn([W], weight_names=["mh_weight"], step=50, per_head=True)
        
        assert loss.item() >= 0
        
        # Check per-head statistics
        stats = loss_fn.get_per_head_statistics()
        
        for head_idx in range(num_heads):
            key = f"violation_mh_weight_head_{head_idx}"
            assert key in stats
    
    def test_non_per_head_mode(self, loss_fn, num_heads):
        """Test that per_head=False works."""
        d_head = 64
        W = torch.randn(num_heads * d_head, 128)
        
        # Should work without splitting into heads
        loss = loss_fn([W], step=50, per_head=False)
        assert loss.item() >= 0
    
    def test_multiple_multihead_weights(self, loss_fn, num_heads):
        """Test with multiple multi-head weight matrices."""
        d_head = 64
        d_in = 128
        d_out = num_heads * d_head
        
        W_Q = torch.randn(d_out, d_in)
        W_K = torch.randn(d_out, d_in)
        W_V = torch.randn(d_out, d_in)
        
        loss = loss_fn(
            [W_Q, W_K, W_V],
            weight_names=["Q", "K", "V"],
            step=50,
            per_head=True
        )
        
        assert loss.item() >= 0
        
        # Should track violations for all heads in all matrices
        stats = loss_fn.get_statistics()
        assert stats["num_weights"] == num_heads * 3  # 3 matrices × 8 heads


@pytest.mark.parametrize("lambda_max", [0.001, 0.01, 0.1])
def test_various_lambda_values(lambda_max):
    """Test with different lambda_max values."""
    loss_fn = OrthogonalityLoss(lambda_max=lambda_max, warmup_steps=1)
    
    W = torch.randn(64, 64)
    loss = loss_fn([W], step=100)
    
    # Loss should scale with lambda_max
    assert loss.item() > 0


@pytest.mark.parametrize("warmup_steps", [10, 100, 1000])
def test_various_warmup_steps(warmup_steps):
    """Test with different warmup durations."""
    loss_fn = OrthogonalityLoss(lambda_max=0.01, warmup_steps=warmup_steps)
    
    # At half warmup, lambda should be half max
    lambda_half = loss_fn.get_lambda(warmup_steps // 2)
    expected_half = 0.005
    
    assert abs(lambda_half - expected_half) < 0.001


@pytest.mark.parametrize("matrix_size", [32, 64, 128, 256])
def test_various_matrix_sizes(matrix_size):
    """Test with different matrix sizes."""
    loss_fn = OrthogonalityLoss(lambda_max=0.01, warmup_steps=1)
    
    W = torch.randn(matrix_size, matrix_size)
    loss = loss_fn([W], step=100)
    
    assert loss.item() > 0


def test_nearly_orthogonal_matrix():
    """Test with nearly orthogonal matrix."""
    loss_fn = OrthogonalityLoss(lambda_max=0.1, warmup_steps=1)
    
    # Start with orthogonal matrix
    W_ortho, _ = torch.linalg.qr(torch.randn(64, 64))
    
    # Add small perturbation
    noise = torch.randn(64, 64) * 0.01
    W_nearly = W_ortho + noise
    
    loss = loss_fn([W_nearly], step=100)
    
    # Loss should be small but non-zero
    assert 0 < loss.item() < 10


def test_device_compatibility():
    """Test that loss works on different devices."""
    loss_fn = OrthogonalityLoss(lambda_max=0.01, warmup_steps=100)
    
    W_cpu = torch.randn(64, 64)
    loss_cpu = loss_fn([W_cpu], step=50)
    
    assert loss_cpu.device.type == "cpu"
    
    if torch.cuda.is_available():
        W_cuda = torch.randn(64, 64, device="cuda")
        loss_cuda = loss_fn([W_cuda], step=50)
        
        assert loss_cuda.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
