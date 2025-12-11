"""
Unit tests for DFA Transformer Block
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.dfa_transformer_block import DFATransformerBlock, FeedForward


class TestDFATransformerBlock:
    """Test suite for DFATransformerBlock."""
    
    @pytest.fixture
    def d_model(self):
        return 256
    
    @pytest.fixture
    def num_heads(self):
        return 8
    
    @pytest.fixture
    def block(self, d_model, num_heads):
        return DFATransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            causal=True,
            orthogonal_init=True
        )
    
    def test_initialization(self, block, d_model, num_heads):
        """Test block initialization."""
        assert block.d_model == d_model
        assert block.num_heads == num_heads
        assert block.d_ff == 4 * d_model  # Default
    
    def test_output_shape(self, block):
        """Test that output shape matches input."""
        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, block.d_model)
        
        output, _ = block(x)
        
        assert output.shape == x.shape
    
    def test_attention_weights(self, block, num_heads):
        """Test attention weights shape."""
        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, block.d_model)
        
        output, attn_weights = block(x, return_attention_weights=True)
        
        assert attn_weights is not None
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    def test_gradient_flow(self, block):
        """Test gradient flow through all components."""
        x = torch.randn(2, 16, block.d_model, requires_grad=True)
        
        output, _ = block(x)
        loss = output.sum()
        loss.backward()
        
        # Input should have gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # All modules should have gradients
        for module in block.get_all_linear_modules():
            assert module.weight.grad is not None
            assert not torch.isnan(module.weight.grad).any()
    
    def test_module_collection(self, block):
        """Test collecting modules for DFA."""
        attn_modules = block.get_attention_modules()
        ff_modules = block.get_feedforward_modules()
        all_modules = block.get_all_linear_modules()
        
        assert len(attn_modules) == 4  # Q, K, V, out
        assert len(ff_modules) == 2    # fc1, fc2
        assert len(all_modules) == 6   # Total
    
    def test_orthogonality_weights(self, block):
        """Test collecting weights for orthogonality loss."""
        weights = block.get_orthogonality_weights()
        
        assert len(weights) == 6  # 4 attention + 2 ff
        
        for name, weight in weights:
            assert isinstance(name, str)
            assert isinstance(weight, torch.Tensor)
            assert weight.dim() == 2
    
    def test_statistics(self, block):
        """Test statistics computation."""
        x = torch.randn(2, 16, block.d_model)
        _ = block(x)
        
        stats = block.get_statistics()
        
        # Should have both attention and feedforward stats
        assert any(k.startswith("attn_") for k in stats.keys())
        assert any(k.startswith("ff_") for k in stats.keys())
    
    def test_causal_mode(self):
        """Test causal (autoregressive) mode."""
        block = DFATransformerBlock(d_model=128, num_heads=4, causal=True)
        
        x = torch.randn(2, 16, 128)
        output, _ = block(x)
        
        assert output.shape == x.shape
    
    def test_non_causal_mode(self):
        """Test non-causal (bidirectional) mode."""
        block = DFATransformerBlock(d_model=128, num_heads=4, causal=False)
        
        x = torch.randn(2, 16, 128)
        output, _ = block(x)
        
        assert output.shape == x.shape
    
    def test_orthogonal_initialization(self):
        """Test orthogonal initialization produces low violations."""
        block = DFATransformerBlock(
            d_model=256,
            num_heads=8,
            orthogonal_init=True
        )
        
        x = torch.randn(2, 16, 256)
        _ = block(x)
        
        stats = block.get_statistics()
        
        # Attention projections should be perfectly orthogonal
        assert stats["attn_total_violation"] < 1e-4
    
    def test_standard_initialization(self):
        """Test standard (non-orthogonal) initialization."""
        block = DFATransformerBlock(
            d_model=256,
            num_heads=8,
            orthogonal_init=False
        )
        
        x = torch.randn(2, 16, 256)
        _ = block(x)
        
        stats = block.get_statistics()
        
        # Should have some orthogonality violations
        assert stats["attn_total_violation"] > 0
    
    def test_different_d_ff(self):
        """Test with different feedforward dimensions."""
        for d_ff in [512, 1024, 2048]:
            block = DFATransformerBlock(d_model=256, num_heads=8, d_ff=d_ff)
            
            x = torch.randn(2, 16, 256)
            output, _ = block(x)
            
            assert output.shape == x.shape
            assert block.d_ff == d_ff
    
    def test_different_sequence_lengths(self, block):
        """Test with various sequence lengths."""
        for seq_len in [8, 32, 128]:
            x = torch.randn(2, seq_len, block.d_model)
            output, _ = block(x)
            
            assert output.shape == x.shape
    
    def test_different_batch_sizes(self, block):
        """Test with various batch sizes."""
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 16, block.d_model)
            output, _ = block(x)
            
            assert output.shape == x.shape
    
    def test_repr(self, block):
        """Test string representation."""
        repr_str = repr(block)
        
        assert "DFATransformerBlock" in repr_str
        assert str(block.d_model) in repr_str
        assert str(block.num_heads) in repr_str


class TestFeedForward:
    """Test suite for FeedForward network."""
    
    @pytest.fixture
    def ff(self):
        return FeedForward(d_model=256, d_ff=1024, activation="gelu")
    
    def test_initialization(self, ff):
        """Test feedforward initialization."""
        assert ff.d_model == 256
        assert ff.d_ff == 1024
    
    def test_output_shape(self, ff):
        """Test output shape matches input."""
        x = torch.randn(4, 32, 256)
        output = ff(x)
        
        assert output.shape == x.shape
    
    def test_gradient_flow(self, ff):
        """Test gradient flow."""
        x = torch.randn(2, 16, 256, requires_grad=True)
        output = ff(x)
        
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert ff.fc1.weight.grad is not None
        assert ff.fc2.weight.grad is not None
    
    def test_activations(self):
        """Test different activation functions."""
        for activation in ["relu", "gelu", "swish", "silu"]:
            ff = FeedForward(d_model=128, d_ff=512, activation=activation)
            
            x = torch.randn(2, 16, 128)
            output = ff(x)
            
            assert output.shape == x.shape
    
    def test_invalid_activation(self):
        """Test that invalid activation raises error."""
        with pytest.raises(ValueError):
            FeedForward(d_model=128, d_ff=512, activation="invalid")
    
    def test_orthogonal_init(self):
        """Test orthogonal initialization."""
        ff = FeedForward(d_model=256, d_ff=1024, orthogonal_init=True)
        
        x = torch.randn(2, 16, 256)
        _ = ff(x)
        
        stats = ff.get_statistics()
        
        # fc1 should be perfectly orthogonal (square matrix)
        assert stats["ortho_violation_fc1"] < 1e-4
    
    def test_standard_init(self):
        """Test standard initialization."""
        ff = FeedForward(d_model=256, d_ff=1024, orthogonal_init=False)
        
        x = torch.randn(2, 16, 256)
        _ = ff(x)
        
        stats = ff.get_statistics()
        
        # Should have violations
        assert stats["ortho_violation_fc1"] > 0
    
    def test_statistics(self, ff):
        """Test statistics computation."""
        x = torch.randn(2, 16, 256)
        _ = ff(x)
        
        stats = ff.get_statistics()
        
        assert "ortho_violation_fc1" in stats
        assert "ortho_violation_fc2" in stats


@pytest.mark.parametrize("d_model,num_heads,d_ff", [
    (128, 4, 512),
    (256, 8, 1024),
    (512, 16, 2048),
])
def test_various_configurations(d_model, num_heads, d_ff):
    """Test with different model configurations."""
    block = DFATransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff
    )
    
    x = torch.randn(2, 16, d_model)
    output, _ = block(x)
    
    assert output.shape == x.shape


def test_parameter_count():
    """Test parameter count calculation."""
    d_model = 256
    d_ff = 1024
    
    block = DFATransformerBlock(d_model=d_model, num_heads=8, d_ff=d_ff)
    
    num_params = sum(p.numel() for p in block.parameters())
    
    # Q, K, V, out: 4 * (d_model² + d_model)
    # fc1: d_ff * d_model + d_ff
    # fc2: d_model * d_ff + d_model
    # LayerNorms: 2 * (d_model + d_model)
    expected = (
        4 * (d_model * d_model + d_model) +  # Attention
        (d_ff * d_model + d_ff) +             # fc1
        (d_model * d_ff + d_model) +          # fc2
        2 * (d_model + d_model)                # LayerNorms
    )
    
    assert num_params == expected


def test_dropout_effect():
    """Test that dropout has effect in training mode."""
    block = DFATransformerBlock(d_model=128, num_heads=4, dropout=0.5)
    block.train()
    
    x = torch.randn(2, 16, 128)
    
    output1, _ = block(x)
    output2, _ = block(x)
    
    # Outputs should differ due to dropout
    diff = (output1 - output2).norm() / output1.norm()
    assert diff > 0.01


def test_eval_mode_deterministic():
    """Test that eval mode is deterministic."""
    block = DFATransformerBlock(d_model=128, num_heads=4, dropout=0.5)
    block.eval()
    
    x = torch.randn(2, 16, 128)
    
    # Redraw random features to ensure consistency
    block.attention.redraw_projection_matrix()
    
    with torch.no_grad():
        output1, _ = block(x)
        output2, _ = block(x)
    
    # Should be identical in eval mode
    assert torch.allclose(output1, output2, rtol=1e-5)


def test_residual_connections():
    """Test that residual connections work correctly."""
    block = DFATransformerBlock(d_model=128, num_heads=4, dropout=0.0)
    
    x = torch.randn(2, 16, 128)
    output, _ = block(x)
    
    # Output should not be identical to input (transformation applied)
    assert not torch.allclose(output, x, rtol=0.1)
    
    # But should have some relationship (residual preserved)
    # Check that output contains some component of input
    correlation = torch.nn.functional.cosine_similarity(
        output.flatten(), 
        x.flatten(), 
        dim=0
    )
    assert correlation > 0  # Some positive correlation


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
