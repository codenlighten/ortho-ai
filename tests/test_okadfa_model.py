"""
Tests for OKADFA full model.

Copyright (c) 2025 Gregory Ward - SmartLedger.Technology
Licensed under the MIT License (see LICENSE file)
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.okadfa_model import (
    PositionalEncoding,
    OKADFAModel,
    create_gpt2_small_okadfa,
    create_gpt2_medium_okadfa,
)
from src.training.orthogonality_loss import OrthogonalityLoss


class TestPositionalEncoding:
    """Test positional encoding module."""
    
    def test_initialization(self):
        """Test positional encoding initialization."""
        pe = PositionalEncoding(d_model=512, max_len=1000)
        assert pe.pe.shape == (1, 1000, 512)
    
    def test_forward_pass(self):
        """Test positional encoding forward pass."""
        pe = PositionalEncoding(d_model=256, max_len=100, dropout=0.0)
        x = torch.randn(2, 50, 256)
        
        output = pe(x)
        assert output.shape == x.shape
    
    def test_different_sequence_lengths(self):
        """Test with various sequence lengths."""
        pe = PositionalEncoding(d_model=128, max_len=1000)
        
        for seq_len in [10, 50, 100, 500]:
            x = torch.randn(2, seq_len, 128)
            output = pe(x)
            assert output.shape == (2, seq_len, 128)


class TestOKADFAModel:
    """Test OKADFAModel class."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = OKADFAModel(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=4,
            d_ff=1024,
        )
        
        assert model.vocab_size == 1000
        assert model.d_model == 256
        assert model.num_layers == 4
        assert len(model.layers) == 4
    
    def test_parameter_count(self):
        """Test parameter counting."""
        model = OKADFAModel(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=2,
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count == model.num_parameters
        assert param_count > 0
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = OKADFAModel(
            vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_layers=2,
        )
        
        input_ids = torch.randint(0, 1000, (2, 32))
        logits = model(input_ids)
        
        assert logits.shape == (2, 32, 1000)
    
    def test_return_hidden_states(self):
        """Test returning hidden states."""
        model = OKADFAModel(
            vocab_size=500,
            d_model=128,
            num_heads=4,
            num_layers=3,
        )
        
        input_ids = torch.randint(0, 500, (2, 16))
        logits, hidden_states = model(input_ids, return_hidden_states=True)
        
        assert len(hidden_states) == 3
        for h in hidden_states:
            assert h.shape == (2, 16, 128)
    
    def test_causal_attention(self):
        """Test with causal attention enabled."""
        model = OKADFAModel(
            vocab_size=500,
            d_model=128,
            num_heads=4,
            num_layers=2,
            use_causal=True,
        )
        
        input_ids = torch.randint(0, 500, (2, 20))
        logits = model(input_ids)
        
        assert logits.shape == (2, 20, 500)
    
    def test_non_causal_attention(self):
        """Test with non-causal attention."""
        model = OKADFAModel(
            vocab_size=500,
            d_model=128,
            num_heads=4,
            num_layers=2,
            use_causal=False,
        )
        
        input_ids = torch.randint(0, 500, (2, 20))
        logits = model(input_ids)
        
        assert logits.shape == (2, 20, 500)
    
    def test_weight_tying(self):
        """Test tied input/output embeddings."""
        model = OKADFAModel(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=2,
            tie_weights=True,
        )
        
        assert model.output_projection.weight is model.token_embedding.weight
    
    def test_no_weight_tying(self):
        """Test separate input/output embeddings."""
        model = OKADFAModel(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=2,
            tie_weights=False,
        )
        
        assert model.output_projection.weight is not model.token_embedding.weight
    
    def test_get_dfa_modules(self):
        """Test DFA module collection."""
        model = OKADFAModel(
            vocab_size=500,
            d_model=128,
            num_heads=4,
            num_layers=3,
        )
        
        modules = model.get_dfa_modules()
        
        # Should have 6 modules per layer (4 attn + 2 ff) * 3 layers
        assert len(modules) == 18
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        model = OKADFAModel(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=4,
        )
        
        stats = model.get_statistics()
        
        assert 'num_parameters' in stats
        assert 'num_layers' in stats
        assert stats['num_layers'] == 4
        assert 'd_model' in stats
        assert 'layers' in stats
        assert len(stats['layers']) == 4
    
    def test_estimate_complexity(self):
        """Test complexity estimation."""
        model = OKADFAModel(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=4,
        )
        
        complexity = model.estimate_complexity(seq_len=512, batch_size=2)
        
        assert 'favor_ops' in complexity
        assert 'standard_ops' in complexity
        assert 'speedup' in complexity
        assert complexity['speedup'] > 0
        assert complexity['total_okadfa_ops'] < complexity['total_standard_ops']
    
    def test_orthogonality_loss(self):
        """Test orthogonality loss computation."""
        model = OKADFAModel(
            vocab_size=500,
            d_model=128,
            num_heads=4,
            num_layers=2,
            orthogonal_init=True,
        )
        
        loss_fn = OrthogonalityLoss(lambda_max=0.01)
        loss = model.get_orthogonality_loss(loss_fn)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_gradient_flow(self):
        """Test gradient flow through model."""
        model = OKADFAModel(
            vocab_size=500,
            d_model=128,
            num_heads=4,
            num_layers=2,
        )
        model.train()
        
        input_ids = torch.randint(0, 500, (2, 16))
        target_ids = torch.randint(0, 500, (2, 16))
        
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, 500),
            target_ids.view(-1)
        )
        loss.backward()
        
        # Check that gradients exist
        has_grads = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for _ in model.parameters())
        
        assert has_grads == total_params


class TestGPT2Configs:
    """Test GPT-2 configuration functions."""
    
    def test_gpt2_small(self):
        """Test GPT-2 Small configuration."""
        model = create_gpt2_small_okadfa()
        
        assert model.d_model == 768
        assert model.num_heads == 12
        assert model.num_layers == 12
        assert model.d_ff == 3072
        assert model.vocab_size == 50257
        
        # Check parameter count (~124M)
        assert 120_000_000 < model.num_parameters < 130_000_000
    
    def test_gpt2_medium(self):
        """Test GPT-2 Medium configuration."""
        model = create_gpt2_medium_okadfa()
        
        assert model.d_model == 1024
        assert model.num_heads == 16
        assert model.num_layers == 24
        assert model.d_ff == 4096
        
        # Check parameter count (~350M)
        assert 340_000_000 < model.num_parameters < 360_000_000
    
    def test_gpt2_small_forward(self):
        """Test GPT-2 Small forward pass."""
        model = create_gpt2_small_okadfa()
        
        input_ids = torch.randint(0, 50257, (1, 64))
        logits = model(input_ids)
        
        assert logits.shape == (1, 64, 50257)


@pytest.mark.parametrize("batch_size,seq_len", [
    (1, 16),
    (2, 32),
    (4, 64),
])
def test_various_batch_sizes(batch_size, seq_len):
    """Test model with various batch sizes and sequence lengths."""
    model = OKADFAModel(
        vocab_size=1000,
        d_model=128,
        num_heads=4,
        num_layers=2,
    )
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    logits = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, 1000)


@pytest.mark.parametrize("num_layers", [1, 2, 4, 8])
def test_various_depths(num_layers):
    """Test model with various depths."""
    model = OKADFAModel(
        vocab_size=500,
        d_model=128,
        num_heads=4,
        num_layers=num_layers,
    )
    
    input_ids = torch.randint(0, 500, (2, 16))
    logits = model(input_ids)
    
    assert logits.shape == (2, 16, 500)
    assert len(model.layers) == num_layers


@pytest.mark.parametrize("d_model,num_heads", [
    (128, 4),
    (256, 8),
    (512, 16),
])
def test_various_model_dims(d_model, num_heads):
    """Test model with various dimensions."""
    model = OKADFAModel(
        vocab_size=500,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=2,
    )
    
    input_ids = torch.randint(0, 500, (2, 16))
    logits = model(input_ids)
    
    assert logits.shape == (2, 16, 500)


def test_eval_mode():
    """Test model in evaluation mode."""
    model = OKADFAModel(
        vocab_size=500,
        d_model=128,
        num_heads=4,
        num_layers=2,
    )
    model.eval()
    
    input_ids = torch.randint(0, 500, (2, 16))
    
    with torch.no_grad():
        logits1 = model(input_ids)
        logits2 = model(input_ids)
    
    # Should be deterministic in eval mode
    assert torch.allclose(logits1, logits2)


def test_train_mode():
    """Test model in training mode."""
    model = OKADFAModel(
        vocab_size=500,
        d_model=128,
        num_heads=4,
        num_layers=2,
        dropout=0.5,  # High dropout to see effect
    )
    model.train()
    
    input_ids = torch.randint(0, 500, (2, 16))
    
    logits1 = model(input_ids)
    logits2 = model(input_ids)
    
    # Should be different due to dropout
    assert not torch.allclose(logits1, logits2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
