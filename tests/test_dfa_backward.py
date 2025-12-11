"""
Unit tests for DFA Backward Hook
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.dfa_backward import DFABackwardHook, HybridDFAHook
from training.dfa_feedback import DFAFeedbackMatrix


class TestDFABackwardHook:
    """Test suite for DFABackwardHook."""
    
    @pytest.fixture
    def layer_dims(self):
        return [64, 128, 128, 64]
    
    @pytest.fixture
    def output_dim(self):
        return 256
    
    @pytest.fixture
    def feedback(self, layer_dims, output_dim):
        return DFAFeedbackMatrix(layer_dims, output_dim)
    
    @pytest.fixture
    def hook(self, feedback):
        return DFABackwardHook(feedback, enabled=True, store_statistics=True)
    
    @pytest.fixture
    def layers(self, layer_dims):
        """Create simple network layers."""
        layers = []
        layers.append(nn.Linear(64, layer_dims[0]))
        for i in range(1, len(layer_dims)):
            layers.append(nn.Linear(layer_dims[i-1], layer_dims[i]))
        return layers
    
    def test_initialization(self, hook, feedback):
        """Test hook initialization."""
        assert hook.feedback_matrix is feedback
        assert hook.enabled is True
        assert hook.store_statistics is True
        assert len(hook._activations) == 0
        assert len(hook._forward_hooks) == 0
        assert len(hook._backward_hooks) == 0
    
    def test_hook_registration(self, hook, layers, layer_dims):
        """Test registering hooks on modules."""
        hook.register_hooks(layers, list(range(len(layers))))
        
        assert len(hook._forward_hooks) == len(layers)
        assert len(hook._backward_hooks) == len(layers)
    
    def test_hook_removal(self, hook, layers):
        """Test removing hooks."""
        hook.register_hooks(layers, list(range(len(layers))))
        assert len(hook._forward_hooks) > 0
        
        hook.remove_hooks()
        assert len(hook._forward_hooks) == 0
        assert len(hook._backward_hooks) == 0
        assert len(hook._activations) == 0
    
    def test_activation_storage(self, hook, layers):
        """Test that activations are stored during forward pass."""
        hook.register_hooks(layers, list(range(len(layers))))
        
        # Forward pass
        x = torch.randn(4, 64)
        for layer in layers:
            x = layer(x)
        
        # Check activations stored
        assert len(hook._activations) == len(layers)
        for idx in range(len(layers)):
            assert idx in hook._activations
            assert hook._activations[idx].shape[0] == 4  # batch size
    
    def test_global_error_setting(self, hook):
        """Test setting and clearing global error."""
        global_error = torch.randn(4, 256)
        
        hook.set_global_error(global_error)
        assert hook._global_error is not None
        assert torch.equal(hook._global_error, global_error)
        
        hook.clear_global_error()
        assert hook._global_error is None
        assert len(hook._activations) == 0
    
    def test_dfa_gradient_computation(self, hook, layers, layer_dims, output_dim):
        """Test that DFA gradients are computed correctly."""
        hook.register_hooks(layers, list(range(len(layers))))
        
        # Forward pass
        batch_size = 4
        x = torch.randn(batch_size, 64)
        for layer in layers:
            x = layer(x)
        
        # Add final projection to output_dim
        final_proj = nn.Linear(layer_dims[-1], output_dim)
        output = final_proj(x)
        
        # Loss
        target = torch.randn_like(output)
        loss = ((output - target) ** 2).mean()
        
        # Set global error
        global_error = 2 * (output - target) / batch_size
        hook.set_global_error(global_error.detach())
        
        # Backward
        loss.backward()
        
        # Check that gradients exist
        for layer in layers:
            assert layer.weight.grad is not None
            assert not torch.isnan(layer.weight.grad).any()
    
    def test_statistics_tracking(self, hook, layers, layer_dims, output_dim):
        """Test that statistics are tracked."""
        hook.register_hooks(layers, list(range(len(layers))))
        
        # Forward + backward
        x = torch.randn(4, 64)
        for layer in layers:
            x = layer(x)
        
        final_proj = nn.Linear(layer_dims[-1], output_dim)
        output = final_proj(x)
        
        target = torch.randn_like(output)
        loss = ((output - target) ** 2).mean()
        
        global_error = 2 * (output - target) / 4
        hook.set_global_error(global_error.detach())
        
        loss.backward()
        
        # Get statistics
        stats = hook.get_statistics()
        
        # Should have grad norm and error norm for each layer
        for idx in range(len(layers)):
            assert f"layer_{idx}_grad_norm" in stats
            assert f"layer_{idx}_local_error_norm" in stats
    
    def test_enable_disable(self, hook, layers, layer_dims, output_dim):
        """Test enabling/disabling DFA."""
        hook.register_hooks(layers, list(range(len(layers))))
        
        # Test with DFA enabled
        hook.enable()
        assert hook.enabled is True
        
        # Test with DFA disabled
        hook.disable()
        assert hook.enabled is False
        
        # When disabled, should use standard BP
        x = torch.randn(4, 64)
        for layer in layers:
            x = layer(x)
        
        final_proj = nn.Linear(layer_dims[-1], output_dim)
        output = final_proj(x)
        
        loss = output.sum()
        hook.set_global_error(torch.ones_like(output))
        
        loss.backward()
        
        # Gradients should still exist (from standard BP)
        assert layers[0].weight.grad is not None
    
    def test_different_batch_sizes(self, hook, layers, layer_dims, output_dim):
        """Test with different batch sizes."""
        hook.register_hooks(layers, list(range(len(layers))))
        
        for batch_size in [1, 4, 16]:
            # Clear previous gradients
            for layer in layers:
                layer.zero_grad()
            hook.clear_global_error()
            
            # Forward
            x = torch.randn(batch_size, 64)
            for layer in layers:
                x = layer(x)
            
            final_proj = nn.Linear(layer_dims[-1], output_dim)
            output = final_proj(x)
            
            # Backward
            target = torch.randn_like(output)
            loss = ((output - target) ** 2).mean()
            
            global_error = 2 * (output - target) / batch_size
            hook.set_global_error(global_error.detach())
            
            loss.backward()
            
            # Check gradients
            assert layers[0].weight.grad is not None
            assert layers[0].weight.grad.shape == layers[0].weight.shape
    
    def test_repr(self, hook):
        """Test string representation."""
        repr_str = repr(hook)
        assert "DFABackwardHook" in repr_str
        assert "enabled" in repr_str


class TestHybridDFAHook:
    """Test suite for HybridDFAHook."""
    
    @pytest.fixture
    def layer_dims(self):
        return [64, 64, 128, 128]
    
    @pytest.fixture
    def output_dim(self):
        return 256
    
    @pytest.fixture
    def feedback(self, layer_dims, output_dim):
        return DFAFeedbackMatrix(layer_dims, output_dim)
    
    @pytest.fixture
    def block_boundaries(self):
        return [0, 2]  # DFA only at layers 0 and 2
    
    @pytest.fixture
    def hybrid_hook(self, feedback, block_boundaries):
        return HybridDFAHook(feedback, block_boundaries)
    
    def test_initialization(self, hybrid_hook, block_boundaries):
        """Test hybrid hook initialization."""
        assert hybrid_hook.block_boundaries == set(block_boundaries)
        assert hybrid_hook.enabled is True
    
    def test_selective_dfa_application(self, hybrid_hook, layer_dims):
        """Test that DFA only applies at block boundaries."""
        layers = [nn.Linear(layer_dims[i], layer_dims[i]) for i in range(len(layer_dims))]
        
        hybrid_hook.register_hooks(layers, list(range(len(layers))))
        
        # The hook should be registered for all layers
        assert len(hybrid_hook._forward_hooks) == len(layers)
        assert len(hybrid_hook._backward_hooks) == len(layers)
        
        # But DFA logic only applies at boundaries (tested via backward pass)


def test_gradient_flow():
    """Test that gradients flow correctly through DFA."""
    layer_dims = [32, 64]
    output_dim = 128
    
    feedback = DFAFeedbackMatrix(layer_dims, output_dim)
    hook = DFABackwardHook(feedback)
    
    layers = [nn.Linear(32, 32), nn.Linear(32, 64)]
    hook.register_hooks(layers, [0, 1])
    
    # Forward
    x = torch.randn(4, 32)
    for layer in layers:
        x = layer(x)
    
    final_proj = nn.Linear(64, output_dim)
    output = final_proj(x)
    
    # Backward with DFA
    loss = output.sum()
    global_error = torch.ones_like(output)
    hook.set_global_error(global_error)
    
    loss.backward()
    
    # All layers should have gradients
    for layer in layers:
        assert layer.weight.grad is not None
        assert not torch.isnan(layer.weight.grad).any()
        assert not torch.isinf(layer.weight.grad).any()


def test_3d_sequence_tensors():
    """Test DFA with 3D tensors (batch, sequence, features)."""
    layer_dims = [64, 128]
    output_dim = 256
    
    feedback = DFAFeedbackMatrix(layer_dims, output_dim)
    hook = DFABackwardHook(feedback)
    
    # Layers that preserve sequence dimension
    layers = [nn.Linear(64, 64), nn.Linear(64, 128)]
    hook.register_hooks(layers, [0, 1])
    
    # 3D input: [batch, seq, features]
    batch_size, seq_len = 4, 16
    x = torch.randn(batch_size, seq_len, 64)
    
    # Forward through layers (maintaining 3D shape)
    for layer in layers:
        x = layer(x)
    
    # Final projection
    final_proj = nn.Linear(128, output_dim)
    output = final_proj(x)
    
    # Loss over sequence
    target = torch.randn_like(output)
    loss = ((output - target) ** 2).mean()
    
    # Global error (3D)
    global_error = 2 * (output - target) / (batch_size * seq_len)
    hook.set_global_error(global_error.detach())
    
    # Backward
    loss.backward()
    
    # Check gradients
    assert layers[0].weight.grad is not None
    assert layers[1].weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
