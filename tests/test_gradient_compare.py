"""
Tests for gradient comparison and diagnostic tools.

Copyright (c) 2025 Gregory Ward - SmartLedger.Technology
Licensed under the MIT License (see LICENSE file)
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diagnostics.gradient_compare import (
    GradientMetrics,
    GradientComparator,
    AttentionComparator,
    DiagnosticLogger,
    quick_diagnostic,
)


class TestGradientMetrics:
    """Test GradientMetrics dataclass."""
    
    def test_initialization(self):
        """Test basic initialization."""
        metrics = GradientMetrics()
        assert metrics.grad_error == 0.0
        assert metrics.grad_cosine == 0.0
        assert metrics.num_layers == 0
        assert len(metrics.layer_metrics) == 0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = GradientMetrics(
            grad_error=0.5,
            grad_cosine=0.8,
            num_layers=3,
        )
        metrics.layer_metrics['layer1'] = {'error': 0.3, 'cosine': 0.9}
        metrics.ortho_violations['attn'] = 0.001
        
        result = metrics.to_dict()
        assert result['grad_error'] == 0.5
        assert result['grad_cosine'] == 0.8
        assert result['num_layers'] == 3
        assert 'layer_layer1' in result
        assert 'ortho_attn' in result


class TestGradientComparator:
    """Test GradientComparator class."""
    
    def test_initialization(self):
        """Test comparator initialization."""
        model = nn.Linear(10, 10)
        comparator = GradientComparator(model)
        
        assert comparator.model is model
        assert comparator.eps == 1e-8
        assert comparator.track_per_layer is True
        assert len(comparator.history) == 0
    
    def test_identical_gradients(self):
        """Test comparison with identical gradients."""
        comparator = GradientComparator(None, eps=1e-8)
        
        grads = {
            'layer1.weight': torch.randn(128, 256),
            'layer2.weight': torch.randn(256, 512),
        }
        
        metrics = comparator.compare_gradients(grads, grads)
        
        # Should have perfect alignment (allow for small numerical error)
        assert metrics.grad_error < 1e-5
        assert abs(metrics.grad_cosine - 1.0) < 1e-5
        assert metrics.dfa_norm == metrics.bp_norm
        assert metrics.num_layers == 2
    
    def test_different_gradients(self):
        """Test comparison with different gradients."""
        comparator = GradientComparator(None)
        
        dfa_grads = {
            'layer1.weight': torch.randn(128, 256),
        }
        bp_grads = {
            'layer1.weight': torch.randn(128, 256),
        }
        
        metrics = comparator.compare_gradients(dfa_grads, bp_grads)
        
        # Should be different
        assert metrics.grad_error > 0
        assert metrics.grad_cosine < 1.0
        assert metrics.num_layers == 1
    
    def test_opposite_gradients(self):
        """Test comparison with opposite gradients."""
        comparator = GradientComparator(None)
        
        base_grad = torch.randn(64, 128)
        dfa_grads = {'layer.weight': base_grad}
        bp_grads = {'layer.weight': -base_grad}
        
        metrics = comparator.compare_gradients(dfa_grads, bp_grads)
        
        # Should have negative cosine similarity
        assert metrics.grad_cosine < 0
        assert metrics.grad_cosine > -1.1  # Allow small numerical error
    
    def test_per_layer_metrics(self):
        """Test per-layer metric tracking."""
        comparator = GradientComparator(None, track_per_layer=True)
        
        dfa_grads = {
            'layer1.weight': torch.randn(32, 64),
            'layer2.weight': torch.randn(64, 128),
        }
        bp_grads = {
            'layer1.weight': dfa_grads['layer1.weight'] + 0.1 * torch.randn(32, 64),
            'layer2.weight': dfa_grads['layer2.weight'] + 0.1 * torch.randn(64, 128),
        }
        
        metrics = comparator.compare_gradients(dfa_grads, bp_grads)
        
        assert 'layer1.weight' in metrics.layer_metrics
        assert 'layer2.weight' in metrics.layer_metrics
        assert 'error' in metrics.layer_metrics['layer1.weight']
        assert 'cosine' in metrics.layer_metrics['layer1.weight']
    
    def test_orthogonality_violations(self):
        """Test orthogonality violation tracking."""
        comparator = GradientComparator(None)
        
        grads = {'layer.weight': torch.randn(32, 64)}
        ortho_violations = {
            'layer1': 0.001,
            'layer2': 0.002,
            'layer3': 0.0015,
        }
        
        metrics = comparator.compare_gradients(grads, grads, ortho_violations)
        
        assert metrics.ortho_violations == ortho_violations
        assert abs(metrics.avg_ortho_violation - 0.0015) < 1e-6
    
    def test_history_tracking(self):
        """Test metric history tracking."""
        comparator = GradientComparator(None)
        
        grads1 = {'layer.weight': torch.randn(32, 64)}
        grads2 = {'layer.weight': torch.randn(32, 64)}
        
        # Add multiple comparisons
        for _ in range(5):
            comparator.compare_gradients(grads1, grads2)
        
        assert len(comparator.history) == 5
    
    def test_get_statistics(self):
        """Test statistics computation."""
        comparator = GradientComparator(None)
        
        # Add some metrics
        for i in range(10):
            grads = {'layer.weight': torch.randn(32, 64)}
            comparator.compare_gradients(grads, grads)
        
        stats = comparator.get_statistics(window=5)
        
        assert 'avg_grad_error' in stats
        assert 'std_grad_error' in stats
        assert 'avg_grad_cosine' in stats
        assert stats['num_samples'] == 5
    
    def test_reset_history(self):
        """Test history reset."""
        comparator = GradientComparator(None)
        
        grads = {'layer.weight': torch.randn(32, 64)}
        comparator.compare_gradients(grads, grads)
        
        assert len(comparator.history) > 0
        
        comparator.reset_history()
        assert len(comparator.history) == 0
    
    def test_empty_gradients(self):
        """Test with empty gradient dictionaries."""
        comparator = GradientComparator(None)
        
        metrics = comparator.compare_gradients({}, {})
        
        assert metrics.grad_error == 0.0
        assert metrics.num_layers == 0
    
    def test_mismatched_keys(self):
        """Test with mismatched gradient keys."""
        comparator = GradientComparator(None)
        
        dfa_grads = {'layer1.weight': torch.randn(32, 64)}
        bp_grads = {'layer2.weight': torch.randn(32, 64)}
        
        metrics = comparator.compare_gradients(dfa_grads, bp_grads)
        
        # No common keys
        assert metrics.num_layers == 0


class TestAttentionComparator:
    """Test AttentionComparator class."""
    
    def test_initialization(self):
        """Test comparator initialization."""
        comparator = AttentionComparator()
        assert comparator.eps == 1e-8
        assert len(comparator.history) == 0
    
    def test_softmax_attention_shape(self):
        """Test softmax attention output shape."""
        comparator = AttentionComparator()
        
        batch, heads, seq_len, d_k = 2, 4, 16, 32
        Q = torch.randn(batch, heads, seq_len, d_k)
        K = torch.randn(batch, heads, seq_len, d_k)
        V = torch.randn(batch, heads, seq_len, d_k)
        
        output = comparator.compute_softmax_attention(Q, K, V)
        
        assert output.shape == (batch, heads, seq_len, d_k)
    
    def test_softmax_attention_with_mask(self):
        """Test softmax attention with causal mask."""
        comparator = AttentionComparator()
        
        batch, heads, seq_len, d_k = 2, 4, 8, 32
        Q = torch.randn(batch, heads, seq_len, d_k)
        K = torch.randn(batch, heads, seq_len, d_k)
        V = torch.randn(batch, heads, seq_len, d_k)
        
        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        
        output = comparator.compute_softmax_attention(Q, K, V, mask)
        
        assert output.shape == (batch, heads, seq_len, d_k)
    
    def test_compute_similarity_identical(self):
        """Test similarity with identical outputs."""
        comparator = AttentionComparator()
        
        output = torch.randn(2, 4, 16, 32)
        similarity = comparator.compute_similarity(output, output)
        
        assert abs(similarity - 1.0) < 1e-6
    
    def test_compute_similarity_different(self):
        """Test similarity with different outputs."""
        comparator = AttentionComparator()
        
        output1 = torch.randn(2, 4, 16, 32)
        output2 = torch.randn(2, 4, 16, 32)
        
        similarity = comparator.compute_similarity(output1, output2)
        
        assert -1.0 <= similarity <= 1.0
    
    def test_compare_attention(self):
        """Test full attention comparison."""
        comparator = AttentionComparator()
        
        batch, heads, seq_len, d_k = 2, 4, 16, 32
        Q = torch.randn(batch, heads, seq_len, d_k)
        K = torch.randn(batch, heads, seq_len, d_k)
        V = torch.randn(batch, heads, seq_len, d_k)
        
        # Compute softmax first to get realistic favor output
        softmax_out = comparator.compute_softmax_attention(Q, K, V)
        favor_out = softmax_out + 0.01 * torch.randn_like(softmax_out)
        
        result = comparator.compare_attention(Q, K, V, favor_out)
        
        assert 'attn_sim' in result
        assert 'attn_error' in result
        assert 'favor_norm' in result
        assert 'softmax_norm' in result
        assert result['attn_sim'] > 0.9  # Should be very similar
    
    def test_history_tracking(self):
        """Test similarity history tracking."""
        comparator = AttentionComparator()
        
        output = torch.randn(2, 4, 16, 32)
        
        for _ in range(5):
            comparator.compute_similarity(output, output)
        
        assert len(comparator.history) == 5
    
    def test_get_avg_similarity(self):
        """Test average similarity computation."""
        comparator = AttentionComparator()
        
        output = torch.randn(2, 4, 16, 32)
        
        # Add several similarity scores
        for _ in range(10):
            comparator.compute_similarity(output, output)
        
        avg = comparator.get_avg_similarity(window=5)
        assert abs(avg - 1.0) < 1e-5  # Should be close to 1.0
    
    def test_get_avg_similarity_empty(self):
        """Test average similarity with no history."""
        comparator = AttentionComparator()
        avg = comparator.get_avg_similarity()
        assert avg == 0.0


class TestDiagnosticLogger:
    """Test DiagnosticLogger class."""
    
    def test_initialization_console_only(self):
        """Test logger initialization with console only."""
        logger = DiagnosticLogger(console_log=True)
        assert logger.console_log is True
        assert logger.use_wandb is False
        assert logger.use_tensorboard is False
        assert logger.step == 0
    
    def test_log_metrics_console(self, capsys):
        """Test logging metrics to console."""
        logger = DiagnosticLogger(console_log=True)
        
        metrics = {'loss': 0.5, 'accuracy': 0.95}
        logger.log_metrics(metrics, step=0)
        
        captured = capsys.readouterr()
        assert 'loss' in captured.out
        assert '0.5' in captured.out
    
    def test_log_gradient_metrics(self):
        """Test logging GradientMetrics."""
        logger = DiagnosticLogger(console_log=False)
        
        metrics = GradientMetrics(grad_error=0.5, grad_cosine=0.8)
        logger.log_gradient_metrics(metrics, step=0)
        
        assert logger.step == 0  # Step shouldn't increment when provided
    
    def test_step_increment(self):
        """Test automatic step increment."""
        logger = DiagnosticLogger(console_log=False)
        
        for i in range(5):
            logger.log_metrics({'value': i})
        
        assert logger.step == 5
    
    def test_close(self):
        """Test logger cleanup."""
        logger = DiagnosticLogger(console_log=False)
        logger.close()  # Should not raise


class TestQuickDiagnostic:
    """Test quick_diagnostic function."""
    
    def test_quick_diagnostic_output(self, capsys):
        """Test quick diagnostic output."""
        dfa_grads = {'layer.weight': torch.randn(32, 64)}
        bp_grads = {'layer.weight': torch.randn(32, 64)}
        ortho_violations = {'attn': 0.001}
        
        quick_diagnostic(None, dfa_grads, bp_grads, ortho_violations)
        
        captured = capsys.readouterr()
        assert 'DIAGNOSTIC REPORT' in captured.out
        assert 'Gradient Error' in captured.out
        assert 'Ortho Violation' in captured.out


@pytest.mark.parametrize("batch_size,seq_len", [
    (1, 8),
    (2, 16),
    (4, 32),
])
def test_attention_comparator_various_sizes(batch_size, seq_len):
    """Test attention comparator with various input sizes."""
    comparator = AttentionComparator()
    
    heads, d_k = 4, 32
    Q = torch.randn(batch_size, heads, seq_len, d_k)
    K = torch.randn(batch_size, heads, seq_len, d_k)
    V = torch.randn(batch_size, heads, seq_len, d_k)
    
    output = comparator.compute_softmax_attention(Q, K, V)
    assert output.shape == (batch_size, heads, seq_len, d_k)


@pytest.mark.parametrize("num_layers", [1, 3, 5])
def test_gradient_comparator_multiple_layers(num_layers):
    """Test gradient comparator with multiple layers."""
    comparator = GradientComparator(None, track_per_layer=True)
    
    dfa_grads = {f'layer{i}.weight': torch.randn(32, 64) for i in range(num_layers)}
    bp_grads = {f'layer{i}.weight': torch.randn(32, 64) for i in range(num_layers)}
    
    metrics = comparator.compare_gradients(dfa_grads, bp_grads)
    
    assert metrics.num_layers == num_layers
    assert len(metrics.layer_metrics) == num_layers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
