"""
Tests for data loading utilities.

Copyright (c) 2025 Gregory Ward - SmartLedger.Technology
Licensed under the MIT License. See LICENSE file in the project root.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
import torch
import tempfile
from data.tokenizer import get_simple_tokenizer, SimpleCharTokenizer
from data.text_dataset import SimpleTextDataset, TextDataset
from data.wikitext_loader import WikiTextDataset


class TestSimpleCharTokenizer:
    """Test simple character tokenizer."""
    
    def test_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = SimpleCharTokenizer(vocab_size=256)
        
        assert tokenizer.vocab_size == 256
        assert tokenizer.pad_token_id == 0
        assert tokenizer.eos_token_id == 1
        assert tokenizer.bos_token_id == 2
    
    def test_encode_decode(self):
        """Test encoding and decoding."""
        tokenizer = SimpleCharTokenizer()
        text = "Hello, World!"
        
        # Encode
        tokens = tokenizer.encode(text, add_special_tokens=False)
        assert len(tokens) == len(text)
        assert all(0 <= t < 256 for t in tokens)
        
        # Decode
        decoded = tokenizer.decode(tokens, skip_special_tokens=False)
        assert decoded == text
    
    def test_special_tokens(self):
        """Test special token handling."""
        tokenizer = SimpleCharTokenizer()
        text = "Hi"
        
        # With special tokens
        tokens = tokenizer.encode(text, add_special_tokens=True)
        assert tokens[0] == tokenizer.bos_token_id
        assert tokens[-1] == tokenizer.eos_token_id
        assert len(tokens) == len(text) + 2
        
        # Decode with skip
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        assert decoded == text
    
    def test_call_interface(self):
        """Test __call__ interface."""
        tokenizer = SimpleCharTokenizer()
        text = "Test"
        
        result = tokenizer(text)
        
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert len(result['input_ids']) == len(result['attention_mask'])
    
    def test_get_simple_tokenizer(self):
        """Test factory function."""
        tokenizer = get_simple_tokenizer(vocab_size=128)
        
        assert isinstance(tokenizer, SimpleCharTokenizer)
        assert tokenizer.vocab_size == 128


class TestSimpleTextDataset:
    """Test simple synthetic text dataset."""
    
    def test_initialization(self):
        """Test dataset initialization."""
        dataset = SimpleTextDataset(
            num_samples=100,
            seq_length=64,
            vocab_size=1000,
            seed=42,
        )
        
        assert len(dataset) == 100
        assert dataset.seq_length == 64
        assert dataset.vocab_size == 1000
    
    def test_getitem(self):
        """Test getting items."""
        dataset = SimpleTextDataset(num_samples=50, seq_length=32)
        
        inputs, targets = dataset[0]
        
        assert inputs.shape == (32,)
        assert targets.shape == (32,)
        assert inputs.dtype == torch.long
        assert targets.dtype == torch.long
    
    def test_sequence_shift(self):
        """Test that targets are shifted by 1."""
        dataset = SimpleTextDataset(num_samples=10, seq_length=8)
        
        inputs, targets = dataset[0]
        
        # Targets should be inputs shifted by 1
        seq = dataset.data[0]
        assert torch.equal(inputs, seq[:-1])
        assert torch.equal(targets, seq[1:])
    
    def test_vocab_range(self):
        """Test vocabulary range."""
        vocab_size = 500
        dataset = SimpleTextDataset(
            num_samples=20,
            seq_length=16,
            vocab_size=vocab_size,
        )
        
        inputs, targets = dataset[0]
        
        assert inputs.min() >= 0
        assert inputs.max() < vocab_size
        assert targets.min() >= 0
        assert targets.max() < vocab_size
    
    def test_reproducibility(self):
        """Test dataset reproducibility with seed."""
        dataset1 = SimpleTextDataset(num_samples=10, seed=123)
        dataset2 = SimpleTextDataset(num_samples=10, seed=123)
        
        inputs1, targets1 = dataset1[0]
        inputs2, targets2 = dataset2[0]
        
        assert torch.equal(inputs1, inputs2)
        assert torch.equal(targets1, targets2)


class TestTextDataset:
    """Test text dataset with file loading."""
    
    @pytest.fixture
    def sample_text_file(self):
        """Create a temporary text file."""
        text = "This is a test file. " * 100  # Repeat for enough tokens
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(text)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink()
    
    def test_initialization(self, sample_text_file):
        """Test dataset initialization from file."""
        tokenizer = get_simple_tokenizer()
        
        dataset = TextDataset(
            file_path=sample_text_file,
            seq_length=32,
            tokenizer=tokenizer,
            max_samples=10,
        )
        
        assert len(dataset) == 10
        assert dataset.seq_length == 32
    
    def test_getitem(self, sample_text_file):
        """Test getting items from file dataset."""
        tokenizer = get_simple_tokenizer()
        
        dataset = TextDataset(
            file_path=sample_text_file,
            seq_length=16,
            tokenizer=tokenizer,
            max_samples=5,
        )
        
        inputs, targets = dataset[0]
        
        assert inputs.shape == (16,)
        assert targets.shape == (16,)
        assert inputs.dtype == torch.long
        assert targets.dtype == torch.long
    
    def test_stride(self, sample_text_file):
        """Test stride parameter."""
        tokenizer = get_simple_tokenizer()
        
        # Default stride (seq_length // 2)
        dataset1 = TextDataset(
            file_path=sample_text_file,
            seq_length=32,
            tokenizer=tokenizer,
        )
        
        # Custom stride
        dataset2 = TextDataset(
            file_path=sample_text_file,
            seq_length=32,
            stride=8,
            tokenizer=tokenizer,
        )
        
        # More samples with smaller stride
        assert len(dataset2) > len(dataset1)
    
    def test_file_not_found(self):
        """Test error handling for missing file."""
        tokenizer = get_simple_tokenizer()
        
        with pytest.raises(FileNotFoundError):
            TextDataset(
                file_path="nonexistent_file.txt",
                seq_length=32,
                tokenizer=tokenizer,
            )


@pytest.mark.parametrize("num_samples,seq_length,vocab_size", [
    (50, 32, 500),
    (100, 64, 1000),
    (200, 128, 5000),
])
def test_dataset_parametrized(num_samples, seq_length, vocab_size):
    """Test dataset with various parameters."""
    dataset = SimpleTextDataset(
        num_samples=num_samples,
        seq_length=seq_length,
        vocab_size=vocab_size,
    )
    
    assert len(dataset) == num_samples
    
    inputs, targets = dataset[0]
    assert inputs.shape == (seq_length,)
    assert targets.shape == (seq_length,)
    assert inputs.max() < vocab_size
    assert targets.max() < vocab_size


def test_dataloader_integration():
    """Test integration with PyTorch DataLoader."""
    from torch.utils.data import DataLoader
    
    dataset = SimpleTextDataset(num_samples=100, seq_length=32)
    
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
    )
    
    batch_inputs, batch_targets = next(iter(loader))
    
    assert batch_inputs.shape == (16, 32)
    assert batch_targets.shape == (16, 32)
    assert batch_inputs.dtype == torch.long
    assert batch_targets.dtype == torch.long


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
