"""
General text dataset for custom data files.

Copyright (c) 2025 Gregory Ward - SmartLedger.Technology
Licensed under the MIT License. See LICENSE file in the project root.
"""

from pathlib import Path
from typing import Optional, Tuple, List, Union
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    General text dataset for language modeling from custom text files.
    
    This dataset can load any plain text file, tokenize it, and create
    sequences for training.
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        seq_length: int = 512,
        stride: Optional[int] = None,
        tokenizer = None,
        max_samples: Optional[int] = None,
        encoding: str = 'utf-8',
    ):
        """
        Initialize text dataset from file.
        
        Args:
            file_path: Path to text file
            seq_length: Maximum sequence length
            stride: Stride for overlapping sequences (default: seq_length // 2)
            tokenizer: Tokenizer object (if None, will use GPT-2 tokenizer)
            max_samples: Maximum number of samples
            encoding: Text file encoding
        """
        super().__init__()
        
        self.file_path = Path(file_path)
        self.seq_length = seq_length
        self.stride = stride if stride is not None else seq_length // 2
        self.max_samples = max_samples
        self.encoding = encoding
        
        # Get tokenizer
        if tokenizer is None:
            try:
                from .tokenizer import get_tokenizer
            except ImportError:
                from data.tokenizer import get_tokenizer
            tokenizer = get_tokenizer('gpt2')
        self.tokenizer = tokenizer
        
        # Load and tokenize
        self.data = self._load_and_tokenize()
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        if self.max_samples is not None:
            self.sequences = self.sequences[:self.max_samples]
    
    def _load_and_tokenize(self) -> torch.Tensor:
        """Load text file and tokenize."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        print(f"Loading {self.file_path.name}...")
        with open(self.file_path, 'r', encoding=self.encoding) as f:
            text = f.read()
        
        print("Tokenizing...")
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        print(f"Loaded {len(tokens):,} tokens")
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def _create_sequences(self) -> List[torch.Tensor]:
        """Create overlapping sequences."""
        sequences = []
        
        for i in range(0, len(self.data) - self.seq_length, self.stride):
            seq = self.data[i:i + self.seq_length + 1]  # +1 for target
            if len(seq) == self.seq_length + 1:
                sequences.append(seq)
        
        print(f"Created {len(sequences):,} sequences")
        return sequences
    
    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sequence.
        
        Returns:
            input_ids: Token IDs [seq_length]
            target_ids: Target token IDs [seq_length]
        """
        seq = self.sequences[idx]
        input_ids = seq[:-1]
        target_ids = seq[1:]
        
        return input_ids, target_ids


class SimpleTextDataset(Dataset):
    """
    Simple text dataset for quick testing without external dependencies.
    
    This creates random sequences for testing the training loop without
    requiring actual text data or tokenizers.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        seq_length: int = 128,
        vocab_size: int = 10000,
        seed: int = 42,
    ):
        """
        Initialize simple synthetic dataset.
        
        Args:
            num_samples: Number of sequences
            seq_length: Sequence length
            vocab_size: Vocabulary size
            seed: Random seed
        """
        super().__init__()
        
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # Generate random data
        torch.manual_seed(seed)
        self.data = torch.randint(
            0, vocab_size,
            (num_samples, seq_length + 1),
            dtype=torch.long
        )
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training sequence."""
        seq = self.data[idx]
        input_ids = seq[:-1]
        target_ids = seq[1:]
        
        return input_ids, target_ids


if __name__ == '__main__':
    """Test text dataset."""
    print("Testing SimpleTextDataset...")
    
    dataset = SimpleTextDataset(
        num_samples=100,
        seq_length=128,
        vocab_size=10000,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    inputs, targets = dataset[0]
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Sample tokens: {inputs[:10].tolist()}")
    
    print("\nSuccess! TextDataset is working.")
