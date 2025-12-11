"""
WikiText dataset loader with proper preprocessing.

Copyright (c) 2025 Gregory Ward - SmartLedger.Technology
Licensed under the MIT License. See LICENSE file in the project root.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader


class WikiTextDataset(Dataset):
    """
    WikiText-2 or WikiText-103 dataset for language modeling.
    
    This dataset uses HuggingFace datasets library for reliable downloading
    and caching of WikiText data.
    
    Attributes:
        data: Tokenized text data
        seq_length: Maximum sequence length
        stride: Stride for creating overlapping sequences
    """
    
    def __init__(
        self,
        dataset_name: str = 'wikitext-2-raw-v1',
        split: str = 'train',
        seq_length: int = 512,
        stride: Optional[int] = None,
        tokenizer = None,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize WikiText dataset.
        
        Args:
            dataset_name: 'wikitext-2-raw-v1' or 'wikitext-103-raw-v1'
            split: 'train', 'valid', or 'test'
            seq_length: Maximum sequence length
            stride: Stride for overlapping sequences (default: seq_length // 2)
            tokenizer: Tokenizer object (if None, will use GPT-2 tokenizer)
            cache_dir: Directory to cache downloaded data
            max_samples: Maximum number of samples (for debugging)
        """
        super().__init__()
        
        self.dataset_name = dataset_name
        self.split = split if split != 'valid' else 'validation'  # HF uses 'validation'
        self.seq_length = seq_length
        self.stride = stride if stride is not None else seq_length // 2
        self.max_samples = max_samples
        self.cache_dir = cache_dir
        
        # Get tokenizer
        if tokenizer is None:
            try:
                from .tokenizer import get_tokenizer
            except ImportError:
                from data.tokenizer import get_tokenizer
            tokenizer = get_tokenizer('gpt2')
        self.tokenizer = tokenizer
        
        # Load and tokenize data
        self.data = self._load_and_tokenize()
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        if self.max_samples is not None:
            self.sequences = self.sequences[:self.max_samples]
    
    def _load_and_tokenize(self) -> torch.Tensor:
        """Load raw text using HuggingFace datasets and tokenize."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library is required. Install with: pip install datasets"
            )
        
        print(f"Loading {self.dataset_name} {self.split} split...")
        
        # Load dataset from HuggingFace
        dataset = load_dataset(
            'wikitext',
            self.dataset_name,
            split=self.split,
            cache_dir=self.cache_dir,
            trust_remote_code=False,
        )
        
        # Join all text
        texts = [item['text'] for item in dataset if item['text'].strip()]
        text = '\n'.join(texts)
        
        print(f"Loaded {len(text):,} characters")
        
        # Tokenize
        print("Tokenizing...")
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        print(f"Created {len(tokens):,} tokens")
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def _create_sequences(self) -> List[torch.Tensor]:
        """Create overlapping sequences from tokenized data."""
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
        input_ids = seq[:-1]  # All but last token
        target_ids = seq[1:]  # All but first token
        
        return input_ids, target_ids


def create_wikitext_dataloaders(
    dataset_name: str = 'wikitext-2-raw-v1',
    seq_length: int = 512,
    batch_size: int = 8,
    tokenizer = None,
    cache_dir: Optional[str] = None,
    num_workers: int = 0,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for WikiText.
    
    Args:
        dataset_name: 'wikitext-2-raw-v1' or 'wikitext-103-raw-v1'
        seq_length: Maximum sequence length
        batch_size: Batch size
        tokenizer: Tokenizer object
        cache_dir: Cache directory
        num_workers: Number of dataloader workers
        max_train_samples: Maximum training samples
        max_val_samples: Maximum validation samples
        
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
    """
    # Create datasets
    train_dataset = WikiTextDataset(
        dataset_name=dataset_name,
        split='train',
        seq_length=seq_length,
        tokenizer=tokenizer,
        cache_dir=cache_dir,
        max_samples=max_train_samples,
    )
    
    val_dataset = WikiTextDataset(
        dataset_name=dataset_name,
        split='valid',
        seq_length=seq_length,
        tokenizer=tokenizer,
        cache_dir=cache_dir,
        max_samples=max_val_samples,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    """Test WikiText dataset loading."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data.tokenizer import get_tokenizer
    
    print("Testing WikiText-2 dataset loading...")
    
    # Get tokenizer
    tokenizer = get_tokenizer('gpt2')
    
    # Create small dataset for testing
    dataset = WikiTextDataset(
        dataset_name='wikitext-2-raw-v1',
        split='train',
        seq_length=128,
        tokenizer=tokenizer,
        max_samples=10,
    )
    
    print(f"\nDataset size: {len(dataset)} sequences")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Test first sample
    input_ids, target_ids = dataset[0]
    print(f"\nFirst sample:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Target shape: {target_ids.shape}")
    print(f"  Input text: {tokenizer.decode(input_ids[:50].tolist())}...")
    
    # Test dataloader
    train_loader, val_loader = create_wikitext_dataloaders(
        dataset_name='wikitext-2-raw-v1',
        seq_length=128,
        batch_size=4,
        tokenizer=tokenizer,
        max_train_samples=20,
        max_val_samples=10,
    )
    
    print(f"\nDataLoader test:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Get first batch
    inputs, targets = next(iter(train_loader))
    print(f"  Batch input shape: {inputs.shape}")
    print(f"  Batch target shape: {targets.shape}")
    
    print("\nSuccess! WikiText dataset is working.")
