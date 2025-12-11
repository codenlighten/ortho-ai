"""
Tokenizer utilities for text preprocessing.

Copyright (c) 2025 Gregory Ward - SmartLedger.Technology
Licensed under the MIT License. See LICENSE file in the project root.
"""

from typing import Optional
import torch


def get_tokenizer(tokenizer_name: str = "gpt2"):
    """
    Get a tokenizer for text preprocessing.
    
    Args:
        tokenizer_name: Name of the tokenizer ('gpt2', 'gpt2-medium', etc.)
        
    Returns:
        Tokenizer object with encode/decode methods
        
    Example:
        >>> tokenizer = get_tokenizer("gpt2")
        >>> ids = tokenizer.encode("Hello world")
        >>> text = tokenizer.decode(ids)
    """
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Ensure pad token is set (GPT-2 doesn't have one by default)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer
        
    except ImportError:
        raise ImportError(
            "transformers library is required for tokenization. "
            "Install it with: pip install transformers"
        )


class SimpleCharTokenizer:
    """
    Simple character-level tokenizer for testing without external dependencies.
    
    This is a fallback tokenizer that operates at the character level.
    It's useful for testing but not recommended for production use.
    """
    
    def __init__(self, vocab_size: int = 256):
        """
        Initialize character tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size (default: 256 for ASCII)
        """
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to list of token IDs."""
        tokens = [min(ord(c), self.vocab_size - 1) for c in text]
        
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
            
        return tokens
        
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if skip_special_tokens:
            token_ids = [
                t for t in token_ids 
                if t not in (self.pad_token_id, self.eos_token_id, self.bos_token_id)
            ]
        
        return ''.join(chr(min(t, 127)) for t in token_ids)
        
    def __call__(self, text: str, **kwargs) -> dict:
        """Tokenize text (compatible with HuggingFace interface)."""
        tokens = self.encode(text, add_special_tokens=kwargs.get('add_special_tokens', True))
        return {
            'input_ids': tokens,
            'attention_mask': [1] * len(tokens)
        }


def get_simple_tokenizer(vocab_size: int = 256) -> SimpleCharTokenizer:
    """
    Get a simple character-level tokenizer for testing.
    
    Args:
        vocab_size: Vocabulary size (default: 256)
        
    Returns:
        SimpleCharTokenizer instance
    """
    return SimpleCharTokenizer(vocab_size=vocab_size)
