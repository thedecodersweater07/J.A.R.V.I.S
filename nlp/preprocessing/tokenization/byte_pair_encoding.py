"""
Byte Pair Encoding (BPE) tokenizer implementation.

This module provides functionality for tokenizing text using Byte Pair Encoding,
which is a subword tokenization algorithm commonly used in modern NLP models.
"""

import re
import os
import json
import unicodedata
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional, Union, Pattern, Any, Callable

class BPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer implementation.
    
    This tokenizer implements the BPE algorithm which is commonly used in modern NLP models.
    It can be trained on a corpus to learn subword units that provide a good balance
    between word-level and character-level representations.
    """
    """Byte Pair Encoding tokenizer implementation."""
    
    def __init__(self, 
                 vocab_size: int = 1000, 
                 special_tokens: Optional[List[str]] = None,
                 lowercase: bool = False,
                 unk_token: str = "[UNK]",
                 pad_token: str = "[PAD]",
                 bos_token: str = "[BOS]",
                 eos_token: str = "[EOS]"):
        """Initialize the BPE tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            special_tokens: List of special tokens to include in the vocabulary
            lowercase: Whether to convert text to lowercase before tokenization
            unk_token: Token to use for unknown words
            pad_token: Token to use for padding
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
        """
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        
        # Default special tokens
        default_special_tokens = [unk_token, pad_token, bos_token, eos_token]
        self.special_tokens = list(set(default_special_tokens + (special_tokens or [])))
        
        # Initialize token mappings and vocabulary
        self.vocab = {}
        self.merges = {}
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Regex pattern for tokenization
        self.pattern = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+\S*|\s+",
            re.UNICODE
        )
        
        # Add special tokens to vocabulary
        self._add_special_tokens()
    
    def _add_special_tokens(self) -> None:
        """Add special tokens to the vocabulary."""
        for token in self.special_tokens:
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
    
    def train(self, texts: List[str]) -> None:
        """Train the BPE tokenizer on the given texts.
        
        Args:
            texts: List of training texts
        """
        # Tokenize the input texts into words
        words = []
        for text in texts:
            tokens = self._tokenize_text(text)
            words.extend(tokens)
        
        # Initialize vocabulary with characters
        vocab = self._get_vocab(words)
        
        # Perform BPE merges
        while len(vocab) < self.vocab_size:
            # Find the most frequent pair
            pairs = self._get_pairs(words)
            if not pairs:
                break
                
            most_frequent_pair = max(pairs, key=pairs.get)  # type: ignore
            
            # Merge the most frequent pair
            new_word = "".join(most_frequent_pair)
            vocab[new_word] = sum(count for word, count in words if new_word in word)
            
            # Update words by merging the pair
            words = self._merge_pair(words, most_frequent_pair, new_word)
            
            # Store the merge
            self.merges[most_frequent_pair] = new_word
        
        # Update vocabulary
        self.vocab = vocab
        
        # Update token mappings
        for i, token in enumerate(sorted(vocab.keys())):
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)
                self.id_to_token[len(self.id_to_token)] = token
    
    def _tokenize_text(self, text: str) -> List[Tuple[str, int]]:
        """Tokenize text into words with counts.
        
        Args:
            text: Input text
            
        Returns:
            List of (word, count) tuples
        """
        tokens = self.pattern.findall(text)
        word_counts = Counter(tokens)
        return list(word_counts.items())
    
    def _get_vocab(self, words: List[Tuple[str, int]]) -> Dict[str, int]:
        """Get initial vocabulary from words.
        
        Args:
            words: List of (word, count) tuples
            
        Returns:
            Vocabulary dictionary mapping tokens to frequencies
        """
        vocab = defaultdict(int)
        for word, count in words:
            for char in word:
                vocab[char] += count
        return dict(vocab)
    
    def _get_pairs(self, words: List[Tuple[str, int]]) -> Dict[Tuple[str, str], int]:
        """Get frequency of adjacent symbol pairs.
        
        Args:
            words: List of (word, count) tuples
            
        Returns:
            Dictionary mapping pairs to their frequencies
        """
        pairs = defaultdict(int)
        for word, count in words:
            symbols = list(word)
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += count
        return pairs
    
    def _merge_pair(self, words: List[Tuple[str, int]], pair: Tuple[str, str], new_token: str) -> List[Tuple[str, int]]:
        """Merge a pair of tokens in the word list.
        
        Args:
            words: List of (word, count) tuples
            pair: Pair of tokens to merge
            new_token: New token to replace the pair with
            
        Returns:
            Updated list of (word, count) tuples
        """
        new_words = []
        for word, count in words:
            new_word = word.replace("".join(pair), new_token)
            new_words.append((new_word, count))
        return new_words
    
    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        tokens = self.pattern.findall(text)
        token_ids = []
        
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                # Handle unknown tokens (can be improved with subword splitting)
                for char in token:
                    if char in self.token_to_id:
                        token_ids.append(self.token_to_id[char])
                    else:
                        # Use UNK token if available, otherwise skip
                        if "<UNK>" in self.token_to_id:
                            token_ids.append(self.token_to_id["<UNK>"])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        tokens = [self.id_to_token.get(token_id, "<UNK>") for token_id in token_ids]
        return "".join(tokens).replace(" ", " ").strip()
    
    def save(self, filepath: str) -> None:
        """Save the tokenizer to a file.
        
        Args:
            filepath: Path to save the tokenizer
        """
        data = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'vocab': self.vocab,
            'merges': {','.join(k): v for k, v in self.merges.items()},
            'token_to_id': self.token_to_id,
            'id_to_token': {int(k): v for k, v in self.id_to_token.items()}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'BPETokenizer':
        """Load a tokenizer from a file.
        
        Args:
            filepath: Path to the saved tokenizer
            
        Returns:
            Loaded BPETokenizer instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(
            vocab_size=data['vocab_size'],
            special_tokens=data['special_tokens']
        )
        
        tokenizer.vocab = data['vocab']
        tokenizer.merges = {tuple(k.split(',')): v for k, v in data['merges'].items()}
        tokenizer.token_to_id = {k: int(v) if isinstance(v, str) else v for k, v in data['token_to_id'].items()}
        tokenizer.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        
        return tokenizer
