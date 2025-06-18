#!/usr/bin/env python3
"""
Test script for tokenizer module.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from nlp.preprocessing.tokenization import WordTokenizer

def test_word_tokenizer():
    """Test the WordTokenizer functionality."""
    text = "Hello, world! This is a test."
    tokenizer = WordTokenizer()
    tokens = tokenizer.tokenize(text)
    print(f"Original text: {text}")
    print(f"Tokens: {tokens}")

if __name__ == "__main__":
    test_word_tokenizer()
