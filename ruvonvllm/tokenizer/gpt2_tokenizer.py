"""
GPT-2 tokenizer wrapper.

This module provides functionality to tokenize and detokenize text using
the GPT-2 tokenizer from HuggingFace. This is essential for converting
between human-readable text and the token IDs that the model understands.
"""

import torch
from transformers import GPT2Tokenizer
from typing import List, Union


class GPT2TokenizerWrapper:
    """
    A wrapper around HuggingFace's GPT2Tokenizer.

    This class provides a clean interface for tokenization operations,
    converting between text and token IDs for our inference engine.
    """

    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the tokenizer.

        Args:
            model_name: The HuggingFace model identifier (default: "gpt2")
        """
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # GPT-2 tokenizer doesn't have a pad token by default, so we set one
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Tokenizer loaded for {model_name}")
        print(f"Vocab size: {self.tokenizer.vocab_size}")

    def encode(
        self, text: str, return_tensors: bool = True
    ) -> Union[torch.Tensor, List[int]]:
        """
        Convert text to token IDs.

        Args:
            text: Input text to tokenize
            return_tensors: If True, return PyTorch tensor; if False, return list

        Returns:
            Token IDs as tensor or list
        """
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)

        if return_tensors:
            return torch.tensor(token_ids).unsqueeze(0)  # Add batch dimension
        return token_ids

    def decode(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Convert token IDs back to text.

        Args:
            token_ids: Token IDs to decode (tensor or list)
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            # If it's a tensor, convert to list and remove batch dimension if present
            if token_ids.dim() > 1:
                token_ids = token_ids.squeeze().tolist()
            else:
                token_ids = token_ids.tolist()

        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_vocab_size(self) -> int:
        """
        Get the size of the tokenizer's vocabulary.

        Returns:
            Vocabulary size
        """
        return self.tokenizer.vocab_size

    def tokenize_and_show_details(self, text: str) -> None:
        """
        Tokenize text and print detailed information for debugging/demonstration.

        This method is useful for understanding how the tokenizer breaks down text.

        Args:
            text: Input text to analyze
        """
        print(f"Input text: '{text}'")

        # Get token IDs
        token_ids = self.encode(text, return_tensors=False)
        print(f"Token IDs: {token_ids}")

        # Get individual tokens
        tokens = self.tokenizer.tokenize(text)
        print(f"Tokens: {tokens}")

        # Show token-by-token breakdown
        print("Token breakdown:")
        for i, (token_id, token) in enumerate(zip(token_ids, tokens + ["<end>"])):
            if i < len(tokens):
                decoded_token = self.tokenizer.decode([token_id])
                print(f"  {i}: {token_id} -> '{token}' -> '{decoded_token}'")

        # Decode back to verify
        decoded_text = self.decode(token_ids)
        print(f"Decoded text: '{decoded_text}'")
        print(f"Round-trip successful: {text == decoded_text}")
