"""
GPT-2 model loader and wrapper.

This module provides functionality to load and use GPT-2 models from HuggingFace.
We start with the 124M parameter model for Day 1 of our tiny vLLM implementation.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Config
from typing import Dict, Any, Optional


class GPT2Model:
    """
    A wrapper around HuggingFace's GPT2LMHeadModel for our inference engine.

    This class encapsulates the model loading and provides a clean interface
    for forward passes. We start with the smallest GPT-2 model (124M parameters)
    and will scale up in later days.
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        """
        Initialize the GPT-2 model.

        Args:
            model_name: The HuggingFace model identifier (default: "gpt2" for 124M)
            device: Device to load the model on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device
        self.model: Optional[GPT2LMHeadModel] = None
        self.config: Optional[GPT2Config] = None

    def load_model(self) -> None:
        """
        Load the GPT-2 model and configuration from HuggingFace.

        This downloads the pretrained weights and sets up the model for inference.
        The model is set to evaluation mode to disable dropout and other training-specific layers.
        """
        print(f"Loading GPT-2 model: {self.model_name}")

        # Load configuration
        self.config = GPT2Config.from_pretrained(self.model_name)
        print(
            f"Model config: {self.config.n_layer} layers, {self.config.n_head} heads, {self.config.n_embd} embedding dim"
        )

        # Load model
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Calculate and display model size
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded with {param_count:,} parameters")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the model.

        Args:
            input_ids: Tensor of token IDs with shape (batch_size, sequence_length)

        Returns:
            Logits tensor with shape (batch_size, sequence_length, vocab_size)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        input_ids = input_ids.to(self.device)

        with torch.no_grad():  # Disable gradient computation for inference
            outputs = self.model(input_ids)
            return outputs.logits

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model metadata
        """
        if self.config is None:
            return {"error": "Model not loaded"}

        return {
            "model_name": self.model_name,
            "n_layers": self.config.n_layer,
            "n_heads": self.config.n_head,
            "n_embd": self.config.n_embd,
            "vocab_size": self.config.vocab_size,
            "device": self.device,
            "parameter_count": sum(p.numel() for p in self.model.parameters())
            if self.model
            else 0,
        }
