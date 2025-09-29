"""
GPT-2 model loader and wrapper.

This module provides functionality to load and use GPT-2 models from HuggingFace.
We start with the 124M parameter model for Day 1 of our tiny vLLM implementation.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Config
from typing import Dict, Any, Optional, List


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

        A forward pass feeds input data through the neural network from input to output,
        layer by layer, to compute predictions. In GPT-2, this process involves:

        1. Token Embedding: Convert token IDs to dense vectors (768 dimensions for GPT-2)
        2. Position Encoding: Add positional information so model knows token order
        3. Transformer Layers (12 layers for GPT-2 124M):
           - Self-Attention: Each token "looks at" other tokens to understand context
           - Feed-Forward Network: Process the attended information
           - Layer Normalization & Residual Connections: Stabilize computation
        4. Output Projection: Convert final hidden states to vocabulary-sized logits (50,257 values)

        The flow: Input Tokens → Embedding → Transformer Layers → Output Logits
        Example: [15496, 995] → [768-dim vectors] → [12 layers] → [50,257 predictions]

        Args:
            input_ids: Tensor of token IDs with shape (batch_size, sequence_length)

        Returns:
            Logits tensor with shape (batch_size, sequence_length, vocab_size)
            Each logit represents the model's confidence for that vocabulary token
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        input_ids = input_ids.to(self.device)

        with torch.no_grad():  # Disable gradient computation for inference
            # This optimization reduces memory usage by 30-50% and improves performance
            # since we don't need gradients for inference (only forward pass predictions)
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

    def generate_greedy(
        self, input_ids: torch.Tensor, max_length: int = 20, show_progress: bool = False
    ) -> List[int]:
        """
        Generate text using greedy decoding (argmax at each step).

        Greedy decoding is the simplest generation strategy:
        1. Start with input tokens (e.g., "Once upon a time")
        2. Run forward pass to get logits for next token
        3. Take the token with highest probability (argmax)
        4. Append it to sequence and repeat

        This creates deterministic output - same input always produces same result.
        While not as creative as sampling methods, it's fast and predictable.

        Args:
            input_ids: Starting tokens with shape (1, sequence_length)
            max_length: Maximum number of NEW tokens to generate
            show_progress: Whether to print each generated token

        Returns:
            List of ALL token IDs (input + generated tokens)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert to list for easier manipulation
        sequence = input_ids.squeeze().tolist()
        original_length = len(sequence)

        if show_progress:
            print(f"Starting generation with {original_length} input tokens...")

        # Generate tokens one by one
        for step in range(max_length):
            # Current sequence as tensor
            current_ids = torch.tensor(sequence).unsqueeze(0).to(self.device)

            # Get logits for next token (forward pass through all 124M parameters)
            with torch.no_grad():
                outputs = self.model(current_ids)
                logits = outputs.logits

            # Get logits for the last position (what comes next)
            next_token_logits = logits[0, -1, :]  # Shape: [vocab_size]

            # Greedy selection: pick token with highest logit (most confident prediction)
            next_token_id = torch.argmax(next_token_logits).item()

            # Add to sequence
            sequence.append(next_token_id)

            if show_progress:
                # Decode the new token to show what was generated
                from ruvonvllm.tokenizer.gpt2_tokenizer import GPT2TokenizerWrapper

                tokenizer = GPT2TokenizerWrapper(self.model_name)
                token_text = tokenizer.decode([next_token_id])
                print(
                    f"Step {step + 1}: Generated token {next_token_id} -> '{token_text}'"
                )

            # Check for end-of-sequence token (optional early stopping)
            if (
                hasattr(self.model.config, "eos_token_id")
                and next_token_id == self.model.config.eos_token_id
            ):
                if show_progress:
                    print("Generated end-of-sequence token, stopping early.")
                break

        if show_progress:
            print(
                f"Generation complete! Generated {len(sequence) - original_length} new tokens."
            )

        return sequence
