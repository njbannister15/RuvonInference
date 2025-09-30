"""
GPT-2 model loader and wrapper.

This module provides functionality to load and use GPT-2 models from HuggingFace.
We start with the 124M parameter model for Day 1 of our tiny vLLM implementation.
"""

import torch
from transformers import DynamicCache, GPT2LMHeadModel, GPT2Config
from typing import Dict, Any, Optional, List

from ruvonvllm.sampling.strategies import sample_token, get_sampling_info


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
        self.model.eval()  # Disable dropout/batch norm for deterministic inference

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

    def generate_greedy_with_cache(
        self, input_ids: torch.Tensor, max_length: int = 20, show_progress: bool = False
    ) -> List[int]:
        """
        Generate text using greedy decoding with KV-cache optimization.

        KV-caching is a crucial optimization for autoregressive generation:
        - Problem: Each generation step recomputes attention for ALL previous tokens
        - Solution: Cache the Key/Value states since they never change for past tokens
        - Result: 10-20x speedup for longer sequences

        How it works:
        1. First forward pass: Compute K/V for all input tokens, cache them
        2. Subsequent passes: Only compute K/V for the new token, append to cache
        3. Attention: Use cached K/V for past tokens + new K/V for current token

        This transforms O(n²) complexity to O(n) for generation.

        Args:
            input_ids: Starting tokens with shape (1, sequence_length)
            max_length: Maximum number of NEW tokens to generate
            show_progress: Whether to print each generated token

        Returns:
            List of ALL token IDs (input + generated tokens)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert tensor (1, seq_len) -> list for easier token appending
        # squeeze() removes batch dimension: (1, seq_len) -> (seq_len), then tolist() -> Python list
        sequence: List = input_ids.squeeze().tolist()
        original_length = len(sequence)

        if show_progress:
            print(f"Starting cached generation with {original_length} input tokens...")

        # Initialize past_key_values cache (will be populated on first forward pass)
        past_key_values: DynamicCache | None = None

        # Generate tokens one by one using KV-cache
        for step in range(max_length):
            if past_key_values is None:
                # First forward pass: process entire input sequence
                current_ids = torch.tensor(sequence).unsqueeze(0).to(self.device)
                if show_progress:
                    print(
                        f"Step {step + 1}: Processing full sequence ({len(sequence)} tokens)"
                    )
            else:
                # Subsequent passes: only process the new token
                current_ids = torch.tensor([sequence[-1]]).unsqueeze(0).to(self.device)
                if show_progress:
                    print(
                        f"Step {step + 1}: Processing only new token (cached: {len(sequence) - 1} tokens)"
                    )

            # Forward pass with KV-cache
            # past_key_values contains cached attention states from previous steps
            with torch.no_grad():
                outputs = self.model(
                    current_ids,
                    past_key_values=past_key_values,
                    use_cache=True,  # Enable caching
                )
                logits = outputs.logits
                past_key_values = outputs.past_key_values  # Update cache

            # Get logits for the last position (what comes next)
            next_token_logits = logits[0, -1, :]

            # Greedy selection: pick token with highest logit
            next_token_id = torch.argmax(next_token_logits).item()

            # Add to sequence
            sequence.append(next_token_id)

            if show_progress:
                # Decode the new token to show what was generated
                from ruvonvllm.tokenizer.gpt2_tokenizer import GPT2TokenizerWrapper

                tokenizer = GPT2TokenizerWrapper(self.model_name)
                token_text = tokenizer.decode([next_token_id])
                cache_info = (
                    f"(cache size: {len(past_key_values[0][0][0])} tokens)"
                    if past_key_values
                    else "(no cache)"
                )
                print(f"Generated token {next_token_id} -> '{token_text}' {cache_info}")

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
                f"Cached generation complete! Generated {len(sequence) - original_length} new tokens."
            )

        return sequence

    def benchmark_generation(
        self, input_ids: torch.Tensor, max_length: int = 20, num_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark generation performance with and without KV-cache.

        This method runs both generation approaches multiple times and measures:
        - Total generation time
        - Time per token
        - Speedup factor
        - Memory efficiency

        Args:
            input_ids: Starting tokens for generation
            max_length: Number of tokens to generate
            num_runs: Number of benchmark runs for averaging

        Returns:
            Dictionary containing performance metrics
        """
        import time

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        print(f"🏁 Benchmarking generation: {max_length} tokens, {num_runs} runs each")
        print("=" * 60)

        # Benchmark without cache (naive approach)
        print("🐌 Testing WITHOUT KV-cache (naive)...")
        no_cache_times = []
        for run in range(num_runs):
            start_time = time.time()
            _ = self.generate_greedy(input_ids, max_length, show_progress=False)
            end_time = time.time()
            run_time = end_time - start_time
            no_cache_times.append(run_time)
            print(f"  Run {run + 1}: {run_time:.3f}s")

        # Benchmark with cache (optimized approach)
        print("\n🚀 Testing WITH KV-cache (optimized)...")
        with_cache_times = []
        for run in range(num_runs):
            start_time = time.time()
            _ = self.generate_greedy_with_cache(
                input_ids, max_length, show_progress=False
            )
            end_time = time.time()
            run_time = end_time - start_time
            with_cache_times.append(run_time)
            print(f"  Run {run + 1}: {run_time:.3f}s")

        # Calculate statistics
        avg_no_cache = sum(no_cache_times) / len(no_cache_times)
        avg_with_cache = sum(with_cache_times) / len(with_cache_times)
        speedup = avg_no_cache / avg_with_cache
        time_per_token_no_cache = avg_no_cache / max_length
        time_per_token_with_cache = avg_with_cache / max_length

        results = {
            "no_cache_avg_time": avg_no_cache,
            "with_cache_avg_time": avg_with_cache,
            "speedup_factor": speedup,
            "time_per_token_no_cache": time_per_token_no_cache,
            "time_per_token_with_cache": time_per_token_with_cache,
            "num_tokens": max_length,
            "num_runs": num_runs,
        }

        print("\n📊 BENCHMARK RESULTS:")
        print("=" * 60)
        print(
            f"Without KV-cache: {avg_no_cache:.3f}s ({time_per_token_no_cache:.3f}s/token)"
        )
        print(
            f"With KV-cache:    {avg_with_cache:.3f}s ({time_per_token_with_cache:.3f}s/token)"
        )
        print(f"Speedup:          {speedup:.1f}x faster! 🚀")
        print("=" * 60)

        return results

    def generate_with_sampling(
        self,
        input_ids: torch.Tensor,
        max_length: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_cache: bool = True,
        show_progress: bool = False,
    ) -> List[int]:
        """
        Generate text using advanced sampling strategies.

        This method implements creative text generation by sampling from probability
        distributions rather than always choosing the most likely token (greedy).

        Sampling strategies:
        1. Temperature: Controls randomness vs confidence
           - Low (0.1-0.7): More focused, deterministic
           - Medium (0.8-1.2): Balanced creativity
           - High (1.3-2.0): More random, creative

        2. Top-k: Only consider k most likely tokens
           - Low k (1-10): Conservative, coherent
           - Medium k (20-100): Balanced variety
           - High k (200+): More diverse choices

        3. Top-p (nucleus): Dynamic vocabulary based on cumulative probability
           - Low p (0.1-0.5): Very focused
           - Medium p (0.6-0.9): Balanced
           - High p (0.95-1.0): More inclusive

        Args:
            input_ids: Starting tokens with shape (1, sequence_length)
            max_length: Maximum number of NEW tokens to generate
            temperature: Randomness control (default: 1.0 = no change)
            top_k: Number of top tokens to consider (default: None = no filtering)
            top_p: Cumulative probability threshold (default: None = no filtering)
            use_cache: Whether to use KV-cache optimization (default: True)
            show_progress: Whether to print generation details (default: False)

        Returns:
            List of ALL token IDs (input + generated tokens)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert tensor to list for easier manipulation
        sequence = input_ids.squeeze().tolist()
        original_length = len(sequence)

        if show_progress:
            print(
                f"Starting sampling generation with {original_length} input tokens..."
            )
            print(f"Parameters: temp={temperature}, top_k={top_k}, top_p={top_p}")

        # Initialize KV cache if using cached generation
        past_key_values = None if use_cache else None

        # Generate tokens one by one
        for step in range(max_length):
            # Prepare input for this step
            if past_key_values is None or not use_cache:
                # First step or no caching: process full sequence
                current_ids = torch.tensor(sequence).unsqueeze(0).to(self.device)
            else:
                # Subsequent steps with caching: only process new token
                current_ids = torch.tensor([sequence[-1]]).unsqueeze(0).to(self.device)

            # Forward pass
            with torch.no_grad():
                if use_cache:
                    outputs = self.model(
                        current_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                else:
                    outputs = self.model(current_ids)

                logits = outputs.logits

            # Get logits for the last position
            next_token_logits = logits[0, -1, :]

            # Use sampling instead of greedy selection
            next_token_id = sample_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Add to sequence
            sequence.append(next_token_id)

            if show_progress:
                # Show sampling info and generated token
                sampling_info = get_sampling_info(
                    next_token_logits, temperature, top_k, top_p
                )

                from ruvonvllm.tokenizer.gpt2_tokenizer import GPT2TokenizerWrapper

                tokenizer = GPT2TokenizerWrapper(self.model_name)
                token_text = tokenizer.decode([next_token_id])

                print(f"Step {step + 1}:")
                print(f"  Generated token: {next_token_id} -> '{token_text}'")
                print(f"  Top token prob: {sampling_info['top_token_prob']:.3f}")
                print(f"  Effective vocab: {sampling_info['effective_vocab_size']}")
                print(f"  Entropy change: {sampling_info['entropy_change']:.3f}")

            # Check for end-of-sequence token
            if (
                hasattr(self.model.config, "eos_token_id")
                and next_token_id == self.model.config.eos_token_id
            ):
                if show_progress:
                    print("Generated end-of-sequence token, stopping early.")
                break

        if show_progress:
            print(
                f"Sampling generation complete! Generated {len(sequence) - original_length} new tokens."
            )

        return sequence

    def compare_sampling_strategies(
        self,
        input_ids: torch.Tensor,
        max_length: int = 10,
        num_samples: int = 3,
    ) -> Dict[str, List[str]]:
        """
        Compare different sampling strategies on the same input.

        This method generates multiple outputs using different sampling approaches
        to demonstrate the variety and creativity that sampling can provide.

        Args:
            input_ids: Starting tokens
            max_length: Number of tokens to generate
            num_samples: Number of samples per strategy

        Returns:
            Dictionary mapping strategy names to lists of generated texts
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from ruvonvllm.tokenizer.gpt2_tokenizer import GPT2TokenizerWrapper

        tokenizer = GPT2TokenizerWrapper(self.model_name)

        prompt_text = tokenizer.decode(input_ids.squeeze().tolist())
        strategies = {
            "greedy": {"temperature": 0.1},
            "low_temp": {"temperature": 0.7},
            "medium_temp": {"temperature": 1.0},
            "high_temp": {"temperature": 1.5},
            "top_k_20": {"temperature": 0.8, "top_k": 20},
            "top_k_50": {"temperature": 0.8, "top_k": 50},
            "top_p_90": {"temperature": 0.8, "top_p": 0.9},
            "top_p_95": {"temperature": 0.8, "top_p": 0.95},
            "nucleus": {"temperature": 0.8, "top_k": 40, "top_p": 0.9},
        }

        results = {}

        print(f"🎭 Comparing sampling strategies on: '{prompt_text}'")
        print("=" * 60)

        for strategy_name, params in strategies.items():
            print(f"\n🎯 {strategy_name.upper()}: {params}")
            strategy_outputs = []

            for i in range(num_samples):
                generated_tokens = self.generate_with_sampling(
                    input_ids, max_length, use_cache=True, **params
                )
                full_text = tokenizer.decode(generated_tokens)
                generated_part = full_text[len(prompt_text) :]
                strategy_outputs.append(generated_part)

                print(f"  {i + 1}: '{generated_part}'")

            results[strategy_name] = strategy_outputs

        return results

    def generate_batch_with_sampling(
        self,
        batch_input_ids: List[torch.Tensor],
        max_length: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_cache: bool = True,
        show_progress: bool = False,
    ) -> List[List[int]]:
        """
        Generate text for multiple requests in a single batched forward pass.

        This is the core of continuous batching - processing multiple requests
        simultaneously to improve throughput while maintaining the same memory footprint.

        Args:
            batch_input_ids: List of input token tensors, one per request
            max_length: Maximum number of NEW tokens to generate per request
            temperature: Randomness control for sampling
            top_k: Number of top tokens to consider for sampling
            top_p: Cumulative probability threshold for nucleus sampling
            use_cache: Whether to use KV-cache optimization
            show_progress: Whether to print generation details

        Returns:
            List of token sequences (one per input request)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not batch_input_ids:
            return []

        batch_size = len(batch_input_ids)
        if show_progress:
            print(f"🚀 Starting batched generation for {batch_size} requests")

        # Convert all inputs to lists for easier manipulation
        batch_sequences = []
        original_lengths = []

        for input_ids in batch_input_ids:
            sequence = input_ids.squeeze().tolist()
            batch_sequences.append(sequence)
            original_lengths.append(len(sequence))

        # Track which sequences are still generating
        active_sequences = list(range(batch_size))

        # Initialize cache for batched processing
        past_key_values = None

        for step in range(max_length):
            if not active_sequences:
                break  # All sequences finished

            if show_progress and step % 5 == 0:
                print(
                    f"  Step {step}/{max_length}, {len(active_sequences)} active sequences"
                )

            # Prepare current batch (only active sequences)
            current_batch_ids = []
            current_indices = []

            for i, seq_idx in enumerate(active_sequences):
                current_batch_ids.append([batch_sequences[seq_idx][-1]])  # Last token
                current_indices.append(seq_idx)

            # Pad the batch to the same length (in this case, all are length 1)
            batch_tensor = torch.tensor(
                current_batch_ids, dtype=torch.long, device=self.device
            )

            # Forward pass for the entire batch
            with torch.no_grad():
                if step == 0:
                    # First step: process full sequences
                    full_batch_ids = []
                    max_seq_len = max(len(seq) for seq in batch_sequences)

                    # Pad sequences to same length
                    for seq_idx in active_sequences:
                        seq = batch_sequences[seq_idx].copy()
                        # Pad with tokenizer's pad token (0 for GPT-2)
                        while len(seq) < max_seq_len:
                            seq = [0] + seq  # Left padding
                        full_batch_ids.append(seq)

                    batch_tensor = torch.tensor(
                        full_batch_ids, dtype=torch.long, device=self.device
                    )

                    # Create attention mask (1 for real tokens, 0 for padding)
                    attention_mask = torch.zeros_like(batch_tensor)
                    for i, seq_idx in enumerate(active_sequences):
                        original_len = len(batch_sequences[seq_idx])
                        attention_mask[i, -original_len:] = 1

                    outputs = self.model(
                        batch_tensor,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                    )
                else:
                    # Subsequent steps: just process new tokens
                    outputs = self.model(
                        batch_tensor,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                    )

                logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

                if use_cache:
                    past_key_values = outputs.past_key_values

            # Sample next tokens for each sequence in the batch
            next_tokens = []
            sequences_to_remove = []

            for i, seq_idx in enumerate(current_indices):
                # Get logits for the last position of this sequence
                sequence_logits = logits[i, -1, :]  # Shape: (vocab_size,)

                # Sample next token using the same strategies as single generation
                next_token_id = sample_token(
                    sequence_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

                # Add token to sequence
                batch_sequences[seq_idx].append(next_token_id)
                next_tokens.append(next_token_id)

                # Check if this sequence should stop
                if (
                    hasattr(self.model.config, "eos_token_id")
                    and next_token_id == self.model.config.eos_token_id
                ):
                    sequences_to_remove.append(seq_idx)
                    if show_progress:
                        print(f"    Sequence {seq_idx} finished (EOS token)")

            # Remove finished sequences from active list
            for seq_idx in sequences_to_remove:
                active_sequences.remove(seq_idx)

        if show_progress:
            total_generated = sum(
                len(seq) - orig_len
                for seq, orig_len in zip(batch_sequences, original_lengths)
            )
            print(
                f"🎉 Batched generation complete! Generated {total_generated} total tokens"
            )

        return batch_sequences
