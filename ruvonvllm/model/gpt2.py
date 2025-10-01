"""
GPT-2 model loader and wrapper.

This module provides functionality to load and use GPT-2 models from HuggingFace.
We start with the 124M parameter model for Day 1 of our tiny vLLM implementation.
"""

import logging
import torch
from transformers import DynamicCache, GPT2LMHeadModel, GPT2Config
from typing import Dict, Any, Optional, List

from ruvonvllm.sampling.strategies import sample_token, get_sampling_info

logger = logging.getLogger(__name__)


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
        logger.info(f"Loading GPT-2 model: {self.model_name}")

        # Load configuration
        self.config = GPT2Config.from_pretrained(self.model_name)
        logger.info(
            f"Model config: {self.config.n_layer} layers, {self.config.n_head} heads, {self.config.n_embd} embedding dim"
        )

        # Load model
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()  # Disable dropout/batch norm for deterministic inference

        # Calculate and display model size
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded with {param_count:,} parameters")

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
            logger.info(f"Starting generation with {original_length} input tokens...")

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
                logger.info(
                    f"Step {step + 1}: Generated token {next_token_id} -> '{token_text}'"
                )

            # Check for end-of-sequence token (optional early stopping)
            if (
                hasattr(self.model.config, "eos_token_id")
                and next_token_id == self.model.config.eos_token_id
            ):
                if show_progress:
                    logger.info("Generated end-of-sequence token, stopping early.")
                break

        if show_progress:
            logger.info(
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
            logger.info(
                f"Starting cached generation with {original_length} input tokens..."
            )

        # Initialize past_key_values cache (will be populated on first forward pass)
        past_key_values: DynamicCache | None = None

        # Generate tokens one by one using KV-cache
        for step in range(max_length):
            if past_key_values is None:
                # First forward pass: process entire input sequence
                current_ids = torch.tensor(sequence).unsqueeze(0).to(self.device)
                if show_progress:
                    logger.info(
                        f"Step {step + 1}: Processing full sequence ({len(sequence)} tokens)"
                    )
            else:
                # Subsequent passes: only process the new token
                current_ids = torch.tensor([sequence[-1]]).unsqueeze(0).to(self.device)
                if show_progress:
                    logger.info(
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
                logger.info(
                    f"Generated token {next_token_id} -> '{token_text}' {cache_info}"
                )

            # Check for end-of-sequence token (optional early stopping)
            if (
                hasattr(self.model.config, "eos_token_id")
                and next_token_id == self.model.config.eos_token_id
            ):
                if show_progress:
                    logger.info("Generated end-of-sequence token, stopping early.")
                break

        if show_progress:
            logger.info(
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

        logger.info(
            f"🏁 Benchmarking generation: {max_length} tokens, {num_runs} runs each"
        )
        logger.info("=" * 60)

        # Benchmark without cache (naive approach)
        logger.info("🐌 Testing WITHOUT KV-cache (naive)...")
        no_cache_times = []
        for run in range(num_runs):
            start_time = time.time()
            _ = self.generate_greedy(input_ids, max_length, show_progress=False)
            end_time = time.time()
            run_time = end_time - start_time
            no_cache_times.append(run_time)
            logger.info(f"  Run {run + 1}: {run_time:.3f}s")

        # Benchmark with cache (optimized approach)
        logger.info("\n🚀 Testing WITH KV-cache (optimized)...")
        with_cache_times = []
        for run in range(num_runs):
            start_time = time.time()
            _ = self.generate_greedy_with_cache(
                input_ids, max_length, show_progress=False
            )
            end_time = time.time()
            run_time = end_time - start_time
            with_cache_times.append(run_time)
            logger.info(f"  Run {run + 1}: {run_time:.3f}s")

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

        logger.info("\n📊 BENCHMARK RESULTS:")
        logger.info("=" * 60)
        logger.info(
            f"Without KV-cache: {avg_no_cache:.3f}s ({time_per_token_no_cache:.3f}s/token)"
        )
        logger.info(
            f"With KV-cache:    {avg_with_cache:.3f}s ({time_per_token_with_cache:.3f}s/token)"
        )
        logger.info(f"Speedup:          {speedup:.1f}x faster! 🚀")
        logger.info("=" * 60)

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
            logger.info(
                f"Starting sampling generation with {original_length} input tokens..."
            )
            logger.info(f"Parameters: temp={temperature}, top_k={top_k}, top_p={top_p}")

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

                logger.info(f"Step {step + 1}:")
                logger.info(f"  Generated token: {next_token_id} -> '{token_text}'")
                logger.info(f"  Top token prob: {sampling_info['top_token_prob']:.3f}")
                logger.info(
                    f"  Effective vocab: {sampling_info['effective_vocab_size']}"
                )
                logger.info(f"  Entropy change: {sampling_info['entropy_change']:.3f}")

            # Check for end-of-sequence token
            if (
                hasattr(self.model.config, "eos_token_id")
                and next_token_id == self.model.config.eos_token_id
            ):
                if show_progress:
                    logger.info("Generated end-of-sequence token, stopping early.")
                break

        if show_progress:
            logger.info(
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

        logger.info(f"🎭 Comparing sampling strategies on: '{prompt_text}'")
        logger.info("=" * 60)

        for strategy_name, params in strategies.items():
            logger.info(f"\n🎯 {strategy_name.upper()}: {params}")
            strategy_outputs = []

            for i in range(num_samples):
                generated_tokens = self.generate_with_sampling(
                    input_ids, max_length, use_cache=True, **params
                )
                full_text = tokenizer.decode(generated_tokens)
                generated_part = full_text[len(prompt_text) :]
                strategy_outputs.append(generated_part)

                logger.info(f"  {i + 1}: '{generated_part}'")

            results[strategy_name] = strategy_outputs

        return results

    def generate_batch_with_sampling(
        self,
        batch_input_ids: List[torch.Tensor],  # List of input tensors, one per request
        max_length: int = 20,  # Maximum number of NEW tokens to generate
        temperature: float = 1.0,  # Controls randomness (1.0 = normal, <1 = focused, >1 = creative)
        top_k: Optional[int] = None,  # Limit sampling to top-k most likely tokens
        top_p: Optional[
            float
        ] = None,  # Nucleus sampling: cumulative probability threshold
        use_cache: bool = True,  # Enable KV-cache for massive speedup
        show_progress: bool = False,  # Print detailed progress information
    ) -> List[List[int]]:
        """
        CORE BATCHED GENERATION: Process multiple requests simultaneously in a single model forward pass.

        This is the heart of prefill batching (Part 7) - instead of processing requests one-by-one,
        we batch them together to maximize GPU utilization and achieve 4x+ throughput improvements.

        🎯 KEY CONCEPTS:
        - PREFILL BATCHING: Process multiple prompts together in parallel
        - KV-CACHE OPTIMIZATION: Reuse attention computations from previous tokens
        - DYNAMIC SEQUENCE MANAGEMENT: Handle sequences that finish at different times
        - MEMORY EFFICIENCY: Shared model weights, efficient tensor operations

        Args:
            batch_input_ids: List of tokenized input sequences, e.g., [tensor([1,2,3]), tensor([4,5])]
            max_length: Number of NEW tokens to generate (not total sequence length)
            temperature: Sampling randomness - 0.1=very focused, 1.0=normal, 2.0=very creative
            top_k: Keep only the k most likely tokens for sampling (None = disabled)
            top_p: Nucleus sampling - keep tokens until cumulative probability reaches p (None = disabled)
            use_cache: Enable KV-cache for 10x+ speedup by reusing attention computations
            show_progress: Print step-by-step generation details for debugging

        Returns:
            List of complete token sequences, one per input request
            Example: [[1,2,3,15,22,8], [4,5,12,19,7]] for 2 input sequences
        """
        # === SAFETY CHECKS ===
        # Ensure model is loaded before attempting generation
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Handle empty input gracefully
        if not batch_input_ids:
            return []

        # Calculate batch size for tracking and progress reporting
        batch_size = len(batch_input_ids)
        if show_progress:
            logger.info(f"🚀 Starting batched generation for {batch_size} requests")

        # === DATA PREPARATION ===
        # Convert tensor inputs to Python lists for easier manipulation during generation
        # We'll build up these sequences token by token as we generate
        batch_sequences = []  # Will contain: [[prompt1_tokens...], [prompt2_tokens...], ...]
        original_lengths = []  # Track original prompt lengths to calculate new tokens generated

        # Process each input sequence (one per request in the batch)
        for input_ids in batch_input_ids:
            # Convert from tensor shape (1, seq_len) to list [token1, token2, ...]
            sequence = (
                input_ids.squeeze().tolist()
            )  # Remove batch dimension, convert to Python list
            batch_sequences.append(sequence)  # Add to our working batch
            original_lengths.append(
                len(sequence)
            )  # Remember original length for metrics

        # === DYNAMIC SEQUENCE TRACKING ===
        # Track which sequences are still actively generating (haven't finished yet)
        # Initially all sequences are active: [0, 1, 2, 3] for batch_size=4
        # As sequences finish (hit EOS or max length), we remove them from this list
        active_sequences = list(range(batch_size))

        # === KV-CACHE INITIALIZATION ===
        # Initialize the key-value cache to None - will be populated after first forward pass
        # KV-cache stores attention keys & values to avoid recomputing them for previous tokens
        # This is THE optimization that makes autoregressive generation fast
        past_key_values = None

        # === MAIN GENERATION LOOP ===
        # Generate one token at a time, up to max_length new tokens
        for step in range(max_length):
            # === EARLY TERMINATION CHECK ===
            # If all sequences have finished (hit EOS token), stop generation early
            # This saves computation when all requests complete before max_length
            if not active_sequences:
                break  # All sequences finished - no more work to do!

            # === PROGRESS REPORTING ===
            # Print progress every 5 steps to track generation without overwhelming output
            if show_progress and step % 5 == 0:
                logger.info(
                    f"  Step {step}/{max_length}, {len(active_sequences)} active sequences"
                )

            # === BATCH PREPARATION FOR CURRENT STEP ===
            # Prepare input for this generation step - only include sequences still generating
            current_batch_ids = []  # Will contain the input tokens for this step
            current_indices = []  # Maps batch position to original sequence index

            # Build batch containing only active sequences
            for i, seq_idx in enumerate(active_sequences):
                # For step > 0: we only need the LAST token from each sequence (incremental generation)
                # For step = 0: we'll handle full sequences differently below
                current_batch_ids.append(
                    [batch_sequences[seq_idx][-1]]
                )  # Get last token as list
                current_indices.append(
                    seq_idx
                )  # Remember which original sequence this corresponds to

            # === TENSOR CREATION ===
            # Convert our batch of token IDs to a PyTorch tensor for model input
            # Shape will be: (num_active_sequences, 1) for steps > 0
            # dtype=long for token IDs, device=self.device for GPU/CPU consistency
            batch_tensor = torch.tensor(
                current_batch_ids, dtype=torch.long, device=self.device
            )

            # === MODEL FORWARD PASS ===
            # Run the transformer model on our batch - this is where the magic happens!
            # torch.no_grad() disables gradient computation for inference (saves memory & speed)
            with torch.no_grad():
                # === DEBUG LOGGING ===
                # Print detailed tensor shapes every 10 steps for debugging
                if show_progress and step % 10 == 0:
                    logger.info(
                        f"    Step {step}: Processing {len(active_sequences)} sequences"
                    )
                    logger.info(f"    batch_tensor shape: {batch_tensor.shape}")
                    if past_key_values is not None:
                        logger.info(
                            f"    KV-cache shape: {past_key_values[0][0].shape}"
                        )

                # === STEP 0: PREFILL PHASE ===
                # First step is special - we process the FULL input sequences to populate KV-cache
                # This is called "prefill" because we're pre-filling the attention cache
                if step == 0:
                    # === PREFILL: PROCESS FULL INPUT SEQUENCES ===
                    # In the first step, we need to process ALL tokens from each input sequence
                    # to populate the KV-cache with attention states for every input token

                    full_batch_ids = []  # Will contain padded sequences: [[pad,pad,1,2,3], [4,5,6,7,8]]

                    # Find longest sequence to determine padding target
                    # All sequences must be same length for efficient batched matrix operations
                    max_seq_len = max(len(seq) for seq in batch_sequences)

                    # === SEQUENCE PADDING ===
                    # Pad all sequences to the same length for batched processing
                    for seq_idx in active_sequences:
                        seq = batch_sequences[
                            seq_idx
                        ].copy()  # Don't modify original sequence

                        # LEFT-PAD with zeros (GPT-2's pad token is 0)
                        # Left padding is important because we want attention to focus on the END
                        # Example: [1,2,3] becomes [0,0,0,0,1,2,3] if max_len=7
                        while len(seq) < max_seq_len:
                            seq = [0] + seq  # Add padding to the LEFT (beginning)

                        full_batch_ids.append(seq)  # Add padded sequence to batch

                    # === CONVERT TO TENSOR ===
                    # Create tensor for full sequences: shape (batch_size, max_seq_len)
                    batch_tensor = torch.tensor(
                        full_batch_ids, dtype=torch.long, device=self.device
                    )

                    # === ATTENTION MASK CREATION ===
                    # Tell the model which tokens are real vs padding
                    # 1 = real token (attend to this), 0 = padding (ignore this)
                    attention_mask = torch.zeros_like(
                        batch_tensor
                    )  # Start with all zeros (all padding)

                    for i, seq_idx in enumerate(active_sequences):
                        original_len = len(
                            batch_sequences[seq_idx]
                        )  # Length of real (non-padded) sequence
                        # Set the LAST original_len positions to 1 (real tokens at the end due to left padding)
                        attention_mask[i, -original_len:] = 1

                    # === PREFILL FORWARD PASS ===
                    # Run the full transformer forward pass on all input tokens
                    # This populates the KV-cache for efficient subsequent generation
                    outputs = self.model(
                        batch_tensor,  # Input tokens: (batch_size, seq_len)
                        attention_mask=attention_mask,  # Mask: (batch_size, seq_len)
                        past_key_values=past_key_values,  # None for first step
                        use_cache=use_cache,  # True - we want to cache K,V for speed
                    )
                else:
                    # === STEP N>0: INCREMENTAL DECODE PHASE ===
                    # For subsequent steps, we only process the NEXT token being generated
                    # The KV-cache contains all previous attention states, so we're very efficient

                    # === CRITICAL KV-CACHE MANAGEMENT ===
                    # Problem: Some sequences may have finished, so active_sequences < original batch_size
                    # Solution: Slice the KV-cache to match only the currently active sequences
                    if (
                        past_key_values
                        is not None  # We have a cache from previous steps
                        and len(active_sequences)
                        < batch_size  # Some sequences have finished
                    ):
                        try:
                            # === KV-CACHE STRUCTURE EXPLANATION ===
                            # past_key_values is a tuple of tuples, one per transformer layer:
                            # (
                            #   (key_layer_0, value_layer_0),    # Layer 0 cache
                            #   (key_layer_1, value_layer_1),    # Layer 1 cache
                            #   ...                               # 12 layers for GPT-2
                            # )
                            # Each key/value tensor has shape: (batch_size, num_heads, seq_len, head_dim)
                            #                                  (     4    ,    12   ,   50    ,   64   )

                            sliced_past_key_values = []  # Will contain cache for only active sequences

                            # Process each transformer layer's cache
                            for layer_idx, layer_cache in enumerate(past_key_values):
                                k_cache, v_cache = (
                                    layer_cache  # Extract key and value cache for this layer
                                )

                                # === SAFETY CHECK ===
                                # Ensure all indices in active_sequences are valid for the cache
                                if max(active_sequences) >= k_cache.shape[0]:
                                    raise ValueError(
                                        f"Invalid sequence index in active_sequences: {active_sequences}, "
                                        f"KV-cache batch size: {k_cache.shape[0]}"
                                    )

                                # === SLICE CACHE TO ACTIVE SEQUENCES ONLY ===
                                # Use fancy indexing to keep only rows corresponding to active sequences
                                # Example: if active_sequences=[0,2], keep batch positions 0 and 2
                                sliced_k = k_cache[
                                    active_sequences
                                ]  # Shape: (len(active_sequences), num_heads, seq_len, head_dim)
                                sliced_v = v_cache[
                                    active_sequences
                                ]  # Shape: (len(active_sequences), num_heads, seq_len, head_dim)

                                sliced_past_key_values.append(
                                    (sliced_k, sliced_v)
                                )  # Store sliced cache for this layer

                            # Replace the full cache with our sliced version
                            past_key_values = tuple(sliced_past_key_values)

                            if show_progress:
                                logger.info(
                                    f"    Sliced KV-cache from {batch_size} to {len(active_sequences)} sequences"
                                )

                        except Exception as e:
                            # === ERROR HANDLING ===
                            # If KV-cache slicing fails, provide detailed debugging information
                            logger.error(f"❌ KV-cache slicing error at step {step}:")
                            logger.error(f"   active_sequences: {active_sequences}")
                            logger.error(f"   batch_size: {batch_size}")
                            if past_key_values:
                                logger.error(
                                    f"   KV-cache shape: {past_key_values[0][0].shape}"
                                )
                            logger.error(f"   Error: {e}")
                            raise e

                    # === INCREMENTAL FORWARD PASS ===
                    # Process only the new tokens (1 token per sequence) using cached attention
                    # This is MUCH faster than reprocessing the entire sequence
                    outputs = self.model(
                        batch_tensor,  # New tokens only: shape (num_active, 1)
                        past_key_values=past_key_values,  # Cached attention from previous tokens
                        use_cache=use_cache,  # Continue building the cache
                    )

                # === EXTRACT MODEL OUTPUT ===
                # Get the logits (raw scores) from the model output
                # Shape: (num_active_sequences, seq_len, vocab_size)
                # For step 0: seq_len = max_seq_len (full sequences)
                # For step N>0: seq_len = 1 (just the new token)
                logits = outputs.logits

                # === UPDATE KV-CACHE ===
                # Store the updated key-value cache for the next generation step
                # This cache now includes attention states for the tokens we just processed
                if use_cache:
                    past_key_values = outputs.past_key_values

            # === TOKEN SAMPLING PHASE ===
            # Now we need to sample the next token for each active sequence
            next_tokens = []  # Will store the sampled token IDs
            sequences_to_remove = []  # Track sequences that finish this step

            # Process each sequence in the current batch
            for i, seq_idx in enumerate(current_indices):
                # === EXTRACT LOGITS FOR NEXT TOKEN PREDICTION ===
                # Get logits for the LAST position of this sequence (where we predict next token)
                # Shape: (vocab_size,) - one score for each possible token in vocabulary
                sequence_logits = logits[i, -1, :]

                # === ADVANCED SAMPLING ===
                # Use sophisticated sampling strategies for creative and controlled generation
                # This is where temperature, top-k, and nucleus (top-p) sampling happen
                next_token_id = sample_token(
                    sequence_logits,  # Raw scores for all possible tokens
                    temperature=temperature,  # Controls randomness: <1.0=focused, >1.0=creative
                    top_k=top_k,  # Keep only top-k most likely tokens (None=disabled)
                    top_p=top_p,  # Nucleus sampling: keep tokens until cumulative prob reaches p
                )

                # === UPDATE SEQUENCE ===
                # Add the newly sampled token to this sequence's growing token list
                batch_sequences[seq_idx].append(next_token_id)
                next_tokens.append(next_token_id)  # Track for debugging

                # === SEQUENCE COMPLETION CHECK ===
                # Check if this sequence should stop generating (hit end-of-sequence token)
                if (
                    hasattr(
                        self.model.config, "eos_token_id"
                    )  # Model has an EOS token defined
                    and next_token_id
                    == self.model.config.eos_token_id  # We just generated EOS
                ):
                    sequences_to_remove.append(
                        seq_idx
                    )  # Mark this sequence as finished
                    if show_progress:
                        logger.info(f"    Sequence {seq_idx} finished (EOS token)")

            # === DYNAMIC BATCH MANAGEMENT ===
            # Remove sequences that finished this step from the active list
            # This is key to continuous batching - the batch composition changes over time
            for seq_idx in sequences_to_remove:
                active_sequences.remove(seq_idx)  # No longer process this sequence

        # === GENERATION COMPLETE ===
        # All sequences have either finished (hit EOS) or reached max_length

        # === FINAL STATISTICS ===
        # Calculate and display generation statistics for educational purposes
        if show_progress:
            # Count total new tokens generated across all sequences
            total_generated = sum(
                len(seq) - orig_len  # Current length minus original prompt length
                for seq, orig_len in zip(batch_sequences, original_lengths)
            )
            logger.info(
                f"🎉 Batched generation complete! Generated {total_generated} total tokens"
            )

        # === RETURN RESULTS ===
        # Return the complete sequences (original prompt + generated tokens)
        # Each sequence in batch_sequences is now: [prompt_tokens... + generated_tokens...]
        return batch_sequences

    def generate_continuous_step(
        self,
        active_requests: List,
        past_key_values: Optional[Any] = None,
        show_progress: bool = False,
    ) -> tuple:
        """
        Generate the next token for all active requests in continuous batching.

        This is the core of continuous batching - generating one token at a time
        for a dynamic batch of requests that can change composition between steps.

        Args:
            active_requests: List of ContinuousRequest objects currently active
            past_key_values: Cached key-value pairs from previous steps
            show_progress: Whether to print progress information

        Returns:
            Tuple of (next_tokens, updated_past_key_values, request_finished_flags)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not active_requests:
            return [], None, []

        batch_size = len(active_requests)
        if show_progress:
            logger.info(
                f"🔄 Continuous generation step for {batch_size} active requests"
            )

        # Prepare input for current step
        if past_key_values is None:
            # First step: process full input sequences (prefill)
            max_input_len = max(len(req.input_tokens) for req in active_requests)

            # Pad all input sequences to same length
            batch_input_ids = []
            attention_mask = []

            for req in active_requests:
                # Pad sequence to max length (left padding for GPT-2)
                padded_seq = req.input_tokens.copy()
                padding_needed = max_input_len - len(padded_seq)
                padded_seq = [0] * padding_needed + padded_seq

                # Create attention mask (0 for padding, 1 for real tokens)
                mask = [0] * padding_needed + [1] * len(req.input_tokens)

                batch_input_ids.append(padded_seq)
                attention_mask.append(mask)

            input_tensor = torch.tensor(
                batch_input_ids, dtype=torch.long, device=self.device
            )
            attention_tensor = torch.tensor(
                attention_mask, dtype=torch.long, device=self.device
            )

            if show_progress:
                logger.info(f"  Prefill step: input shape {input_tensor.shape}")

        else:
            # Subsequent steps: just process last generated token for each request
            last_tokens = []
            for req in active_requests:
                if req.generated_tokens:
                    last_tokens.append([req.generated_tokens[-1]])
                else:
                    # This shouldn't happen in continuous batching, but handle gracefully
                    last_tokens.append([req.input_tokens[-1]])

            input_tensor = torch.tensor(
                last_tokens, dtype=torch.long, device=self.device
            )
            attention_tensor = None  # Not needed for subsequent steps with cache

            if show_progress:
                logger.info(f"  Generation step: input shape {input_tensor.shape}")

        # Forward pass through model
        with torch.no_grad():
            if attention_tensor is not None:
                # Prefill step with attention mask
                outputs = self.model(
                    input_tensor,
                    attention_mask=attention_tensor,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                # Generation step with cache
                outputs = self.model(
                    input_tensor,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
            new_past_key_values = outputs.past_key_values

        # Sample next tokens for each request
        next_tokens = []
        finished_flags = []

        for i, req in enumerate(active_requests):
            # Get logits for the last position of this sequence
            sequence_logits = logits[i, -1, :]  # Shape: (vocab_size,)

            # Sample next token using request's sampling parameters
            next_token_id = sample_token(
                sequence_logits,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            )

            next_tokens.append(next_token_id)

            # Add token to request's generated sequence
            req.generated_tokens.append(next_token_id)
            req.generation_step += 1

            # Check if this request should finish
            is_finished = len(req.generated_tokens) >= req.max_tokens or (
                hasattr(self.model.config, "eos_token_id")
                and next_token_id == self.model.config.eos_token_id
            )

            finished_flags.append(is_finished)

            if show_progress and is_finished:
                logger.info(
                    f"    Request {req.id} finished after {len(req.generated_tokens)} tokens"
                )

        return next_tokens, new_past_key_values, finished_flags
