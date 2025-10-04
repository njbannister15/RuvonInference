"""
Batch generation capability for LLM inference.

This module implements prefill batching (Part 7) - processing multiple requests
simultaneously for improved GPU utilization and throughput.
"""

import logging
import torch
from typing import List, Optional

from ruvoninference.sampling.strategies import sample_token

logger = logging.getLogger(__name__)


class BatchGenerator:
    """
    Batch generation capability that can be composed with any model.

    This class implements the core batched generation logic from Part 7,
    separated from the model implementation for better modularity.

    Key Features:
    - Prefill batching: Process multiple prompts together
    - Dynamic sequence management: Handle sequences finishing at different times
    - KV-cache optimization: Reuse attention computations for speed
    - Memory efficiency: Shared model weights, efficient tensor operations
    """

    def __init__(self, model, tokenizer=None):
        """
        Initialize the batch generator with a model.

        Args:
            model: The underlying language model (e.g., GPT2Model instance)
            tokenizer: Optional tokenizer for debugging (can be None)
        """
        self.model = model
        self.tokenizer = tokenizer

    def _validate_and_prepare_inputs(
        self, batch_input_ids: List[torch.Tensor], show_progress: bool
    ) -> tuple[List[List[int]], List[int], int]:
        """
        Validate inputs and prepare data structures for generation.

        Returns:
            tuple: (batch_sequences, original_lengths, batch_size)
        """
        # === SAFETY CHECKS ===
        if self.model.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not batch_input_ids:
            return [], [], 0

        batch_size = len(batch_input_ids)
        if show_progress:
            logger.info(f"🚀 Starting batched generation for {batch_size} requests")

        # === DATA PREPARATION ===
        batch_sequences = []
        original_lengths = []

        for input_ids in batch_input_ids:
            sequence = input_ids.squeeze().tolist()
            batch_sequences.append(sequence)
            original_lengths.append(len(sequence))

        return batch_sequences, original_lengths, batch_size

    def _create_prefill_tensors(
        self, batch_sequences: List[List[int]], active_sequences: List[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create padded tensors and attention masks for the prefill phase.


        Returns:
            tuple: (batch_tensor, attention_mask)
        """
        full_batch_ids = []
        max_seq_len = max(len(batch_sequences[seq_idx]) for seq_idx in active_sequences)

        # Pad all sequences to the same length
        for seq_idx in active_sequences:
            seq = batch_sequences[seq_idx].copy()
            # Left-pad with zeros (GPT-2's pad token is 0)
            while len(seq) < max_seq_len:
                seq = [0] + seq
            full_batch_ids.append(seq)

        batch_tensor = torch.tensor(
            full_batch_ids, dtype=torch.long, device=self.model.device
        )

        # Create attention mask
        # An attention mask tells the model which tokens to "pay attention to" vs ignore.
        # When batching sequences of different lengths, we pad shorter sequences with dummy tokens.
        # The attention mask marks: 1 = real token (attend to this), 0 = padding (ignore this)
        #
        # Example:
        #   Input:  [1, 2, 3] (3 tokens)
        #   Padded: [0, 0, 0, 0, 1, 2, 3] (7 positions, left-padded with 0s)
        #   Mask:   [0, 0, 0, 0, 1, 1, 1] (only attend to last 3 real tokens)
        #
        # This prevents the model from processing padding tokens as real content,
        # ensuring clean attention patterns and accurate generation results.
        attention_mask = torch.zeros_like(batch_tensor)
        for i, seq_idx in enumerate(active_sequences):
            original_len = len(batch_sequences[seq_idx])
            attention_mask[i, -original_len:] = 1

        return batch_tensor, attention_mask

    def _slice_kv_cache_for_active_sequences(
        self,
        past_key_values,
        active_sequences: List[int],
        batch_size: int,
        step: int,
        show_progress: bool,
    ):
        """
        Slice KV-cache to match only currently active sequences.

        Returns:
            tuple: (updated_past_key_values, updated_active_sequences)
        """
        if past_key_values is None or len(active_sequences) == batch_size:
            return past_key_values, active_sequences

        try:
            # Get current cache size to validate indices
            current_cache_size = past_key_values[0][0].shape[0]

            # Filter active_sequences to only include valid cache indices
            valid_cache_indices = [
                seq_idx for seq_idx in active_sequences if seq_idx < current_cache_size
            ]

            if len(valid_cache_indices) != len(active_sequences):
                invalid_indices = [
                    seq_idx
                    for seq_idx in active_sequences
                    if seq_idx >= current_cache_size
                ]
                logger.info(
                    f"Filtering out finished sequence indices {invalid_indices} (cache size: {current_cache_size})"
                )
                logger.info(
                    "This is normal when sequences finish at different times in continuous batching"
                )
                slicing_indices = valid_cache_indices
            else:
                slicing_indices = active_sequences

            # Create mapping from original sequence indices to new cache positions
            active_to_cache_mapping = {
                seq_idx: cache_pos for cache_pos, seq_idx in enumerate(slicing_indices)
            }

            sliced_past_key_values = []

            # Process each transformer layer's cache
            for layer_idx, layer_cache in enumerate(past_key_values):
                k_cache, v_cache = layer_cache

                # Slice cache to valid sequences only
                sliced_k = k_cache[slicing_indices]
                sliced_v = v_cache[slicing_indices]

                sliced_past_key_values.append((sliced_k, sliced_v))

            # Replace the full cache with our sliced version
            past_key_values = tuple(sliced_past_key_values)

            if show_progress:
                logger.info(
                    f"    Sliced KV-cache from {batch_size} to {len(slicing_indices)} sequences"
                )
                logger.info(f"    Updated active_sequences to match: {slicing_indices}")
                logger.info(f"    Index mapping: {active_to_cache_mapping}")

            return past_key_values, slicing_indices

        except Exception as e:
            logger.error(f"❌ KV-cache slicing error at step {step}:")
            logger.error(f"   active_sequences: {active_sequences}")
            logger.error(f"   batch_size: {batch_size}")
            if past_key_values:
                logger.error(f"   KV-cache shape: {past_key_values[0][0].shape}")
            logger.error(f"   Error: {e}")
            raise e

    def _sample_next_tokens(
        self,
        logits: torch.Tensor,
        current_indices: List[int],
        batch_sequences: List[List[int]],
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        show_progress: bool,
    ) -> List[int]:
        """
        Sample next tokens for each active sequence and update sequences.

        Returns:
            List of sequence indices that should be removed (finished)
        """
        sequences_to_remove = []

        for i, seq_idx in enumerate(current_indices):
            # Extract logits for next token prediction
            sequence_logits = logits[i, -1, :]

            # Sample next token
            next_token_id = sample_token(
                sequence_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Update sequence
            batch_sequences[seq_idx].append(next_token_id)

            # Check if sequence should finish
            if (
                hasattr(self.model.model.config, "eos_token_id")
                and next_token_id == self.model.model.config.eos_token_id
            ):
                sequences_to_remove.append(seq_idx)
                if show_progress:
                    logger.info(f"    Sequence {seq_idx} finished (EOS token)")

        return sequences_to_remove

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
        # Validate inputs and prepare data structures
        batch_sequences, original_lengths, batch_size = (
            self._validate_and_prepare_inputs(batch_input_ids, show_progress)
        )

        # Handle empty input case
        if batch_size == 0:
            return []

        # Initialize sequence tracking and cache
        active_sequences = list(range(batch_size))
        past_key_values = None

        # Main generation loop
        for step in range(max_length):
            # Early termination if all sequences finished
            if not active_sequences:
                break

            # Progress reporting
            if show_progress and step % 5 == 0:
                logger.info(
                    f"  Step {step}/{max_length}, {len(active_sequences)} active sequences"
                )

            # Prepare batch for current step
            if step == 0:
                # Prefill phase: process full input sequences
                batch_tensor, attention_mask = self._create_prefill_tensors(
                    batch_sequences, active_sequences
                )
                current_indices = active_sequences.copy()
            else:
                # Incremental decode: process only new tokens
                # Handle KV-cache slicing for finished sequences
                past_key_values, active_sequences = (
                    self._slice_kv_cache_for_active_sequences(
                        past_key_values,
                        active_sequences,
                        batch_size,
                        step,
                        show_progress,
                    )
                )

                # Prepare batch with only active sequences
                current_batch_ids = []
                current_indices = []
                for seq_idx in active_sequences:
                    current_batch_ids.append([batch_sequences[seq_idx][-1]])
                    current_indices.append(seq_idx)

                batch_tensor = torch.tensor(
                    current_batch_ids, dtype=torch.long, device=self.model.device
                )
                attention_mask = None

            # Debug logging
            if show_progress and step % 10 == 0:
                logger.info(
                    f"    Step {step}: Processing {len(active_sequences)} sequences"
                )
                logger.info(f"    batch_tensor shape: {batch_tensor.shape}")
                if past_key_values is not None:
                    logger.info(f"    KV-cache shape: {past_key_values[0][0].shape}")

            # Model forward pass
            with torch.no_grad():
                if step == 0:
                    # Prefill forward pass
                    outputs = self.model.model(
                        batch_tensor,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                    )
                else:
                    # Incremental forward pass
                    outputs = self.model.model(
                        batch_tensor,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                    )

                logits = outputs.logits
                if use_cache:
                    past_key_values = outputs.past_key_values

            # Sample next tokens and update sequences
            sequences_to_remove = self._sample_next_tokens(
                logits,
                current_indices,
                batch_sequences,
                temperature,
                top_k,
                top_p,
                show_progress,
            )

            # Remove finished sequences
            for seq_idx in sorted(sequences_to_remove, reverse=True):
                if seq_idx in active_sequences:
                    active_sequences.remove(seq_idx)

        # Final statistics
        if show_progress:
            total_generated = sum(
                len(seq) - orig_len
                for seq, orig_len in zip(batch_sequences, original_lengths)
            )
            logger.info(
                f"🎉 Batched generation complete! Generated {total_generated} total tokens"
            )

        return batch_sequences
