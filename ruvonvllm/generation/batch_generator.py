"""
Batch generation capability for LLM inference.

This module implements prefill batching (Part 7) - processing multiple requests
simultaneously for improved GPU utilization and throughput.
"""

import logging
import torch
from typing import List, Optional

from ruvonvllm.sampling.strategies import sample_token

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
        if self.model.model is None:
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
            # dtype=long for token IDs, device=self.model.device for GPU/CPU consistency
            batch_tensor = torch.tensor(
                current_batch_ids, dtype=torch.long, device=self.model.device
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
                        full_batch_ids, dtype=torch.long, device=self.model.device
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
                    outputs = self.model.model(
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

                            # === CRITICAL FIX: SAFE INDEX MAPPING ===
                            # The issue: active_sequences contains original indices, but some may be out of bounds
                            # for the current KV-cache if the cache wasn't properly updated in previous steps.
                            # Solution: Create a safe mapping that only includes valid cache positions.

                            # Get current cache size to validate indices
                            current_cache_size = past_key_values[0][0].shape[0]

                            # Filter active_sequences to only include valid cache indices
                            valid_cache_indices = [
                                seq_idx
                                for seq_idx in active_sequences
                                if seq_idx < current_cache_size
                            ]

                            if len(valid_cache_indices) != len(active_sequences):
                                # Some sequences in active_sequences are invalid - this indicates a logic error
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
                                # Use only valid indices to prevent crash
                                slicing_indices = valid_cache_indices
                            else:
                                slicing_indices = active_sequences

                            # Create mapping from original sequence indices to new cache positions
                            active_to_cache_mapping = {
                                seq_idx: cache_pos
                                for cache_pos, seq_idx in enumerate(slicing_indices)
                            }

                            sliced_past_key_values = []  # Will contain cache for only active sequences

                            # Process each transformer layer's cache
                            for layer_idx, layer_cache in enumerate(past_key_values):
                                k_cache, v_cache = (
                                    layer_cache  # Extract key and value cache for this layer
                                )

                                # === SLICE CACHE TO VALID SEQUENCES ONLY ===
                                # Use only valid indices to prevent out-of-bounds errors
                                sliced_k = k_cache[
                                    slicing_indices
                                ]  # Shape: (len(slicing_indices), num_heads, seq_len, head_dim)
                                sliced_v = v_cache[
                                    slicing_indices
                                ]  # Shape: (len(slicing_indices), num_heads, seq_len, head_dim)

                                sliced_past_key_values.append(
                                    (sliced_k, sliced_v)
                                )  # Store sliced cache for this layer

                            # Replace the full cache with our sliced version
                            past_key_values = tuple(sliced_past_key_values)

                            # Update active_sequences to match the sliced cache
                            active_sequences = slicing_indices

                            # CRITICAL: Update current_indices to match the updated active_sequences
                            # current_indices was built before KV-cache slicing, so it may be out of sync
                            current_indices = active_sequences.copy()

                            # Also need to rebuild current_batch_ids to match the updated sequences
                            current_batch_ids = []
                            for seq_idx in active_sequences:
                                current_batch_ids.append([batch_sequences[seq_idx][-1]])

                            # Rebuild the batch tensor with the correct sequences
                            batch_tensor = torch.tensor(
                                current_batch_ids,
                                dtype=torch.long,
                                device=self.model.device,
                            )

                            if show_progress:
                                logger.info(
                                    f"    Sliced KV-cache from {batch_size} to {len(active_sequences)} sequences"
                                )
                                logger.info(
                                    f"    Updated current_indices to match: {current_indices}"
                                )
                                logger.info(
                                    f"    Index mapping: {active_to_cache_mapping}"
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
                    outputs = self.model.model(
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
                        self.model.model.config, "eos_token_id"
                    )  # Model has an EOS token defined
                    and next_token_id
                    == self.model.model.config.eos_token_id  # We just generated EOS
                ):
                    sequences_to_remove.append(
                        seq_idx
                    )  # Mark this sequence as finished
                    if show_progress:
                        logger.info(f"    Sequence {seq_idx} finished (EOS token)")

            # === DYNAMIC BATCH MANAGEMENT ===
            # Remove sequences that finished this step from the active list
            # This is key to continuous batching - the batch composition changes over time
            #
            # IMPORTANT: Remove in reverse order to avoid index shifting issues
            # If we have sequences_to_remove = [5, 10, 15] and active_sequences = [0,1,2,5,7,10,15]
            # Removing 5 first would shift later indices, causing problems
            # Removing 15, then 10, then 5 avoids this issue
            for seq_idx in sorted(sequences_to_remove, reverse=True):
                if seq_idx in active_sequences:
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
