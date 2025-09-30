"""
Sampling strategies for text generation.

This module implements various sampling methods to make LLM generation more creative
and diverse, moving beyond simple greedy decoding to probabilistic approaches.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply temperature scaling to logits.

    Temperature controls the "creativity" of the model:
    - temperature = 1.0: Use raw model probabilities (no change)
    - temperature < 1.0: Make the model more confident (sharper distribution)
    - temperature > 1.0: Make the model more random (flatter distribution)
    - temperature → 0: Approaches greedy decoding (almost deterministic)
    - temperature → ∞: Approaches uniform random selection

    Mathematical explanation:
    - Original probability: P(token_i) = exp(logit_i) / Σ(exp(logit_j))
    - With temperature: P(token_i) = exp(logit_i / T) / Σ(exp(logit_j / T))
    - Lower T makes high-probability tokens even more likely
    - Higher T flattens the distribution, giving low-probability tokens more chance

    Args:
        logits: Raw model logits with shape [vocab_size]
        temperature: Temperature scaling factor (> 0)

    Returns:
        Temperature-scaled logits
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Apply top-k filtering to logits.

    Top-k sampling restricts the model to only consider the k most likely tokens
    at each step. This prevents the model from selecting very unlikely tokens
    while still allowing some randomness among the top choices.

    How it works:
    1. Find the k highest logits
    2. Set all other logits to -inf (probability 0)
    3. Sample from the filtered distribution

    Args:
        logits: Raw model logits with shape [vocab_size]
        k: Number of top tokens to keep (1 = greedy, vocab_size = no filtering)

    Returns:
        Filtered logits with only top-k tokens possible
    """
    if k <= 0:
        raise ValueError("k must be positive")

    if k >= logits.size(-1):
        return logits  # No filtering needed

    # Get the k-th largest logit value (threshold)
    top_k_logits, _ = torch.topk(logits, k)
    min_top_k = top_k_logits[..., -1, None]  # k-th largest value

    # Set all logits below threshold to -inf
    return torch.where(logits < min_top_k, torch.tensor(float("-inf")), logits)


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Apply nucleus (top-p) sampling to logits.

    Top-p sampling dynamically selects the smallest set of tokens whose
    cumulative probability exceeds p. This adapts to the model's confidence:
    - When the model is confident: few tokens are selected
    - When the model is uncertain: more tokens are selected

    How it works:
    1. Convert logits to probabilities
    2. Sort probabilities in descending order
    3. Find the cutoff where cumulative probability exceeds p
    4. Keep only tokens above the cutoff

    Example with p=0.9:
    - If top token has 95% probability → only select that token
    - If top 5 tokens have 20% each → select all 5 tokens

    Args:
        logits: Raw model logits with shape [vocab_size]
        p: Cumulative probability threshold (0 < p <= 1)

    Returns:
        Filtered logits with only nucleus tokens possible
    """
    if not 0 < p <= 1:
        raise ValueError("p must be between 0 and 1")

    if p >= 1.0:
        return logits  # No filtering needed

    # Convert to probabilities and sort
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find where cumulative probability exceeds p
    # We keep tokens up to and including the first one that exceeds p
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False  # Always keep the most likely token

    # Create mask for original token order
    indices_to_remove = sorted_indices_to_remove.scatter(
        0, sorted_indices, sorted_indices_to_remove
    )

    # Set filtered tokens to -inf
    filtered_logits = logits.clone()
    filtered_logits[indices_to_remove] = float("-inf")

    return filtered_logits


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> int:
    """
    Sample a token using the specified strategy.

    This function combines multiple sampling techniques in a specific order that matters:
    1. Apply top-k filtering (if specified) - Hard vocabulary limit
    2. Apply top-p filtering (if specified) - Dynamic vocabulary adjustment
    3. Apply temperature scaling - Adjust randomness/creativity
    4. Sample from the resulting distribution

    WHY THE ORDER MATTERS:

    Top-k → Top-p → Temperature is the standard approach because:

    1. TOP-K FIRST: Provides a safety net by removing very unlikely tokens
       - Sets a hard upper bound on vocabulary size
       - Prevents the model from choosing completely nonsensical words
       - Example: With k=50, we'll never consider the 51st most likely token

    2. TOP-P SECOND: Dynamically adjusts based on model confidence
       - Among the top-k tokens, keeps only those needed for cumulative probability p
       - When model is confident (one token has high prob), keeps fewer tokens
       - When model is uncertain (many tokens have similar prob), keeps more tokens
       - Example: If top token has 80% probability, top-p=0.9 might only keep 2-3 tokens
                  If top token has 20% probability, top-p=0.9 might keep 10+ tokens

    3. TEMPERATURE LAST: Fine-tunes the final distribution
       - Applied to the already-filtered set of viable tokens
       - Doesn't change which tokens are considered, only their relative probabilities
       - Lower temp: Make likely tokens even more likely
       - Higher temp: Flatten the distribution among viable tokens

    ALTERNATIVE ORDERS AND WHY WE DON'T USE THEM:

    - Temperature → Top-k → Top-p: BAD because temperature might promote
      unlikely tokens that top-k would later remove anyway

    - Top-p → Top-k → Temperature: LESS EFFICIENT because top-p operates
      on full vocabulary, then top-k narrows it down (redundant work)

    - Top-k + Top-p → Temperature: CURRENT APPROACH (most efficient and effective)

    Args:
        logits: Raw model logits with shape [vocab_size]
        temperature: Temperature for scaling (default: 1.0)
        top_k: Number of top tokens to consider (default: None = no filtering)
        top_p: Cumulative probability threshold (default: None = no filtering)

    Returns:
        Selected token ID
    """
    # Make a copy to avoid modifying the original logits
    filtered_logits = logits.clone()

    # STEP 1: Apply top-k filtering first (if specified)
    # This sets a hard upper bound on vocabulary size by keeping only the
    # k most likely tokens and setting all others to -inf (probability 0)
    if top_k is not None:
        filtered_logits = apply_top_k(filtered_logits, top_k)
        # Result: At most k tokens can be selected (others have -inf logits)

    # STEP 2: Apply top-p (nucleus) filtering second (if specified)
    # This dynamically adjusts vocabulary size based on the cumulative probability
    # distribution of the remaining tokens from step 1
    if top_p is not None:
        filtered_logits = apply_top_p(filtered_logits, top_p)
        # Result: Among the top-k tokens (or all tokens if no top-k),
        # keep only those needed to reach cumulative probability p

    # STEP 3: Apply temperature scaling last
    # This adjusts the "sharpness" of the probability distribution without
    # changing which tokens are eligible for selection
    if temperature != 1.0:
        filtered_logits = apply_temperature(filtered_logits, temperature)
        # Result: Same eligible tokens, but different relative probabilities

    # STEP 4: Convert to probabilities and sample
    probs = F.softmax(filtered_logits, dim=-1)

    # Sample from the distribution
    # multinomial() samples according to the probability distribution
    # Higher probability tokens are more likely to be selected, but any
    # token with non-zero probability could potentially be chosen
    token_id = torch.multinomial(probs, num_samples=1).item()

    return token_id


# CONCRETE EXAMPLE OF WHY ORDER MATTERS:
#
# Imagine we have logits for next word after "The cat sat on the":
# Original distribution (top 10 shown):
#   "mat":     logit=5.0  → prob=35%
#   "floor":   logit=4.5  → prob=20%
#   "chair":   logit=4.0  → prob=15%
#   "table":   logit=3.8  → prob=10%
#   "couch":   logit=3.5  → prob=8%
#   "ground":  logit=3.2  → prob=5%
#   "bed":     logit=3.0  → prob=4%
#   "roof":    logit=2.8  → prob=2%
#   "car":     logit=2.5  → prob=1%
#   "banana":  logit=1.0  → prob=0.01%
#   ... 50,247 other tokens with even lower probabilities
#
# SCENARIO: top_k=5, top_p=0.8, temperature=1.5
#
# STEP 1 - Apply top_k=5:
#   Keep: "mat", "floor", "chair", "table", "couch"
#   Remove: "ground", "bed", "roof", "car", "banana", ... (50,252 tokens set to -inf)
#   Result: 5 possible tokens
#
# STEP 2 - Apply top_p=0.8:
#   Cumulative probabilities among remaining 5 tokens:
#   "mat": 35% (cumulative: 35%)
#   "floor": 20% (cumulative: 55%)
#   "chair": 15% (cumulative: 70%)
#   "table": 10% (cumulative: 80%) ← Reaches 80% threshold!
#   "couch": 8% (cumulative: 88%) ← Would exceed 80%, so remove this
#   Result: 4 possible tokens ("mat", "floor", "chair", "table")
#
# STEP 3 - Apply temperature=1.5:
#   Flatten the distribution among the 4 remaining tokens
#   Higher temperature makes the distribution more uniform
#   Result: 4 possible tokens with more balanced probabilities
#
# FINAL SAMPLING:
#   Sample from 4 tokens with adjusted probabilities
#   Much more controlled than sampling from all 50,257 tokens!
#
# If we did DIFFERENT ORDER (temperature first):
#   1. Temperature=1.5 → All 50,257 tokens get flattened probabilities
#   2. Top-k=5 → Keep top 5 after flattening (might be different tokens!)
#   3. Top-p=0.8 → Apply to those 5 tokens
#   Result: Could end up with completely different final vocabulary!


def get_sampling_info(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> dict:
    """
    Get information about the sampling process for debugging/analysis.

    This function provides insights into how the sampling parameters
    affect the token selection process.

    Args:
        logits: Raw model logits
        temperature: Temperature scaling
        top_k: Top-k filtering
        top_p: Top-p filtering

    Returns:
        Dictionary with sampling statistics
    """
    original_probs = F.softmax(logits, dim=-1)
    original_entropy = -torch.sum(original_probs * torch.log(original_probs + 1e-10))

    # Apply filtering and temperature
    filtered_logits = logits.clone()
    if top_k is not None:
        filtered_logits = apply_top_k(filtered_logits, top_k)
    if top_p is not None:
        filtered_logits = apply_top_p(filtered_logits, top_p)
    if temperature != 1.0:
        filtered_logits = apply_temperature(filtered_logits, temperature)

    final_probs = F.softmax(filtered_logits, dim=-1)
    final_entropy = -torch.sum(final_probs * torch.log(final_probs + 1e-10))

    # Count effective vocabulary size (non-zero probability tokens)
    effective_vocab_size = torch.sum(final_probs > 1e-10).item()

    return {
        "original_entropy": original_entropy.item(),
        "final_entropy": final_entropy.item(),
        "entropy_change": final_entropy.item() - original_entropy.item(),
        "effective_vocab_size": effective_vocab_size,
        "original_vocab_size": logits.size(-1),
        "top_token_prob": torch.max(final_probs).item(),
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }
