# PyTorch Glossary

This document provides definitions of key terms and detailed explanations of common PyTorch functions used in the RuvonVLLM project.

## Key Terminology

### Logit

**Definition**: Raw, unnormalized output values from a neural network's final layer before applying any activation function.

**Characteristics**:
- Can be any real number (positive, negative, or zero)
- Higher values indicate higher confidence/likelihood
- Named after the logistic function (logit is the inverse of the sigmoid function)
- Not bounded and don't sum to any particular value

**Context in Language Models**:
In language models, logits represent the model's raw preference scores for each token in the vocabulary. A logit of 5.2 for token "cat" and 1.8 for token "dog" means the model has a stronger preference for "cat", but these values don't directly represent probabilities until processed through softmax.

**Example**:
```python
# Raw logits from model output layer
logits = torch.tensor([3.2, 1.8, -0.5, 4.1])  # Unbounded values
# Convert to probabilities using softmax
probs = torch.softmax(logits, dim=0)  # [0.1966, 0.0486, 0.0048, 0.4999]
```

### Tensor

**Definition**: A multi-dimensional array that serves as the fundamental data structure in PyTorch, generalizing scalars, vectors, and matrices to arbitrary dimensions.

**Mathematical Hierarchy**:
- **Rank 0**: Scalar (single number) - `torch.tensor(5.0)`
- **Rank 1**: Vector (1D array) - `torch.tensor([1, 2, 3])`
- **Rank 2**: Matrix (2D array) - `torch.tensor([[1, 2], [3, 4]])`
- **Rank 3+**: Higher-dimensional arrays - `torch.tensor([[[1, 2]], [[3, 4]]])`

**Key Properties**:
- **Shape**: Dimensions of the tensor (e.g., `[3, 4, 5]` for a 3D tensor)
- **Dtype**: Data type of elements (float32, int64, bool, etc.)
- **Device**: Where the tensor is stored (CPU, GPU)
- **Requires_grad**: Whether to track gradients for backpropagation

**Context in Language Models**:
- **Input IDs**: `[batch_size, sequence_length]` - token indices
- **Embeddings**: `[batch_size, sequence_length, hidden_size]` - dense representations
- **Attention weights**: `[batch_size, num_heads, seq_len, seq_len]` - attention patterns
- **Logits**: `[batch_size, sequence_length, vocab_size]` - output predictions

**Example**:
```python
# Different tensor ranks in NLP context
input_ids = torch.tensor([1, 15, 23, 45])  # 1D: sequence of token IDs
embeddings = torch.randn(4, 768)  # 2D: [seq_len, embedding_dim]
batch_embeddings = torch.randn(8, 4, 768)  # 3D: [batch, seq_len, embed_dim]
```

## torch.softmax

**Purpose**: Applies the softmax function to convert raw logits into a probability distribution.

**Signature**: `torch.softmax(input, dim, dtype=None)`

**Parameters**:
- `input`: Input tensor containing raw scores/logits
- `dim`: Dimension along which to apply softmax (typically the last dimension for token probabilities)
- `dtype`: Optional data type for the computation

**What it does**:
Converts raw scores into probabilities that sum to 1.0 along the specified dimension. The softmax function is defined as:
```
softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
```

**Common use cases**:
- Converting model logits to token probabilities in language models
- Final layer activation in classification tasks
- Attention weight normalization

**Example**:
```python
import torch

# Raw logits from model output
logits = torch.tensor([2.0, 1.0, 0.1])
# Convert to probabilities
probs = torch.softmax(logits, dim=0)
print(probs)  # tensor([0.6590, 0.2424, 0.0986])
print(probs.sum())  # tensor(1.0000) - probabilities sum to 1
```

## torch.tensor

**Purpose**: Creates a tensor from data (lists, numpy arrays, scalars, etc.).

**Signature**: `torch.tensor(data, dtype=None, device=None, requires_grad=False)`

**Parameters**:
- `data`: Input data (can be list, numpy array, scalar, or another tensor)
- `dtype`: Data type (e.g., torch.float32, torch.int64)
- `device`: Device placement ('cpu', 'cuda', etc.)
- `requires_grad`: Whether to track gradients for automatic differentiation

**What it does**:
Creates a new tensor by copying the provided data. Unlike `torch.as_tensor()`, this always creates a new tensor.

**Common use cases**:
- Converting Python lists/arrays to tensors
- Creating input data for models
- Initializing model parameters or constants

**Example**:
```python
import torch

# From Python list
tensor_from_list = torch.tensor([1, 2, 3, 4])
print(tensor_from_list)  # tensor([1, 2, 3, 4])

# With specific dtype and device
float_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]],
                           dtype=torch.float32,
                           device='cpu')

# For gradient computation
param_tensor = torch.tensor([1.0, 2.0], requires_grad=True)
```

## torch.allclose

**Purpose**: Checks if two tensors are element-wise equal within a tolerance.

**Signature**: `torch.allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False)`

**Parameters**:
- `input`: First tensor to compare
- `other`: Second tensor to compare
- `rtol`: Relative tolerance (proportional to magnitude of values)
- `atol`: Absolute tolerance (fixed threshold)
- `equal_nan`: Whether NaN values should be considered equal

**What it does**:
Returns `True` if all elements are close within the specified tolerances. The comparison uses:
```
|input - other| <= atol + rtol * |other|
```

**Common use cases**:
- Unit testing model outputs
- Comparing floating-point computations
- Validating numerical implementations
- Checking gradient computations

**Example**:
```python
import torch

# Exact match
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([1.0, 2.0, 3.0])
print(torch.allclose(a, b))  # True

# Close within tolerance
c = torch.tensor([1.0000, 2.0001, 3.0000])
d = torch.tensor([1.0001, 2.0000, 2.9999])
print(torch.allclose(c, d, rtol=1e-3, atol=1e-3))  # True

# Not close enough
e = torch.tensor([1.0, 2.0, 3.0])
f = torch.tensor([1.1, 2.0, 3.0])
print(torch.allclose(e, f, rtol=1e-3, atol=1e-3))  # False
```

## torch.argmax

**Purpose**: Returns the indices of maximum values along a specified dimension.

**Signature**: `torch.argmax(input, dim=None, keepdim=False)`

**Parameters**:
- `input`: Input tensor
- `dim`: Dimension along which to find the maximum (if None, returns index of global maximum)
- `keepdim`: Whether to keep the reduced dimension as size 1

**What it does**:
Finds the index (position) of the maximum value along the specified dimension. If no dimension is specified, it returns the index of the global maximum in the flattened tensor.

**Common use cases**:
- Getting predicted class indices from logits/probabilities
- Finding the most likely token in language model outputs
- Implementing greedy decoding strategies
- Converting probabilities back to discrete choices

**Example**:
```python
import torch

# 1D tensor - find global maximum index
scores = torch.tensor([0.1, 0.7, 0.2])
max_idx = torch.argmax(scores)
print(max_idx)  # tensor(1) - index of 0.7

# 2D tensor - find maximum along dimension
logits = torch.tensor([[2.0, 1.0, 0.5],
                       [0.1, 3.0, 1.5]])

# Maximum index along dim=1 (columns)
max_cols = torch.argmax(logits, dim=1)
print(max_cols)  # tensor([0, 1]) - first row max at index 0, second row at index 1

# Maximum index along dim=0 (rows)
max_rows = torch.argmax(logits, dim=0)
print(max_rows)  # tensor([0, 1, 1]) - compare values across rows

# In language model context
batch_probs = torch.softmax(logits, dim=1)
predicted_tokens = torch.argmax(batch_probs, dim=1)
print(predicted_tokens)  # tensor([0, 1]) - predicted token IDs
```

## torch.multinomial

**Purpose**: Samples indices from a probability distribution using weighted random sampling.

**Signature**: `torch.multinomial(input, num_samples, replacement=False, generator=None)`

**Parameters**:
- `input`: Tensor containing probabilities (must be non-negative, doesn't need to sum to 1)
- `num_samples`: Number of samples to draw
- `replacement`: Whether to sample with replacement (can draw same index multiple times)
- `generator`: Optional random number generator for reproducible results

**What it does**:
Performs weighted random sampling from a probability distribution. Unlike `torch.argmax` which always picks the highest probability item, `multinomial` randomly selects based on the probabilities - higher probability items are more likely to be chosen, but any item with non-zero probability could be selected.

**Mathematical intuition**:
Think of it like a weighted lottery wheel where each token gets a slice proportional to its probability. The wheel spins randomly, but tokens with higher probabilities have larger slices and are more likely to be selected.

**Common use cases**:
- **Sampling-based text generation** (temperature, top-k, nucleus sampling)
- **Stochastic decoding** in language models
- **Monte Carlo methods** and probabilistic algorithms
- **Data augmentation** with weighted random selection

**Comparison with other selection methods**:
- `torch.argmax`: Always picks highest probability (deterministic)
- `torch.multinomial`: Randomly samples based on probabilities (stochastic)
- `random.choice`: Uniform random selection (ignores probabilities)

**Example**:
```python
import torch

# Probability distribution over 5 tokens
probs = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1])

# Sample one token using multinomial
sampled_idx = torch.multinomial(probs, num_samples=1)
print(sampled_idx)  # Could be any index, but 2 is most likely (40% chance)

# Multiple samples with replacement
samples = torch.multinomial(probs, num_samples=100, replacement=True)
print(samples.bincount())  # Should roughly match the probability distribution

# In language model context - creative text generation
logits = torch.tensor([2.0, 1.0, 4.0, 0.5, 1.5])  # Raw model outputs
probs = torch.softmax(logits, dim=0)  # Convert to probabilities

# Greedy (deterministic) - always picks token 2
greedy_token = torch.argmax(probs)
print(f"Greedy choice: {greedy_token}")  # Always: tensor(2)

# Sampling (stochastic) - picks based on probabilities
sampled_token = torch.multinomial(probs, num_samples=1)
print(f"Sampled choice: {sampled_token}")  # Could be 0,1,2,3,4 with different probabilities

# Temperature sampling example
def sample_with_temperature(logits, temperature=1.0):
    # Apply temperature scaling
    scaled_logits = logits / temperature
    # Convert to probabilities
    probs = torch.softmax(scaled_logits, dim=0)
    # Sample using multinomial
    return torch.multinomial(probs, num_samples=1)

# Lower temperature = more deterministic (focused on high-prob tokens)
low_temp_sample = sample_with_temperature(logits, temperature=0.5)

# Higher temperature = more random (considers low-prob tokens more)
high_temp_sample = sample_with_temperature(logits, temperature=2.0)
```

**Context in RuvonVLLM Sampling**:
In our Day 5 sampling implementation, `torch.multinomial` is the core function that enables creative text generation:

```python
# From ruvonvllm/sampling/strategies.py
def sample_token(logits, temperature=1.0, top_k=None, top_p=None):
    # Apply filtering and temperature scaling
    filtered_logits = apply_filters(logits, top_k, top_p, temperature)

    # Convert to probabilities
    probs = torch.softmax(filtered_logits, dim=-1)

    # The magic happens here - weighted random sampling!
    token_id = torch.multinomial(probs, num_samples=1).item()

    return token_id
```

This is what transforms our inference engine from deterministic (always same output) to creative (varied, interesting outputs while respecting the model's learned probabilities).

## Entropy

**Definition**: A measure of uncertainty, randomness, or "surprise" in a probability distribution, derived from information theory.

**Mathematical Formula**:
```
Entropy = -Σ(p_i * log(p_i))
```
Where `p_i` is the probability of each possible outcome.

**Units**: Measured in bits (when using log base 2) or nats (when using natural log).

**Intuitive Understanding**:
Think of entropy as measuring how "surprised" you'd be by an outcome. It quantifies the unpredictability of a system:

- **Low Entropy (0-2 bits)**: Very predictable, not surprising
  - Example: Coin with heads=99%, tails=1% → entropy ≈ 0.08 bits
  - After "The sun rises in the", next word is almost certainly "east"

- **High Entropy (many bits)**: Very unpredictable, surprising
  - Example: Fair coin with heads=50%, tails=50% → entropy = 1.0 bits
  - After "I think the", many words are possible: "best", "worst", "most", etc.

**Key Properties**:
- **Minimum entropy = 0**: When one outcome has 100% probability (completely predictable)
- **Maximum entropy**: When all outcomes are equally likely (completely random)
- **Higher entropy = more uncertainty/information content**

**Entropy in Different Scenarios**:
```python
import torch

# Very confident distribution (low entropy)
confident = torch.tensor([0.9, 0.05, 0.03, 0.02])
entropy_low = -torch.sum(confident * torch.log(confident + 1e-10))
# Result: ~0.57 bits (predictable)

# Uniform distribution (maximum entropy for 4 options)
uniform = torch.tensor([0.25, 0.25, 0.25, 0.25])
entropy_max = -torch.sum(uniform * torch.log(uniform + 1e-10))
# Result: ~1.39 bits (maximum uncertainty)

# Moderate uncertainty
moderate = torch.tensor([0.4, 0.3, 0.2, 0.1])
entropy_mid = -torch.sum(moderate * torch.log(moderate + 1e-10))
# Result: ~1.28 bits (balanced)
```

**Context in Language Models**:
In language models, entropy measures how uncertain the model is about the next token:

- **Low entropy context**: "United States of ___" → model confident next word is "America"
- **High entropy context**: "The ___" → many possible words ("cat", "dog", "house", etc.)

**Why Entropy Matters in Sampling**:

1. **Quality Control**:
   - **Too low entropy**: Repetitive, boring text
   - **Too high entropy**: Incoherent, nonsensical text
   - **Optimal entropy**: Creative but coherent text

2. **Understanding Sampling Effects**:
   ```python
   # Temperature effects on entropy
   original_entropy = 3.5 bits

   # Low temperature (0.5) → decreases entropy → more focused
   low_temp_entropy = 2.1 bits  # Change: -1.4 bits

   # High temperature (2.0) → increases entropy → more random
   high_temp_entropy = 4.2 bits  # Change: +0.7 bits
   ```

3. **Debugging Generation**:
   - Tracking entropy changes helps understand why output is boring/chaotic
   - Guides parameter tuning for optimal creativity vs coherence

**RuvonVLLM Implementation**:
Our `get_sampling_info()` function tracks entropy to analyze sampling effectiveness:

```python
def get_sampling_info(logits, temperature, top_k, top_p):
    # Original model uncertainty
    original_probs = F.softmax(logits, dim=-1)
    original_entropy = -torch.sum(original_probs * torch.log(original_probs + 1e-10))

    # Apply sampling filters
    filtered_logits = apply_sampling(logits, temperature, top_k, top_p)
    final_probs = F.softmax(filtered_logits, dim=-1)
    final_entropy = -torch.sum(final_probs * torch.log(final_probs + 1e-10))

    return {
        "original_entropy": original_entropy.item(),
        "final_entropy": final_entropy.item(),
        "entropy_change": final_entropy.item() - original_entropy.item(),
        # Negative change = more focused, positive = more random
    }
```

**Information Theory Connection**:
Entropy originates from Claude Shannon's information theory (1948). It quantifies:
- How much "information" is contained in a message
- How efficiently data can be compressed
- The minimum number of bits needed to encode outcomes

**Practical Examples in RuvonVLLM**:
When running `python cli.py sample --show-steps`:
```
Step 1:
  Generated token: 42 -> ' bright'
  Top token prob: 0.156
  Effective vocab: 234
  Entropy change: -2.341
```

This tells us:
- **Original entropy**: ~6.8 bits (estimated from vocab size)
- **Final entropy**: ~4.5 bits (after filtering)
- **Entropy change -2.341**: Our sampling reduced uncertainty by 2.3 bits
- **Interpretation**: We made the model more focused while preserving some creativity

Entropy is essentially **uncertainty engineering** - our sampling strategies control how much randomness vs predictability we want in text generation!

## Usage in RuvonVLLM Context

These functions are commonly used together in language model inference:

```python
# Typical inference pipeline
def generate_next_token(model, input_ids):
    # Get model logits
    logits = model(input_ids)  # Shape: [batch_size, vocab_size]

    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Greedy selection: pick most likely token
    next_token = torch.argmax(probs, dim=-1)

    return next_token

# Testing model outputs
def test_model_output(expected_logits, actual_logits):
    # Check if outputs are numerically close
    assert torch.allclose(expected_logits, actual_logits, rtol=1e-4)

    # Verify predictions match
    expected_tokens = torch.argmax(expected_logits, dim=-1)
    actual_tokens = torch.argmax(actual_logits, dim=-1)
    assert torch.equal(expected_tokens, actual_tokens)
```
