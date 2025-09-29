# Day 2: My First Tiny LLM - Teaching a Transformer to Tell Stories

*From single predictions to complete narratives: implementing greedy decoding*

Yesterday we built the foundation - loading GPT-2, tokenizing text, and getting single predictions. Today, we taught our transformer to tell complete stories by implementing the greedy decode loop. The result? Our tiny inference engine can now generate coherent text sequences, one carefully chosen token at a time.

## What We Built

By the end of Day 2, our engine gained a crucial capability:
- ✅ **Greedy decoding algorithm** - iterative text generation using argmax
- ✅ **Sequence generation** - complete stories from simple prompts
- ✅ **Step-by-step visualization** - see each token being generated live
- ✅ **Beautiful CLI interface** - Rich styling for the generation process
- ✅ **Performance tracking** - generation statistics and timing

The demo successfully generates: `"Once upon a time" → "Once upon a time, the world was a place of great beauty and great danger. The world was a place of great"` - a coherent 20-token story!

## Understanding Greedy Decoding

### The Algorithm

Greedy decoding is elegantly simple:
1. **Start** with input tokens (e.g., "Once upon a time" → `[7454, 2402, 257, 640]`)
2. **Forward pass** through 124M parameters to get next-token predictions
3. **Pick** the token with highest confidence (argmax of logits)
4. **Append** it to the sequence
5. **Repeat** until desired length

### Step-by-Step Generation

Our `--show-steps` feature reveals the magic:

```
Step 1: Generated token 11 → ','
Step 2: Generated token 262 → ' the'
Step 3: Generated token 995 → ' world'
Step 4: Generated token 373 → ' was'
Step 5: Generated token 257 → ' a'
```

Each step represents a complete forward pass through the entire neural network, considering all previous context to predict what comes next.

### Why "Greedy"?

It's called "greedy" because at each step, we greedily take the best option without considering future consequences. This creates:
- **Deterministic output** - same input always produces same result
- **Fast generation** - no complex sampling computations
- **Predictable behavior** - good for debugging and testing

The trade-off is creativity - greedy decoding can be repetitive since it always picks the "safest" choice.

### The Repetition Problem (A Discovery!)

While testing longer generations, we discovered a classic issue with greedy decoding - **repetitive loops**! When generating 100+ tokens, our model produces hilariously repetitive text:

```
"Once upon a time, the world was a place of great beauty and great danger.
The world was a place of great danger, and the world was a place of great danger.
The world was a place of great danger, and the world was a place of great danger..."
```

**What's happening?** The model gets trapped in a high-probability cycle:
1. After "great danger" → most likely token is "."
2. After "." → most likely token is " The"
3. After "The" → most likely token is " world"
4. **Infinite loop** of the same safe choices!

This perfectly illustrates why **greedy decoding is terrible for creative generation** but excellent for understanding the basic mechanics. Real text generation needs randomness and exploration - exactly what we'll implement in Day 5 with temperature, top-k, and top-p sampling methods.

## Technical Implementation

### The Core Loop

```python
def generate_greedy(self, input_ids, max_length=20):
    sequence = input_ids.squeeze().tolist()

    for step in range(max_length):
        # Forward pass through all 12 transformer layers
        current_ids = torch.tensor(sequence).unsqueeze(0)
        logits = self.model(current_ids).logits

        # Greedy selection: pick highest confidence token
        next_token_id = torch.argmax(logits[0, -1, :]).item()
        sequence.append(next_token_id)

    return sequence
```

### Why This Works

Each iteration processes the **entire sequence** through the transformer:
- **Self-attention** lets each position look at all previous tokens
- **Position encoding** maintains understanding of sequence order
- **Causal masking** prevents looking ahead (maintains autoregressive property)

This means token 20 is generated with full awareness of tokens 1-19, creating coherent long-form text.

## The Beautiful CLI Experience

Day 2's generation command showcases our Rich + Typer interface:

```bash
# Basic story generation
make generate

# Custom prompt with specific length
python cli.py generate --text "The future of AI is" --max-length 15

# Watch the magic happen step-by-step
python cli.py generate --text "Science is" --show-steps
```

The CLI provides:
- 🎨 **Colorful progress indicators** during model loading
- 📊 **Generation statistics** showing token counts and method
- 🎭 **Real-time step visualization** for educational purposes
- 📖 **Formatted results** separating original vs generated text

## Performance Insights

Our current implementation processes:
- **Input**: 4 tokens ("Once upon a time")
- **Generated**: 20 new tokens in ~2-3 seconds on CPU
- **Method**: Greedy (deterministic argmax selection)
- **Model**: GPT-2 124M parameters

Each token generation requires a full forward pass through 124 million parameters - this is why Day 3 will focus on KV-caching to dramatically improve efficiency.

## What's Next: Day 3 - KV-Cache Optimization

Tomorrow we'll tackle the performance bottleneck: recomputing attention for previous tokens. Currently, generating 20 tokens requires 20 full forward passes through the entire sequence. With KV-caching, we'll cache the attention keys and values, reducing subsequent generations to single-token computations.

This optimization will:
- **Cut latency** by 10-20x for longer sequences
- **Reduce memory waste** from redundant computations
- **Enable real-time** interactive generation
- **Scale efficiently** to much longer contexts

---

*This is part of a 20-day series building a tiny vLLM inference engine from scratch. Follow along as we add KV-caching, HTTP APIs, continuous batching, and more.*
