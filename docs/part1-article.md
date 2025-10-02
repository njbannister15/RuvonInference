# Part 1: How I Ran a Transformer From Scratch

*Building the foundation: from text to tokens to predictions*

ToPart marked the beginning of my 20-part journey to build a tiny vLLM inference engine from scratch. The goal for Part 1 was to load GPT-2 124M weights, tokenize a string, run it through the model, and decode the predictions back to text.

## What We Built

By the end of Part 1, I had:
- ✅ A functional GPT-2 model wrapper that loads 124M parameters
- ✅ A tokenizer that converts text to tokens and back
- ✅ A complete forward pass pipeline from text → tokens → logits → predictions
- ✅ A CLI interface using Rich and Typer
- ✅ Clean, documented code with proper error handling

The demo successfully shows "Hello world" being tokenized into `[15496, 995]` (representing "Hello" and " world"), processed through 124 million parameters, and predicting "," as the most likely next token with 27.5% probability.

## The Problems We Encountered (And How We Solved Them)

### Problem 1: PyTorch Installation on my Intel Mac

**The Issue**: Modern PyTorch has dropped support for Intel macOS after version 2.2.0. When trying to install recent PyTorch versions, we got cryptic errors about missing wheels for `macosx_15_0_x86_64`.

**The Investigation**: A deep dive into PyTorch's development discussions revealed that [PyTorch stopped building Intel macOS binaries in January 2024](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690). The last compatible version was 2.2.2.

**The Solution**:
1. Pin PyTorch to version 2.2.2 from the CPU index
2. Use `uv pip install` with the PyTorch CPU index URL
3. Add PyTorch installation to our Makefile setup command to ensure it persists

```bash
uv pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
```

### Problem 2: NumPy Version Conflicts

**The Issue**: PyTorch 2.2.2 was compiled with NumPy 1.x, but our dependency resolver pulled in NumPy 2.1.2, causing runtime crashes with the message: "module that was compiled using NumPy 1.x cannot be run in NumPy 2.1.2"

**The Solution**: Constrain NumPy to version 1.x in our `pyproject.toml`:
```toml
"numpy>=1.24.0,<2.0"
```

This ensures compatibility between PyTorch and NumPy while still getting performance improvements from modern NumPy 1.x versions.

### Problem 3: Persistent Dependencies with uv

**The Issue**: PyTorch kept getting uninstalled every time we ran `uv sync`, because uv manages dependencies strictly based on what's in `pyproject.toml`.

**The Solution**: Rather than fighting the dependency manager, we embraced uv's pip interface and added PyTorch installation as a post-sync step in our Makefile. This way, core dependencies are managed by uv, but PyTorch (which needs special index handling) is installed separately.

## Understanding GPT-2 Tokenization

One interesting discovery was understanding the difference between tokens and their decoded representation. For example:

- Token: `'Ġfuture'` → Decoded: `' future'`

The `Ġ` symbol is GPT-2's way of representing leading spaces in its Byte Pair Encoding (BPE). This allows the model to understand word boundaries while working with subword tokens.

## What's Next: Part 2

Tomorrow, we'll implement the greedy decode loop - turning our single forward pass into iterative generation. Instead of just predicting the next token, we'll generate complete sequences by feeding predictions back into the model.

---

*This is part of a 20-Part series building a tiny vLLM inference engine from scratch. Follow along as we add features like KV-caching, continuous batching, FlashAttention, and more.*
