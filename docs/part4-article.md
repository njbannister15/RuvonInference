# Part 4: From Script to Service

*Standing up a GPT-2 API in 24 hours: Building our first HTTP inference server*

Part 4 marked a pivotal transformation in our tiny vLLM journey. We took our command-line inference engine and wrapped it in a **production-ready HTTP API** with streaming capabilities. Suddenly, our GPT-2 model went from a developer tool to something that looks and feels like a real product.

## The Product Surface Problem

After three parts of building solid foundations - tokenization, generation, and KV-cache optimization - we had a powerful inference engine. But there was one problem: **investors don't curl into terminal applications**.

To demonstrate real product potential, we needed to transform our CLI tool into something that could serve multiple users over HTTP, with the kind of streaming responses that make modern AI applications feel responsive and interactive.

## Building the FastAPI Server

The solution was FastAPI - Python's modern, fast web framework. In just a few hours, we built a complete API server with:

### Core Endpoints

```python
@app.post("/completions")
async def create_completion(request: CompletionRequest):
    """OpenAI-compatible text completion with streaming support"""

@app.get("/health")
async def health_check():
    """Health check with model status"""

@app.get("/")
async def root():
    """API information and quick start guide"""
```

### Streaming Magic

The real breakthrough was implementing **token-by-token streaming**. Instead of waiting for complete generation, our API now streams each token as it's generated:

```bash
curl -X POST http://localhost:8000/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 15, "stream": true}'
```

Results in real-time output:
```
data: {"choices":[{"text":" uncertain","finish_reason":null}]}
data: {"choices":[{"text":".","finish_reason":null}]}
data: {"choices":[{"text":" The","finish_reason":null}]}
data: {"choices":[{"text":" future","finish_reason":null}]}
...
data: [DONE]
```

## Technical Implementation Highlights

### 1. **Lazy Model Loading**
Models only load when first requested, saving memory and startup time:

```python
def get_model(model_name: str = "gpt2") -> GPT2Model:
    if model_name not in model_instances:
        model = GPT2Model(model_name, device="cpu")
        model.load_model()
        model_instances[model_name] = model
    return model_instances[model_name]
```

### 2. **Streaming Generator**
Async generator yields JSON chunks following OpenAI's format:

```python
async def generate_completion_stream(request):
    for step in range(request.max_tokens):
        # Generate next token
        next_token_id = torch.argmax(next_token_logits).item()
        new_token_text = tokenizer.decode([next_token_id])

        # Yield streaming chunk
        chunk = CompletionStreamChunk(...)
        yield f"data: {chunk.model_dump_json()}\n\n"

        # Small delay for demo smoothness
        await asyncio.sleep(0.05)
```

### 3. **OpenAI API Compatibility**
Our response format matches OpenAI's API, making integration seamless:

```json
{
  "id": "cmpl-1759202217",
  "object": "text_completion",
  "created": 1759202217,
  "model": "gpt2",
  "choices": [{
    "text": ", the world was a place of great beauty and",
    "index": 0,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 10,
    "total_tokens": 14
  }
}
```

## The Developer Experience

We didn't just build an API - we built an **experience**. The CLI now includes a beautiful `serve` command:

```bash
python cli.py serve --port 8000
```

Which displays:
- Server address and status
- Interactive documentation URL (`/docs`)
- Copy-paste curl examples for testing
- Health check endpoint for monitoring

## Performance Results

Our Part 4 API leverages all previous optimizations:
- ✅ **KV-cache optimization** for 5.1x generation speedup
- ✅ **Streaming responses** for immediate user feedback
- ✅ **Lazy loading** for fast server startup
- ✅ **Health monitoring** for production readiness

## The Demo That Matters

The real win? You can now demo our inference engine to anyone with a simple curl command:

```bash
# Quick health check
curl http://localhost:8000/health

# Generate text instantly
curl -X POST http://localhost:8000/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 10}'
```

No more "let me show you this Python script" - now it's "here's our API endpoint."

## What This Unlocks

Part 4's HTTP server transforms our project from a **proof of concept** to a **product demo**. Investors can:

1. **See it working** - Real HTTP endpoints they can test
2. **Experience streaming** - Token-by-token generation feels responsive
3. **Understand scale** - Multiple concurrent requests (though we'll optimize this in Week 2)
4. **Imagine integration** - OpenAI-compatible API means easy adoption

## Next Steps

With our HTTP foundation solid, Part 5 will add **creative sampling strategies** - temperature, top-k, and nucleus sampling. Because while greedy decoding is deterministic and fast, creativity requires a bit of controlled randomness.

Our tiny vLLM engine now has a product surface. Time to make it creative.

---

*Tomorrow: Teaching our API to be creative with advanced sampling strategies*
