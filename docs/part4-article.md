# Part 4: From Script to Service

*Standing up a GPT-2 API in 24 hours: Building our first HTTP inference server*

Part 4 marked a pivotal transformation in our educational inference engine journey. We took our command-line inference engine and wrapped it in an HTTP API with queue-based processing.

## The Product Surface Problem

After three parts of building solid foundations - tokenization, generation, and KV-cache optimization - we have a simple engine.

To demonstrate real product potential, we needed to transform our CLI tool into something that could serve multiple users over HTTP, with the kind of queue-based processing that makes production applications scalable.

## Building the FastAPI Server

Enter FastAPI - Python's modern, fast web framework. In just a few hours, we built a complete API server with:

### Core Endpoints

```python
@app.post("/completions")
async def create_completion(request: CompletionRequest):
    """OpenAI-compatible text completion with queue processing"""

@app.get("/health")
async def health_check():
    """Health check with queue status and model info"""

@app.get("/")
async def root():
    """API information and endpoint documentation"""

@app.get("/queue")
async def get_queue_status():
    """Queue status and processing statistics"""

@app.get("/requests/{request_id}")
async def get_request_status(request_id: str):
    """Individual request status tracking"""
```

### OpenAI-Compatible API

The real breakthrough was building a **production-ready HTTP API** with OpenAI-compatible request/response formats. Our API provides complete text generation with advanced sampling parameters:

```bash
curl -X POST http://localhost:8000/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 15, "temperature": 0.8}'
```

Returns a complete OpenAI-style response:
```json
{
  "id": "cmpl-1759202217",
  "object": "text_completion",
  "created": 1759202217,
  "model": "gpt2",
  "choices": [{"text": " bright and full of possibilities", "index": 0, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11}
}
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

### 2. **Queue-Based Processing**
Requests are processed through a configurable queue strategy system:

```python
@app.post("/completions")
async def create_completion(request: CompletionRequest):
    """Create a text completion using the configured queue strategy."""
    try:
        if request.stream:
            # Streaming not yet supported with queue processing
            raise HTTPException(status_code=400, detail="Streaming not supported")

        # Delegate to the configured strategy
        return await queue_strategy.process_request(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request processing failed: {str(e)}")
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

## Architecture Features

Our Part 4 API provides production-ready capabilities:
- ✅ **KV-cache optimization** integrated from Part 3
- ✅ **Queue-based processing** for handling multiple requests
- ✅ **Lazy loading** for efficient memory usage
- ✅ **Health monitoring** and request tracking
- ✅ **OpenAI compatibility** for easy integration

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
2. **Experience queuing** - Professional request handling and tracking
3. **Understand scale** - Multiple concurrent requests (though we'll optimize this in Week 2)
4. **Imagine integration** - OpenAI-compatible API means easy adoption

## Next Steps

With our HTTP foundation solid, Part 5 will add **creative sampling strategies** - temperature, top-k, and nucleus sampling. Because while greedy decoding is deterministic and fast, creativity requires a bit of controlled randomness.

Our educational inference engine now has a product surface. Time to make it creative.

---

## Navigation

← **Previous**: [Part 3: KV-Cache Optimization](part3-article.md) | **Next**: [Part 5: Sampling Strategies](part5-article.md) →

---

*Tomorrow: Teaching our API to be creative with advanced sampling strategies*
