"""
FastAPI server for RuvonVLLM inference engine.

This module implements Day 4's HTTP API server with a /completions endpoint
that provides streaming text generation compatible with OpenAI-like interfaces.
"""

import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ruvonvllm.model.gpt2 import GPT2Model
from ruvonvllm.tokenizer.gpt2_tokenizer import GPT2TokenizerWrapper


class CompletionRequest(BaseModel):
    """
    Request model for text completion endpoint.

    This follows OpenAI's completion API structure while supporting our
    current greedy generation capabilities.
    """

    prompt: str = Field(..., description="The text prompt to complete")
    max_tokens: int = Field(
        default=20, ge=1, le=500, description="Maximum tokens to generate"
    )
    model: str = Field(default="gpt2", description="Model to use for generation")
    stream: bool = Field(default=False, description="Whether to stream the response")
    use_cache: bool = Field(
        default=True, description="Whether to use KV-cache optimization"
    )


class CompletionChoice(BaseModel):
    """Single completion choice in the response."""

    text: str
    index: int
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    """
    Response model for text completion.

    Follows OpenAI API structure for compatibility with existing clients.
    """

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Dict[str, int]


class CompletionStreamChunk(BaseModel):
    """Single chunk in a streaming completion response."""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[Dict[str, Any]]


# Global model instances (initialized on startup)
model_instances: Dict[str, GPT2Model] = {}
tokenizer_instances: Dict[str, GPT2TokenizerWrapper] = {}


def get_model(model_name: str = "gpt2") -> GPT2Model:
    """
    Get or create a model instance.

    This implements lazy loading - models are only loaded when first requested
    to save memory and startup time.

    Args:
        model_name: Name of the model to load

    Returns:
        Loaded GPT2Model instance
    """
    if model_name not in model_instances:
        print(f"Loading model: {model_name}")
        model = GPT2Model(model_name, device="cpu")  # TODO: Add GPU support detection
        model.load_model()
        model_instances[model_name] = model
        print(f"Model {model_name} loaded and cached")
    return model_instances[model_name]


def get_tokenizer(model_name: str = "gpt2") -> GPT2TokenizerWrapper:
    """
    Get or create a tokenizer instance.

    Args:
        model_name: Name of the model tokenizer to load

    Returns:
        Loaded GPT2TokenizerWrapper instance
    """
    if model_name not in tokenizer_instances:
        print(f"Loading tokenizer: {model_name}")
        tokenizer = GPT2TokenizerWrapper(model_name)
        tokenizer_instances[model_name] = tokenizer
        print(f"Tokenizer {model_name} loaded and cached")
    return tokenizer_instances[model_name]


async def generate_completion_stream(
    request: CompletionRequest,
) -> AsyncGenerator[str, None]:
    """
    Generate streaming completion response.

    This function implements token-by-token streaming by:
    1. Running generation step-by-step
    2. Yielding each new token as a JSON chunk
    3. Following OpenAI's streaming format

    Args:
        request: The completion request

    Yields:
        JSON-formatted streaming chunks
    """
    try:
        # Get model and tokenizer
        model = get_model(request.model)
        tokenizer = get_tokenizer(request.model)

        # Tokenize input
        input_ids = tokenizer.encode(request.prompt, return_tensors=True)

        # Create unique request ID and timestamp
        request_id = f"cmpl-{int(time.time())}"
        created_time = int(time.time())

        # Track generation state
        current_sequence = input_ids.squeeze().tolist()
        past_key_values = None

        # Generate tokens one by one for streaming
        for step in range(request.max_tokens):
            # Prepare input for this step
            if past_key_values is None or not request.use_cache:
                # First step or no caching: process full sequence
                current_ids = torch.tensor(current_sequence).unsqueeze(0)
            else:
                # Subsequent steps with caching: only process new token
                current_ids = torch.tensor([current_sequence[-1]]).unsqueeze(0)

            # Forward pass
            with torch.no_grad():
                if request.use_cache:
                    outputs = model.model(
                        current_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                else:
                    outputs = model.model(current_ids)

                logits = outputs.logits

            # Get next token (greedy selection)
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            current_sequence.append(next_token_id)

            # Decode the new token
            new_token_text = tokenizer.decode([next_token_id])

            # Create streaming chunk
            chunk = CompletionStreamChunk(
                id=request_id,
                created=created_time,
                model=request.model,
                choices=[
                    {
                        "text": new_token_text,
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
            )

            # Yield the chunk as JSON
            yield f"data: {chunk.model_dump_json()}\n\n"

            # Small delay for demo purposes (remove in production)
            await asyncio.sleep(0.05)

            # Check for end-of-sequence
            if (
                hasattr(model.model.config, "eos_token_id")
                and next_token_id == model.model.config.eos_token_id
            ):
                break

        # Send final chunk indicating completion
        final_chunk = CompletionStreamChunk(
            id=request_id,
            created=created_time,
            model=request.model,
            choices=[
                {
                    "text": "",
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        # Error handling for streaming
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "generation_error",
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


def generate_completion_sync(request: CompletionRequest) -> CompletionResponse:
    """
    Generate non-streaming completion response.

    This generates the complete text at once and returns it in a single response,
    following the OpenAI API format.

    Args:
        request: The completion request

    Returns:
        Complete CompletionResponse
    """
    # Get model and tokenizer
    model = get_model(request.model)
    tokenizer = get_tokenizer(request.model)

    # Tokenize input
    input_ids = tokenizer.encode(request.prompt, return_tensors=True)

    # Generate text
    if request.use_cache:
        generated_tokens = model.generate_greedy_with_cache(
            input_ids, request.max_tokens, show_progress=False
        )
    else:
        generated_tokens = model.generate_greedy(
            input_ids, request.max_tokens, show_progress=False
        )

    # Decode the full text
    full_text = tokenizer.decode(generated_tokens)
    generated_text = full_text[len(request.prompt) :]  # Remove original prompt

    # Create response
    response = CompletionResponse(
        id=f"cmpl-{int(time.time())}",
        created=int(time.time()),
        model=request.model,
        choices=[CompletionChoice(text=generated_text, index=0, finish_reason="stop")],
        usage={
            "prompt_tokens": len(input_ids[0]),
            "completion_tokens": len(generated_tokens) - len(input_ids[0]),
            "total_tokens": len(generated_tokens),
        },
    )

    return response


# Create FastAPI app
app = FastAPI(
    title="RuvonVLLM API",
    description="Tiny vLLM Inference Engine - Day 4: HTTP Server with Streaming",
    version="0.1.0",
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "🚀 RuvonVLLM API - Tiny vLLM Inference Engine",
        "version": "0.1.0",
        "day": 4,
        "description": "HTTP Server with streaming text completion",
        "endpoints": {
            "/completions": "Text completion endpoint (POST)",
            "/health": "Health check endpoint (GET)",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": list(model_instances.keys()),
        "tokenizers_loaded": list(tokenizer_instances.keys()),
    }


@app.post("/completions")
async def create_completion(request: CompletionRequest):
    """
    Create a text completion.

    This endpoint accepts a text prompt and generates a completion using
    our GPT-2 model. It supports both streaming and non-streaming responses.

    Args:
        request: Completion request parameters

    Returns:
        Streaming or complete response based on request.stream
    """
    try:
        if request.stream:
            # Return streaming response
            return StreamingResponse(
                generate_completion_stream(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Return complete response
            return generate_completion_sync(request)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting RuvonVLLM API Server...")
    print("📚 Day 4: HTTP Server with Streaming")
    print("-" * 50)

    uvicorn.run(
        "ruvonvllm.api.server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
