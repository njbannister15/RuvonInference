"""
FastAPI server for RuvonVLLM inference engine.

This module implements Day 4's HTTP API server with a /completions endpoint
that provides streaming text generation compatible with OpenAI-like interfaces.
"""

import asyncio
import json
import os
import time
import threading
from typing import AsyncGenerator, Dict, Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ruvonvllm.model.gpt2 import GPT2Model
from ruvonvllm.tokenizer.gpt2_tokenizer import GPT2TokenizerWrapper
from ruvonvllm.sampling.strategies import sample_token
from ruvonvllm.api.queue import request_queue, RequestStatus
from ruvonvllm.api.batched_queue import batched_request_queue
from ruvonvllm.api.continuous_queue import continuous_scheduler


class CompletionRequest(BaseModel):
    """
    Request model for text completion endpoint.

    This follows OpenAI's completion API structure while supporting our
    advanced sampling capabilities for creative text generation.
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

    # Sampling parameters
    temperature: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Temperature for sampling (0.1=focused, 1.0=balanced, 2.0=creative)",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=1000,
        description="Top-k sampling: only consider k most likely tokens",
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.01,
        le=1.0,
        description="Nucleus sampling: dynamic cutoff based on cumulative probability",
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

            # Get next token using sampling
            next_token_logits = logits[0, -1, :]
            next_token_id = sample_token(
                next_token_logits,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
            )
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

    # Generate text using sampling
    generated_tokens = model.generate_with_sampling(
        input_ids,
        max_length=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        use_cache=request.use_cache,
        show_progress=False,
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


def process_request_sync(request: CompletionRequest) -> CompletionResponse:
    """
    Process a single request synchronously.

    This is the core processing function that handles one request at a time.
    It's called by the queue processor to ensure sequential execution.

    Args:
        request: The completion request to process

    Returns:
        CompletionResponse with the generated text
    """
    # Generate text using sampling
    generated_tokens = get_model(request.model).generate_with_sampling(
        get_tokenizer(request.model).encode(request.prompt, return_tensors=True),
        max_length=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        use_cache=request.use_cache,
        show_progress=False,
    )

    # Decode the full text
    full_text = get_tokenizer(request.model).decode(generated_tokens)
    generated_text = full_text[len(request.prompt) :]

    # Create response
    response = CompletionResponse(
        id=f"cmpl-{int(time.time())}",
        created=int(time.time()),
        model=request.model,
        choices=[CompletionChoice(text=generated_text, index=0, finish_reason="stop")],
        usage={
            "prompt_tokens": len(
                get_tokenizer(request.model).encode(
                    request.prompt, return_tensors=True
                )[0]
            ),
            "completion_tokens": len(generated_tokens)
            - len(
                get_tokenizer(request.model).encode(
                    request.prompt, return_tensors=True
                )[0]
            ),
            "total_tokens": len(generated_tokens),
        },
    )

    return response


def queue_processor():
    """
    Background thread that processes requests from the queue sequentially.

    This function runs in a separate thread and continuously processes requests
    from the queue one at a time, ensuring that model access is serialized.
    """
    import logging

    logger = logging.getLogger(__name__)

    logger.info("Queue processor started")

    while True:
        try:
            # Get next request from queue
            queued_request = request_queue.get_next_request()

            if queued_request is None:
                # No requests in queue, sleep briefly
                time.sleep(0.1)
                continue

            # Mark request as processing
            request_queue.start_processing(queued_request.id)

            try:
                # Process the request
                result = process_request_sync(queued_request.request_data)

                # Mark as completed
                request_queue.complete_request(queued_request.id, result)

            except Exception as e:
                # Mark as failed
                request_queue.fail_request(queued_request.id, str(e))
                logger.error(f"Request {queued_request.id} failed: {e}")

        except Exception as e:
            logger.error(f"Queue processor error: {e}")
            time.sleep(1)  # Brief pause on error


# Start the background queue processor
queue_thread = threading.Thread(target=queue_processor, daemon=True)
queue_thread.start()


def process_batch_sync(batch_requests: list) -> list:
    """
    Process a batch of requests using batched model inference.

    This is the core of continuous batching - taking multiple requests
    and processing them simultaneously in a single model forward pass.

    Args:
        batch_requests: List of CompletionRequest objects

    Returns:
        List of CompletionResponse objects
    """
    if not batch_requests:
        return []

    # Use the first request's model for the batch (assume all use same model)
    model_name = batch_requests[0].model
    model = get_model(model_name)
    tokenizer = get_tokenizer(model_name)

    # Tokenize all inputs
    batch_input_ids = []
    for request in batch_requests:
        input_ids = tokenizer.encode(request.prompt, return_tensors=True)
        batch_input_ids.append(input_ids)

    # Generate text using batched inference
    batch_generated_tokens = model.generate_batch_with_sampling(
        batch_input_ids,
        max_length=batch_requests[0].max_tokens,  # Use first request's settings
        temperature=batch_requests[0].temperature,
        top_k=batch_requests[0].top_k,
        top_p=batch_requests[0].top_p,
        use_cache=batch_requests[0].use_cache,
        show_progress=False,
    )

    # Create responses for each request
    responses = []
    for i, (request, generated_tokens) in enumerate(
        zip(batch_requests, batch_generated_tokens)
    ):
        # Decode the full text
        full_text = tokenizer.decode(generated_tokens)
        generated_text = full_text[len(request.prompt) :]  # Remove original prompt

        # Create response
        response = CompletionResponse(
            id=f"cmpl-{int(time.time())}-{i}",
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(text=generated_text, index=0, finish_reason="stop")
            ],
            usage={
                "prompt_tokens": len(batch_input_ids[i][0]),
                "completion_tokens": len(generated_tokens) - len(batch_input_ids[i][0]),
                "total_tokens": len(generated_tokens),
            },
        )
        responses.append(response)

    return responses


def batched_queue_processor():
    """
    Background thread that processes requests from the batched queue.

    This implements continuous batching by:
    1. Collecting multiple requests into batches
    2. Processing entire batches in single model calls
    3. Distributing results back to individual requests

    This should show "Processing: 2, 3, 4" instead of always "Processing: 1"
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Batched queue processor started")

    while True:
        try:
            # Collect next batch of requests
            batch = batched_request_queue.collect_next_batch()

            if batch is None:
                # No requests available, sleep briefly
                time.sleep(0.05)  # 50ms sleep
                continue

            # Mark batch as processing
            batched_request_queue.start_batch_processing(batch)

            try:
                # Extract request data from batch
                request_data_list = [req.request_data for req in batch.requests]

                # Process the entire batch in one model call
                results = process_batch_sync(request_data_list)

                # Mark batch as completed
                batched_request_queue.complete_batch(batch.id, results)

                logger.info(
                    f"Processed batch {batch.id} with {batch.size} requests "
                    f"in {batch.processing_time:.3f}s"
                )

            except Exception as e:
                # Mark batch as failed
                batched_request_queue.fail_batch(batch.id, str(e))
                logger.error(f"Batch {batch.id} failed: {e}")

        except Exception as e:
            logger.error(f"Batched queue processor error: {e}")
            time.sleep(1)  # Brief pause on error


# Read queue mode from environment variable (set by CLI)
QUEUE_MODE = os.getenv(
    "QUEUE_MODE", "batched"
).lower()  # sequential, batched, continuous

# Start the appropriate queue processor
if QUEUE_MODE == "continuous":
    # Start continuous batching processor
    continuous_processor_thread = threading.Thread(
        target=lambda: asyncio.run(continuous_scheduler.continuous_generation_loop()),
        daemon=True,
    )
    continuous_processor_thread.start()
    print("⚡ Started continuous batching processor (TRUE CONTINUOUS BATCHING)")
elif QUEUE_MODE == "batched":
    batched_queue_thread = threading.Thread(target=batched_queue_processor, daemon=True)
    batched_queue_thread.start()
    print("🚀 Started batched queue processor (PREFILL BATCHING)")
else:
    print("📋 Using sequential queue processor (SEQUENTIAL PROCESSING)")


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
        "part": 6,
        "description": "Multiple sequential requests with queue processing",
        "endpoints": {
            "/completions": "Text completion endpoint (POST) - queued processing",
            "/health": "Health check endpoint (GET) - includes queue status",
            "/queue": "Queue status and statistics (GET)",
            "/requests/{id}": "Individual request status (GET)",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with queue status."""
    queue_stats = request_queue.stats
    return {
        "status": "healthy",
        "models_loaded": list(model_instances.keys()),
        "tokenizers_loaded": list(tokenizer_instances.keys()),
        "queue": queue_stats,
    }


@app.post("/completions")
async def create_completion(request: CompletionRequest):
    """
    Create a text completion.

    This endpoint accepts a text prompt and adds it to the processing queue.
    For Part 6, we handle multiple sequential requests by queuing them and
    processing one at a time.

    Args:
        request: Completion request parameters

    Returns:
        Streaming or complete response based on request.stream
    """
    try:
        if request.stream:
            # Streaming not yet supported with queue (Part 6 focuses on sequential non-streaming)
            raise HTTPException(
                status_code=400,
                detail="Streaming not supported with queue processing in Part 6",
            )
        else:
            if QUEUE_MODE == "continuous":
                # Add request to continuous scheduler (Part 8)
                request_id = continuous_scheduler.add_request(request)

                # Wait for request to complete
                max_wait_time = 300  # 5 minutes timeout
                start_time = time.time()

                while time.time() - start_time < max_wait_time:
                    continuous_request = continuous_scheduler.get_request_status(
                        request_id
                    )

                    if continuous_request is None:
                        raise HTTPException(
                            status_code=500,
                            detail="Request not found in continuous scheduler",
                        )

                    if continuous_request["status"] == "completed":
                        return continuous_request["result"]

                    elif continuous_request["status"] == "failed":
                        raise HTTPException(
                            status_code=500,
                            detail=f"Request failed: {continuous_request.get('error', 'Unknown error')}",
                        )

                    # Still processing, wait a bit
                    await asyncio.sleep(0.1)

                # Timeout
                raise HTTPException(status_code=408, detail="Request timeout")

            elif QUEUE_MODE == "batched":
                # Add request to batched queue (Part 7)
                request_id = batched_request_queue.add_request(request)

                # Wait for request to complete
                max_wait_time = 300  # 5 minutes timeout
                start_time = time.time()

                while time.time() - start_time < max_wait_time:
                    batched_request = batched_request_queue.get_request_status(
                        request_id
                    )

                    if batched_request is None:
                        raise HTTPException(
                            status_code=500, detail="Request not found in batched queue"
                        )

                    if batched_request.status == "completed":
                        return batched_request.result

                    elif batched_request.status == "failed":
                        raise HTTPException(
                            status_code=500,
                            detail=f"Request failed: {batched_request.error}",
                        )

                    # Still processing, wait a bit
                    await asyncio.sleep(0.1)

                # Timeout
                raise HTTPException(status_code=408, detail="Request timeout")

            else:
                # Add request to sequential queue (Part 6)
                request_id = request_queue.add_request(request)

                # Wait for request to complete
                max_wait_time = 300  # 5 minutes timeout
                start_time = time.time()

                while time.time() - start_time < max_wait_time:
                    queued_request = request_queue.get_request_status(request_id)

                    if queued_request is None:
                        raise HTTPException(
                            status_code=500, detail="Request not found in queue"
                        )

                    if queued_request.status == RequestStatus.COMPLETED:
                        return queued_request.result

                    elif queued_request.status == RequestStatus.FAILED:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Request failed: {queued_request.error}",
                        )

                    # Still processing, wait a bit
                    await asyncio.sleep(0.1)

                # Timeout
                raise HTTPException(status_code=408, detail="Request timeout")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Request processing failed: {str(e)}"
        )


@app.get("/requests/{request_id}")
async def get_request_status(request_id: str):
    """
    Get the status of a specific request.

    Args:
        request_id: The request ID to look up

    Returns:
        Request status information
    """
    queued_request = request_queue.get_request_status(request_id)

    if queued_request is None:
        raise HTTPException(status_code=404, detail="Request not found")

    return {
        "id": queued_request.id,
        "status": queued_request.status.value,
        "created_at": queued_request.created_at,
        "started_at": queued_request.started_at,
        "completed_at": queued_request.completed_at,
        "wait_time": queued_request.wait_time,
        "processing_time": queued_request.processing_time,
        "total_time": queued_request.total_time,
        "error": queued_request.error,
    }


@app.get("/queue")
async def get_queue_status():
    """
    Get detailed queue status and statistics.

    Returns:
        Queue metrics and current status - continuous, batched, or sequential based on mode
    """
    if QUEUE_MODE == "continuous":
        stats = continuous_scheduler.stats
        stats["mode"] = "continuous"
        stats["part"] = 8
        return stats
    elif QUEUE_MODE == "batched":
        stats = batched_request_queue.stats
        stats["mode"] = "batched"
        stats["part"] = 7
        return stats
    else:
        stats = request_queue.stats
        stats["mode"] = "sequential"
        stats["part"] = 6
        return stats


@app.get("/queue/recent")
async def get_recent_completions(limit: int = 20):
    """
    Get recently completed requests with their prompts and responses.

    Args:
        limit: Maximum number of recent completions to return (default: 20)

    Returns:
        List of recent completion data including prompts and responses from appropriate queue
    """
    if QUEUE_MODE == "continuous":
        return continuous_scheduler.get_recent_completions(limit)
    elif QUEUE_MODE == "batched":
        return batched_request_queue.get_recent_completions(limit)
    else:
        return request_queue.get_recent_completions(limit)


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
