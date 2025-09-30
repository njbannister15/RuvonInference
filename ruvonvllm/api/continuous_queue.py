"""
True continuous batching request queue for Part 8.

This implements dynamic batch scheduling where requests can join and leave
batches during generation, enabling optimal GPU utilization and throughput.
"""

import time
import uuid
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from queue import Queue
import threading


logger = logging.getLogger(__name__)


class RequestState(Enum):
    """State of a request in continuous batching."""

    WAITING = "waiting"  # Waiting to join a batch
    ACTIVE = "active"  # Currently generating in batch
    COMPLETED = "completed"  # Generation finished
    FAILED = "failed"  # Request failed


@dataclass
class ContinuousRequest:
    """
    A request in the continuous batching system.

    Unlike prefill batching, each request has independent lifecycle
    and can join/leave batches at different times.
    """

    id: str
    request_data: Any
    state: RequestState
    created_at: float

    # Generation tracking
    input_tokens: List[int] = field(default_factory=list)
    generated_tokens: List[int] = field(default_factory=list)
    max_tokens: int = 20
    current_length: int = 0

    # Batch participation
    joined_batch_at: Optional[float] = None
    batch_position: Optional[int] = None
    generation_step: int = 0

    # Completion tracking
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    # Sampling parameters
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    use_cache: bool = True

    @property
    def is_finished(self) -> bool:
        """Check if request has finished generating."""
        return len(self.generated_tokens) >= self.max_tokens or self.state in [
            RequestState.COMPLETED,
            RequestState.FAILED,
        ]

    @property
    def total_length(self) -> int:
        """Total sequence length (input + generated)."""
        return len(self.input_tokens) + len(self.generated_tokens)

    @property
    def wait_time(self) -> Optional[float]:
        """Time spent waiting before joining batch."""
        if self.joined_batch_at is None:
            return None
        return self.joined_batch_at - self.created_at

    @property
    def generation_time(self) -> Optional[float]:
        """Time spent in generation."""
        if self.joined_batch_at is None or self.completed_at is None:
            return None
        return self.completed_at - self.joined_batch_at


@dataclass
class DynamicBatch:
    """
    A batch that can dynamically change composition during generation.

    This is the core of continuous batching - maintaining a batch that
    can add new requests and remove completed ones at each generation step.
    """

    id: str
    requests: Dict[str, ContinuousRequest]
    created_at: float
    generation_step: int = 0
    past_key_values: Optional[Any] = None

    @property
    def size(self) -> int:
        """Current number of active requests in batch."""
        return len(self.requests)

    @property
    def max_sequence_length(self) -> int:
        """Maximum sequence length across all requests."""
        if not self.requests:
            return 0
        return max(req.total_length for req in self.requests.values())

    def add_request(self, request: ContinuousRequest) -> None:
        """Add a new request to the batch."""
        request.state = RequestState.ACTIVE
        request.joined_batch_at = time.time()
        request.batch_position = len(self.requests)
        request.generation_step = self.generation_step
        self.requests[request.id] = request

        logger.info(
            f"Request {request.id} joined batch {self.id} at step {self.generation_step}"
        )

    def remove_request(self, request_id: str) -> Optional[ContinuousRequest]:
        """Remove a completed request from the batch."""
        if request_id in self.requests:
            request = self.requests.pop(request_id)
            request.completed_at = time.time()
            logger.info(
                f"Request {request_id} left batch {self.id} at step {self.generation_step}"
            )
            return request
        return None

    def get_active_requests(self) -> List[ContinuousRequest]:
        """Get list of currently active requests."""
        return [req for req in self.requests.values() if not req.is_finished]

    def get_finished_requests(self) -> List[ContinuousRequest]:
        """Get list of finished requests to remove."""
        return [req for req in self.requests.values() if req.is_finished]


class ContinuousBatchScheduler:
    """
    Dynamic batch scheduler implementing true continuous batching.

    This scheduler maintains a dynamic batch where:
    1. New requests can join at any time
    2. Completed requests leave immediately
    3. Generation continues with changing batch composition
    4. Optimal GPU utilization through dynamic sizing
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_sequence_length: int = 512,
        generation_interval: float = 0.01,  # 10ms between generation steps
    ):
        """
        Initialize the continuous batch scheduler.

        Args:
            max_batch_size: Maximum number of requests in batch
            max_sequence_length: Maximum sequence length to handle
            generation_interval: Time between generation steps
        """
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.generation_interval = generation_interval

        # Request management
        self.waiting_queue: Queue[ContinuousRequest] = Queue()
        self.active_requests: Dict[str, ContinuousRequest] = {}
        self.completed_requests: List[ContinuousRequest] = []

        # Batch management
        self.current_batch: Optional[DynamicBatch] = None
        self.batch_counter = 0

        # Control
        self._shutdown = False
        self._lock = threading.Lock()

        # Statistics
        self.total_requests = 0
        self.total_completed = 0
        self.total_failed = 0
        self.total_generation_steps = 0

    def add_request(self, request_data: Any) -> str:
        """
        Add a new request to the continuous batching system.

        Args:
            request_data: The completion request data

        Returns:
            Unique request ID for tracking
        """
        request_id = str(uuid.uuid4())

        # Extract parameters from request data
        max_tokens = getattr(request_data, "max_tokens", 20)
        temperature = getattr(request_data, "temperature", 1.0)
        top_k = getattr(request_data, "top_k", None)
        top_p = getattr(request_data, "top_p", None)
        use_cache = getattr(request_data, "use_cache", True)

        continuous_request = ContinuousRequest(
            id=request_id,
            request_data=request_data,
            state=RequestState.WAITING,
            created_at=time.time(),
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=use_cache,
        )

        with self._lock:
            self.waiting_queue.put(continuous_request)
            self.total_requests += 1

        logger.info(f"Added request {request_id} to continuous batching queue")
        return request_id

    def get_request_status(self, request_id: str) -> Optional[ContinuousRequest]:
        """Get the current status of a request."""
        with self._lock:
            # Check active requests
            if request_id in self.active_requests:
                return self.active_requests[request_id]

            # Check completed requests
            for req in self.completed_requests:
                if req.id == request_id:
                    return req

            # Check current batch
            if self.current_batch and request_id in self.current_batch.requests:
                return self.current_batch.requests[request_id]

        return None

    def add_waiting_requests_to_batch(self) -> int:
        """
        Add waiting requests to the current batch.

        Returns:
            Number of requests added
        """
        if not self.current_batch:
            # Create new batch if none exists
            self.batch_counter += 1
            self.current_batch = DynamicBatch(
                id=f"batch-{self.batch_counter}",
                requests={},
                created_at=time.time(),
            )

        added_count = 0
        max_to_add = self.max_batch_size - self.current_batch.size

        # Add requests from waiting queue
        while added_count < max_to_add and not self.waiting_queue.empty():
            try:
                request = self.waiting_queue.get_nowait()
                self.current_batch.add_request(request)
                self.active_requests[request.id] = request
                added_count += 1
            except Exception:
                break

        return added_count

    def remove_completed_requests(self) -> List[ContinuousRequest]:
        """
        Remove completed requests from the current batch.

        Returns:
            List of completed requests
        """
        if not self.current_batch:
            return []

        completed = []
        finished_requests = self.current_batch.get_finished_requests()

        for request in finished_requests:
            removed_request = self.current_batch.remove_request(request.id)
            if removed_request:
                removed_request.state = RequestState.COMPLETED
                completed.append(removed_request)

                # Move from active to completed
                if request.id in self.active_requests:
                    del self.active_requests[request.id]
                self.completed_requests.append(removed_request)
                self.total_completed += 1

        # If batch is empty, clean it up
        if self.current_batch.size == 0:
            logger.info(
                f"Batch {self.current_batch.id} completed after {self.current_batch.generation_step} steps"
            )
            self.current_batch = None

        return completed

    @property
    def stats(self) -> Dict[str, Any]:
        """Get comprehensive continuous batching statistics."""
        with self._lock:
            active_count = len(self.active_requests)
            waiting_count = self.waiting_queue.qsize()

            # Batch statistics
            current_batch_size = self.current_batch.size if self.current_batch else 0
            current_generation_step = (
                self.current_batch.generation_step if self.current_batch else 0
            )

            # Performance metrics
            completed_recent = (
                self.completed_requests[-20:] if self.completed_requests else []
            )
            avg_wait_time = None
            avg_generation_time = None

            if completed_recent:
                wait_times = [
                    r.wait_time for r in completed_recent if r.wait_time is not None
                ]
                gen_times = [
                    r.generation_time
                    for r in completed_recent
                    if r.generation_time is not None
                ]

                if wait_times:
                    avg_wait_time = sum(wait_times) / len(wait_times)
                if gen_times:
                    avg_generation_time = sum(gen_times) / len(gen_times)

            return {
                # Queue metrics
                "waiting_requests": waiting_count,
                "active_requests": active_count,
                "total_requests": self.total_requests,
                "total_completed": self.total_completed,
                "total_failed": self.total_failed,
                # Batch metrics
                "current_batch_size": current_batch_size,
                "current_generation_step": current_generation_step,
                "total_generation_steps": self.total_generation_steps,
                "max_batch_size": self.max_batch_size,
                # Performance metrics
                "average_wait_time": avg_wait_time,
                "average_generation_time": avg_generation_time,
                # Mode identification
                "mode": "continuous",
                "part": 8,
            }

    def get_recent_completions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recently completed requests."""
        with self._lock:
            recent = self.completed_requests[-limit:] if self.completed_requests else []

            results = []
            for request in reversed(recent):  # Newest first
                try:
                    # Extract prompt and response
                    prompt = "Unknown"
                    response = "No response"

                    if hasattr(request.request_data, "prompt"):
                        prompt = request.request_data.prompt
                    elif (
                        isinstance(request.request_data, dict)
                        and "prompt" in request.request_data
                    ):
                        prompt = request.request_data["prompt"]

                    if request.state == RequestState.COMPLETED and request.result:
                        # Handle both dict and Pydantic model formats
                        if (
                            isinstance(request.result, dict)
                            and "choices" in request.result
                        ):
                            choices = request.result["choices"]
                            if choices and "text" in choices[0]:
                                response = choices[0]["text"]
                        elif (
                            hasattr(request.result, "choices")
                            and request.result.choices
                        ):
                            if hasattr(request.result.choices[0], "text"):
                                response = request.result.choices[0].text
                    elif request.state == RequestState.FAILED:
                        response = f"Error: {request.error or 'Unknown error'}"

                    results.append(
                        {
                            "id": request.id,
                            "prompt": prompt,
                            "response": response,
                            "status": request.state.value,
                            "completed_at": request.completed_at,
                            "wait_time": request.wait_time,
                            "generation_time": request.generation_time,
                            "generation_step": request.generation_step,
                            "total_tokens": len(request.generated_tokens),
                        }
                    )

                except Exception as e:
                    results.append(
                        {
                            "id": request.id,
                            "prompt": "Parse error",
                            "response": f"Error: {str(e)}",
                            "status": request.state.value,
                            "completed_at": request.completed_at,
                        }
                    )

            return results

    async def continuous_generation_loop(self):
        """
        Main continuous generation loop.

        This is the heart of continuous batching - a loop that:
        1. Adds new requests to the current batch
        2. Generates next token for all active requests
        3. Removes completed requests
        4. Repeats with dynamic batch composition
        """
        from ruvonvllm.model.gpt2 import GPT2Model
        from ruvonvllm.tokenizer.gpt2_tokenizer import GPT2TokenizerWrapper

        # Load model and tokenizer once
        model = GPT2Model("gpt2", device="cpu")
        model.load_model()
        tokenizer = GPT2TokenizerWrapper("gpt2")

        logger.info("🚀 Starting continuous generation loop")

        past_key_values = None
        generation_step_counter = 0

        while not self._shutdown:
            try:
                # 1. Add waiting requests to current batch
                added_count = self.add_waiting_requests_to_batch()
                if added_count > 0:
                    logger.info(f"Added {added_count} new requests to batch")
                    # Reset cache when batch composition changes significantly
                    past_key_values = None

                # 2. Check if we have any active requests
                if not self.current_batch or self.current_batch.size == 0:
                    await asyncio.sleep(self.generation_interval)
                    continue

                # 3. Prepare requests for generation
                active_requests = self.current_batch.get_active_requests()
                if not active_requests:
                    await asyncio.sleep(self.generation_interval)
                    continue

                # 4. Tokenize input for new requests (only on their first step)
                for request in active_requests:
                    if not request.input_tokens and hasattr(
                        request.request_data, "prompt"
                    ):
                        # Tokenize the prompt
                        input_ids = tokenizer.encode(
                            request.request_data.prompt, return_tensors=True
                        )
                        request.input_tokens = input_ids.squeeze().tolist()

                # 5. Generate next token for all active requests
                try:
                    next_tokens, past_key_values, finished_flags = (
                        model.generate_continuous_step(
                            active_requests=active_requests,
                            past_key_values=past_key_values,
                            show_progress=False,  # Set to True for debugging
                        )
                    )

                    generation_step_counter += 1
                    self.total_generation_steps += 1

                    if self.current_batch:
                        self.current_batch.generation_step += 1

                    # Debug logging
                    if generation_step_counter % 10 == 0:
                        logger.info(
                            f"Generation step {generation_step_counter}: "
                            f"{len(active_requests)} active requests, "
                            f"batch step {self.current_batch.generation_step if self.current_batch else 0}"
                        )

                except Exception as e:
                    logger.error(f"Generation step failed: {e}")
                    # Mark all requests as failed
                    for request in active_requests:
                        request.state = RequestState.FAILED
                        request.error = str(e)
                    await asyncio.sleep(self.generation_interval)
                    continue

                # 6. Remove completed requests and create results
                completed_requests = self.remove_completed_requests()
                for request in completed_requests:
                    # Create result for completed request
                    if request.state == RequestState.COMPLETED:
                        # Decode the generated text
                        full_tokens = request.input_tokens + request.generated_tokens
                        full_text = tokenizer.decode(full_tokens)
                        generated_text = full_text[len(request.request_data.prompt) :]

                        # Create response in expected format
                        from ruvonvllm.api.server import (
                            CompletionResponse,
                            CompletionChoice,
                        )

                        result = CompletionResponse(
                            id=f"cmpl-{int(time.time())}-{request.id[:8]}",
                            created=int(time.time()),
                            model="gpt2",
                            choices=[
                                CompletionChoice(
                                    text=generated_text,
                                    index=0,
                                    finish_reason="stop"
                                    if len(request.generated_tokens)
                                    < request.max_tokens
                                    else "length",
                                )
                            ],
                            usage={
                                "prompt_tokens": len(request.input_tokens),
                                "completion_tokens": len(request.generated_tokens),
                                "total_tokens": len(request.input_tokens)
                                + len(request.generated_tokens),
                            },
                        )
                        request.result = result

                # 7. Handle cache invalidation for dynamic batch changes
                if completed_requests:
                    # If requests completed, we need to reconstruct the cache
                    # This is a simplification - production systems use more sophisticated cache management
                    past_key_values = None

                # 8. Sleep briefly before next generation step
                await asyncio.sleep(self.generation_interval)

            except Exception as e:
                logger.error(f"Continuous generation loop error: {e}")
                await asyncio.sleep(1.0)  # Longer sleep on unexpected errors

        logger.info("Continuous generation loop stopped")

    def shutdown(self):
        """Gracefully shutdown the continuous scheduler."""
        self._shutdown = True
        logger.info("Continuous batch scheduler shutdown initiated")


# Global continuous batching scheduler
continuous_scheduler = ContinuousBatchScheduler(
    max_batch_size=8,  # Up to 8 requests in dynamic batch
    max_sequence_length=512,  # Handle longer sequences
    generation_interval=0.01,  # 10ms between generation steps for responsiveness
)
