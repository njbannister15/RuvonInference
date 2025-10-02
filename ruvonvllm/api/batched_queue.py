"""
Continuous batching request queue for Part 7.

This module implements batched request processing, allowing multiple requests
to be processed simultaneously in a single model forward pass for improved throughput.
"""

import time
import uuid
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
from queue import Queue
import threading

logger = logging.getLogger(__name__)


class BatchStatus(Enum):
    """Status of a batch in the processing system."""

    COLLECTING = "collecting"  # Gathering requests for the batch
    PROCESSING = "processing"  # Running model inference
    COMPLETED = "completed"  # All requests in batch finished
    FAILED = "failed"  # Batch processing failed


@dataclass
class BatchedRequest:
    """
    A request in the batched processing queue.

    Similar to QueuedRequest but designed for batch processing.
    """

    id: str
    request_data: Any  # CompletionRequest object
    status: str
    created_at: float
    batch_id: Optional[str] = None
    batch_position: Optional[int] = None  # Position within batch
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    @property
    def wait_time(self) -> Optional[float]:
        """Time spent waiting before processing started."""
        if self.started_at is None:
            return None
        return self.started_at - self.created_at

    @property
    def processing_time(self) -> Optional[float]:
        """Time spent in batch processing."""
        if self.started_at is None or self.completed_at is None:
            return None
        return self.completed_at - self.started_at

    @property
    def total_time(self) -> Optional[float]:
        """Total time from creation to completion."""
        if self.completed_at is None:
            return None
        return self.completed_at - self.created_at


@dataclass
class RequestBatch:
    """
    A batch of requests to be processed together.
    """

    id: str
    requests: List[BatchedRequest]
    status: BatchStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def size(self) -> int:
        """Number of requests in this batch."""
        return len(self.requests)

    @property
    def processing_time(self) -> Optional[float]:
        """Time spent processing this batch."""
        if self.started_at is None or self.completed_at is None:
            return None
        return self.completed_at - self.started_at


class BatchedRequestQueue:
    """
    Continuous batching request queue manager.

    This queue collects multiple requests and processes them together in batches
    to improve throughput while maintaining low latency. Key features:

    1. Dynamic batch sizing based on queue depth and timing
    2. Configurable max batch size and wait times
    3. Automatic batch formation and processing
    4. Detailed monitoring of batch performance
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_time: float = 0.1,  # 100ms max wait to form batch
        min_batch_size: int = 1,
    ):
        """
        Initialize the batched request queue.

        Args:
            max_batch_size: Maximum number of requests per batch
            max_wait_time: Maximum time to wait before processing partial batch
            min_batch_size: Minimum batch size (1 for immediate processing)
        """
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.min_batch_size = min_batch_size

        # Request management
        self._queue: Queue[BatchedRequest] = Queue()
        self._requests: Dict[str, BatchedRequest] = {}
        self._lock = threading.Lock()

        # Batch management
        self._current_batches: Dict[str, RequestBatch] = {}
        self._processing_batches: Dict[str, RequestBatch] = {}
        self._completed_batches: List[RequestBatch] = []

        # Statistics
        self._total_requests = 0
        self._total_batches = 0
        self._total_processed = 0
        self._total_failed = 0

        # Control flags
        self._shutdown = False

    def add_request(self, request_data: Any) -> str:
        """
        Add a new request to the queue.

        Args:
            request_data: The completion request data

        Returns:
            Unique request ID for tracking
        """
        request_id = str(uuid.uuid4())
        batched_request = BatchedRequest(
            id=request_id,
            request_data=request_data,
            status="queued",
            created_at=time.time(),
        )

        with self._lock:
            self._requests[request_id] = batched_request
            self._queue.put(batched_request)
            self._total_requests += 1

        logger.info(
            f"Added request {request_id} to batched queue. Queue size: {self.queue_size}"
        )
        return request_id

    def get_request_status(self, request_id: str) -> Optional[BatchedRequest]:
        """Get the status of a specific request."""
        with self._lock:
            return self._requests.get(request_id)

    def collect_next_batch(self) -> Optional[RequestBatch]:
        """
        Collect the next batch of requests for processing.

        This implements the core batching logic:
        1. Try to collect up to max_batch_size requests
        2. Wait up to max_wait_time for more requests
        3. Process when batch is full or timeout reached

        Returns:
            RequestBatch ready for processing, or None if no requests
        """
        batch_requests = []
        start_time = time.time()

        # Collect initial requests (non-blocking)
        while len(batch_requests) < self.max_batch_size:
            try:
                request = self._queue.get_nowait()
                batch_requests.append(request)
            except Exception:
                break  # No more requests available immediately

        # If we have some requests but batch isn't full, wait a bit for more
        if batch_requests and len(batch_requests) < self.max_batch_size:
            remaining_wait = self.max_wait_time - (time.time() - start_time)

            while remaining_wait > 0 and len(batch_requests) < self.max_batch_size:
                try:
                    # Wait briefly for more requests
                    time.sleep(min(0.01, remaining_wait))  # 10ms or remaining time

                    # Check for new requests
                    while len(batch_requests) < self.max_batch_size:
                        try:
                            request = self._queue.get_nowait()
                            batch_requests.append(request)
                        except Exception:
                            break

                    remaining_wait = self.max_wait_time - (time.time() - start_time)

                except Exception:
                    break

        # Create batch if we have enough requests
        if len(batch_requests) >= self.min_batch_size:
            batch_id = str(uuid.uuid4())
            batch = RequestBatch(
                id=batch_id,
                requests=batch_requests,
                status=BatchStatus.COLLECTING,
                created_at=time.time(),
            )

            # Update request statuses and assign batch info
            with self._lock:
                for i, request in enumerate(batch_requests):
                    request.batch_id = batch_id
                    request.batch_position = i
                    request.status = "batched"

                self._current_batches[batch_id] = batch
                self._total_batches += 1

            logger.info(
                f"Created batch {batch_id} with {len(batch_requests)} requests "
                f"(waited {time.time() - start_time:.3f}s)"
            )

            return batch

        return None

    def start_batch_processing(self, batch: RequestBatch) -> None:
        """Mark a batch as started processing."""
        with self._lock:
            batch.status = BatchStatus.PROCESSING
            batch.started_at = time.time()

            # Move to processing batches
            if batch.id in self._current_batches:
                del self._current_batches[batch.id]
            self._processing_batches[batch.id] = batch

            # Update all request statuses
            for request in batch.requests:
                request.status = "processing"
                request.started_at = batch.started_at

            logger.info(
                f"Started processing batch {batch.id} with {batch.size} requests"
            )

    def complete_batch(self, batch_id: str, results: List[Any]) -> None:
        """
        Mark a batch as completed successfully.

        Args:
            batch_id: The batch ID that completed
            results: List of results, one per request in the batch
        """
        with self._lock:
            if batch_id not in self._processing_batches:
                logger.error(
                    f"Batch {batch_id} not found in processing batches {self._processing_batches}"
                )
                return

            batch = self._processing_batches[batch_id]
            batch.status = BatchStatus.COMPLETED
            batch.completed_at = time.time()

            # Update all requests with their results
            for i, (request, result) in enumerate(zip(batch.requests, results)):
                request.status = "completed"
                request.completed_at = batch.completed_at
                request.result = result
                self._total_processed += 1

            # Move to completed batches
            del self._processing_batches[batch_id]
            self._completed_batches.append(batch)

            # Keep only recent completed batches (last 100)
            if len(self._completed_batches) > 100:
                self._completed_batches = self._completed_batches[-100:]

            logger.info(
                f"Completed batch {batch_id}. "
                f"Batch time: {batch.processing_time:.3f}s, "
                f"Requests: {batch.size}"
            )

    def fail_batch(self, batch_id: str, error: str) -> None:
        """
        Mark a batch as failed.

        Args:
            batch_id: The batch ID that failed
            error: Error message
        """
        with self._lock:
            if batch_id not in self._processing_batches:
                logger.error(f"Batch {batch_id} not found in processing batches")
                return

            batch = self._processing_batches[batch_id]
            batch.status = BatchStatus.FAILED
            batch.completed_at = time.time()

            # Mark all requests as failed
            for request in batch.requests:
                request.status = "failed"
                request.completed_at = batch.completed_at
                request.error = error
                self._total_failed += len(batch.requests)

            # Move to completed batches
            del self._processing_batches[batch_id]
            self._completed_batches.append(batch)

            logger.error(f"Failed batch {batch_id}: {error}")

    @property
    def queue_size(self) -> int:
        """Current number of requests in queue."""
        return self._queue.qsize()

    @property
    def processing_request_count(self) -> int:
        """Number of requests currently being processed in batches."""
        with self._lock:
            return sum(batch.size for batch in self._processing_batches.values())

    @property
    def stats(self) -> Dict[str, Any]:
        """
        Get comprehensive queue and batch statistics.

        Returns:
            Dictionary with queue, batch, and performance metrics
        """
        with self._lock:
            # Count requests by status
            queued_count = sum(
                1 for r in self._requests.values() if r.status == "queued"
            )

            processing_count = sum(
                1
                for r in self._requests.values()
                if r.status in ["batched", "processing"]
            )

            # Calculate batch statistics
            recent_batches = (
                self._completed_batches[-20:] if self._completed_batches else []
            )
            avg_batch_size = None
            avg_batch_time = None
            avg_wait_time = None

            if recent_batches:
                avg_batch_size = sum(b.size for b in recent_batches) / len(
                    recent_batches
                )
                avg_batch_time = sum(
                    b.processing_time for b in recent_batches if b.processing_time
                ) / len([b for b in recent_batches if b.processing_time])

                # Calculate average wait time from completed requests
                completed_requests = [
                    r
                    for batch in recent_batches
                    for r in batch.requests
                    if r.wait_time is not None
                ]
                if completed_requests:
                    avg_wait_time = sum(r.wait_time for r in completed_requests) / len(
                        completed_requests
                    )

            return {
                # Queue metrics
                "queue_size": self.queue_size,
                "queued_requests": queued_count,
                "processing_requests": processing_count,
                "total_requests": self._total_requests,
                "total_processed": self._total_processed,
                "total_failed": self._total_failed,
                # Batch metrics
                "active_batches": len(self._processing_batches),
                "total_batches": self._total_batches,
                "max_batch_size": self.max_batch_size,
                "current_batch_sizes": [
                    b.size for b in self._processing_batches.values()
                ],
                # Performance metrics
                "average_batch_size": avg_batch_size,
                "average_batch_time": avg_batch_time,
                "average_wait_time": avg_wait_time,
            }

    def get_recent_completions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recently completed requests from batches.

        Args:
            limit: Maximum number of recent completions to return

        Returns:
            List of completion data including batch information
        """
        with self._lock:
            # Collect completed requests from recent batches
            completed_requests = []
            for batch in reversed(self._completed_batches[-10:]):  # Last 10 batches
                for request in batch.requests:
                    if request.status in ["completed", "failed"]:
                        completed_requests.append((request, batch))

            # Sort by completion time (newest first)
            completed_requests.sort(key=lambda x: x[0].completed_at or 0, reverse=True)

            # Format for monitor display
            results = []
            for request, batch in completed_requests[:limit]:
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

                    if request.status == "completed" and request.result:
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
                    elif request.status == "failed":
                        response = f"Error: {request.error or 'Unknown error'}"

                    results.append(
                        {
                            "id": request.id,
                            "prompt": prompt,
                            "response": response,
                            "status": request.status,
                            "completed_at": request.completed_at,
                            "wait_time": request.wait_time,
                            "processing_time": request.processing_time,
                            "total_time": request.total_time,
                            "batch_id": request.batch_id,
                            "batch_size": batch.size,
                            "batch_position": request.batch_position,
                        }
                    )

                except Exception as e:
                    results.append(
                        {
                            "id": request.id,
                            "prompt": "Parse error",
                            "response": f"Error: {str(e)}",
                            "status": request.status,
                            "completed_at": request.completed_at,
                            "batch_id": request.batch_id,
                        }
                    )

            return results

    def shutdown(self):
        """Gracefully shutdown the batched queue."""
        self._shutdown = True
        logger.info("Batched request queue shutdown initiated")


# Global batched request queue instance
batched_request_queue = BatchedRequestQueue(
    max_batch_size=32,  # Process up to 32 requests together
    max_wait_time=0.1,  # Wait max 100ms to form larger batches
    min_batch_size=1,  # Process single requests immediately if queue is empty
)
