"""
Request queue management for sequential request processing.

This module implements Part 6's request queuing system, allowing the API server
to handle multiple requests in a first-in-first-out (FIFO) order while processing
them sequentially to avoid resource conflicts.
"""

import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Any
from queue import Queue
import threading
import logging

logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    """Status of a request in the queue."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class QueuedRequest:
    """
    A request in the processing queue.

    This tracks everything needed to process a request and return the result
    to the correct client, including timing information for monitoring.
    """

    id: str
    request_data: Any  # CompletionRequest object
    status: RequestStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    @property
    def wait_time(self) -> Optional[float]:
        """Time spent waiting in queue before processing started."""
        if self.started_at is None:
            return None
        return self.started_at - self.created_at

    @property
    def processing_time(self) -> Optional[float]:
        """Time spent actually processing the request."""
        if self.started_at is None or self.completed_at is None:
            return None
        return self.completed_at - self.started_at

    @property
    def total_time(self) -> Optional[float]:
        """Total time from request creation to completion."""
        if self.completed_at is None:
            return None
        return self.completed_at - self.created_at


class RequestQueue:
    """
    Sequential request queue manager.

    This class manages a FIFO queue of requests and processes them one at a time.
    It provides thread-safe operations and detailed monitoring capabilities.

    The design philosophy is simple:
    1. Accept requests and assign them unique IDs
    2. Add them to a FIFO queue
    3. Process them sequentially (one at a time)
    4. Track status and timing for monitoring
    """

    def __init__(self):
        """Initialize the request queue."""
        self._queue: Queue[QueuedRequest] = Queue()
        self._requests: Dict[str, QueuedRequest] = {}
        self._current_request: Optional[QueuedRequest] = None
        self._lock = threading.Lock()
        self._processing = False
        self._total_processed = 0
        self._total_failed = 0

    def add_request(self, request_data: Any) -> str:
        """
        Add a new request to the queue.

        Args:
            request_data: The completion request data

        Returns:
            Unique request ID for tracking
        """
        request_id = str(uuid.uuid4())
        queued_request = QueuedRequest(
            id=request_id,
            request_data=request_data,
            status=RequestStatus.QUEUED,
            created_at=time.time(),
        )

        with self._lock:
            self._requests[request_id] = queued_request
            self._queue.put(queued_request)

        logger.info(
            f"Added request {request_id} to queue. Queue size: {self.queue_size}"
        )
        return request_id

    def get_request_status(self, request_id: str) -> Optional[QueuedRequest]:
        """
        Get the status of a specific request.

        Args:
            request_id: The request ID to look up

        Returns:
            QueuedRequest object or None if not found
        """
        with self._lock:
            return self._requests.get(request_id)

    def start_processing(self, request_id: str) -> None:
        """
        Mark a request as started processing.

        Args:
            request_id: The request ID that started processing
        """
        with self._lock:
            if request_id in self._requests:
                request = self._requests[request_id]
                request.status = RequestStatus.PROCESSING
                request.started_at = time.time()
                self._current_request = request
                logger.info(f"Started processing request {request_id}")

    def complete_request(self, request_id: str, result: Any) -> None:
        """
        Mark a request as completed successfully.

        Args:
            request_id: The request ID that completed
            result: The result data
        """
        with self._lock:
            if request_id in self._requests:
                request = self._requests[request_id]
                request.status = RequestStatus.COMPLETED
                request.completed_at = time.time()
                request.result = result
                self._current_request = None
                self._total_processed += 1

                logger.info(
                    f"Completed request {request_id}. "
                    f"Wait: {request.wait_time:.3f}s, "
                    f"Process: {request.processing_time:.3f}s, "
                    f"Total: {request.total_time:.3f}s"
                )

    def fail_request(self, request_id: str, error: str) -> None:
        """
        Mark a request as failed.

        Args:
            request_id: The request ID that failed
            error: Error message
        """
        with self._lock:
            if request_id in self._requests:
                request = self._requests[request_id]
                request.status = RequestStatus.FAILED
                request.completed_at = time.time()
                request.error = error
                self._current_request = None
                self._total_failed += 1

                logger.error(
                    f"Failed request {request_id}: {error}. "
                    f"Wait: {request.wait_time:.3f}s"
                )

    def get_next_request(self) -> Optional[QueuedRequest]:
        """
        Get the next request to process.

        Returns:
            Next queued request or None if queue is empty
        """
        try:
            return self._queue.get_nowait()
        except Exception:
            return None

    @property
    def queue_size(self) -> int:
        """Current number of requests in queue."""
        return self._queue.qsize()

    @property
    def current_request(self) -> Optional[QueuedRequest]:
        """Currently processing request."""
        with self._lock:
            return self._current_request

    @property
    def stats(self) -> Dict[str, Any]:
        """
        Get queue statistics.

        Returns:
            Dictionary with queue metrics
        """
        with self._lock:
            queued_count = sum(
                1 for r in self._requests.values() if r.status == RequestStatus.QUEUED
            )
            processing_count = sum(
                1
                for r in self._requests.values()
                if r.status == RequestStatus.PROCESSING
            )

            # Calculate average wait times for completed requests
            completed_requests = [
                r
                for r in self._requests.values()
                if r.status == RequestStatus.COMPLETED
            ]
            avg_wait_time = None
            avg_processing_time = None

            if completed_requests:
                avg_wait_time = sum(r.wait_time for r in completed_requests) / len(
                    completed_requests
                )
                avg_processing_time = sum(
                    r.processing_time for r in completed_requests
                ) / len(completed_requests)

            return {
                "queue_size": self.queue_size,
                "queued_requests": queued_count,
                "processing_requests": processing_count,
                "total_processed": self._total_processed,
                "total_failed": self._total_failed,
                "average_wait_time": avg_wait_time,
                "average_processing_time": avg_processing_time,
                "current_request_id": self._current_request.id
                if self._current_request
                else None,
            }

    def get_recent_completions(self, limit: int = 20) -> list:
        """
        Get recently completed requests with their results.

        Args:
            limit: Maximum number of recent completions to return

        Returns:
            List of recent completion data including prompts and responses
        """
        with self._lock:
            # Get completed and failed requests
            completed_requests = [
                r
                for r in self._requests.values()
                if r.status in [RequestStatus.COMPLETED, RequestStatus.FAILED]
            ]

            # Sort by completion time (newest first)
            completed_requests.sort(key=lambda x: x.completed_at or 0, reverse=True)

            # Format for monitor display
            results = []
            for req in completed_requests[:limit]:
                try:
                    # Extract prompt and response from request data and result
                    prompt = "Unknown"
                    response = "No response"

                    if hasattr(req.request_data, "prompt"):
                        prompt = req.request_data.prompt
                    elif (
                        isinstance(req.request_data, dict)
                        and "prompt" in req.request_data
                    ):
                        prompt = req.request_data["prompt"]

                    if req.status == RequestStatus.COMPLETED and req.result:
                        # Handle both dict and Pydantic model formats
                        if isinstance(req.result, dict) and "choices" in req.result:
                            choices = req.result["choices"]
                            if choices and "text" in choices[0]:
                                response = choices[0]["text"]
                        elif hasattr(req.result, "choices") and req.result.choices:
                            # Pydantic model format
                            if hasattr(req.result.choices[0], "text"):
                                response = req.result.choices[0].text
                    elif req.status == RequestStatus.FAILED:
                        response = f"Error: {req.error or 'Unknown error'}"

                    results.append(
                        {
                            "id": req.id,
                            "prompt": prompt,
                            "response": response,
                            "status": req.status.value,
                            "completed_at": req.completed_at,
                            "wait_time": req.wait_time,
                            "processing_time": req.processing_time,
                            "total_time": req.total_time,
                        }
                    )
                except Exception as e:
                    # Fallback for any parsing errors
                    results.append(
                        {
                            "id": req.id,
                            "prompt": "Parse error",
                            "response": f"Error: {str(e)}",
                            "status": req.status.value,
                            "completed_at": req.completed_at,
                            "wait_time": req.wait_time,
                            "processing_time": req.processing_time,
                            "total_time": req.total_time,
                        }
                    )

            return results

    def cleanup_old_requests(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up old completed/failed requests to prevent memory leaks.

        Args:
            max_age_seconds: Maximum age of requests to keep

        Returns:
            Number of requests cleaned up
        """
        current_time = time.time()
        to_remove = []

        with self._lock:
            for request_id, request in self._requests.items():
                if (
                    request.status in [RequestStatus.COMPLETED, RequestStatus.FAILED]
                    and request.completed_at
                    and current_time - request.completed_at > max_age_seconds
                ):
                    to_remove.append(request_id)

            for request_id in to_remove:
                del self._requests[request_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old requests")

        return len(to_remove)


# Global request queue instance
request_queue = RequestQueue()
