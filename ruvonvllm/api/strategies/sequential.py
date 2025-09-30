"""
Sequential queue processing strategy (Part 6).

This strategy processes requests one at a time in the order they arrive,
providing simple and predictable behavior for basic inference scenarios.
"""

import time
import asyncio
from typing import Dict, Any, List, TYPE_CHECKING

from .base import QueueStrategy
from ruvonvllm.api.queue import request_queue

if TYPE_CHECKING:
    from ruvonvllm.api.server import CompletionRequest, CompletionResponse


class SequentialQueueStrategy(QueueStrategy):
    """
    Sequential processing strategy that handles one request at a time.

    This is the simplest queue strategy, processing requests in FIFO order
    without any batching optimizations. It provides:
    - Predictable latency per request
    - Simple resource management
    - Easy debugging and monitoring
    - Guaranteed sequential execution

    Ideal for:
    - Development and testing
    - Low-throughput scenarios
    - When predictable timing is important
    """

    def __init__(self):
        """Initialize the sequential queue strategy."""
        self._queue = request_queue

    async def process_request(
        self, request: "CompletionRequest"
    ) -> "CompletionResponse":
        """
        Process a single request through the sequential queue.

        The request is added to the queue and processed when its turn comes.
        This ensures fair ordering and prevents resource conflicts.

        Args:
            request: The completion request to process

        Returns:
            CompletionResponse with generated text

        Raises:
            Exception: If request processing fails or times out
        """
        # Add request to sequential queue
        request_id = self._queue.add_request(request)

        # Wait for request to complete with timeout
        max_wait_time = 300  # 5 minutes timeout
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            queued_request = self._queue.get_request_status(request_id)

            if queued_request is None:
                raise Exception("Request not found in queue")

            # Import here to avoid circular imports
            from ruvonvllm.api.queue import RequestStatus

            if queued_request.status == RequestStatus.COMPLETED:
                return queued_request.result

            elif queued_request.status == RequestStatus.FAILED:
                raise Exception(f"Request failed: {queued_request.error}")

            # Still processing, wait a bit
            await asyncio.sleep(0.1)

        # Request timed out
        raise Exception("Request timeout after 5 minutes")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics for the sequential queue.

        Returns:
            Dictionary containing sequential queue metrics and status
        """
        stats = self._queue.stats
        stats["mode"] = "sequential"
        stats["part"] = self.part_number
        return stats

    def get_recent_completions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recently completed requests from the sequential queue.

        Args:
            limit: Maximum number of recent completions to return

        Returns:
            List of recent completion data including prompts and responses
        """
        return self._queue.get_recent_completions(limit)

    @property
    def strategy_name(self) -> str:
        """Get the human-readable name of this strategy."""
        return "Sequential"

    @property
    def part_number(self) -> int:
        """Get the part number this strategy represents."""
        return 6

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check for the sequential queue strategy.

        Returns:
            Health status information including queue statistics
        """
        base_health = super().health_check()

        # Add strategy-specific health information
        stats = self.get_stats()
        base_health.update(
            {
                "queue_size": stats.get("queue_size", 0),
                "processing_requests": stats.get("processing", 0),
                "total_processed": stats.get("completed", 0),
                "queue_processor_running": True,  # Since we have a background thread
            }
        )

        return base_health
