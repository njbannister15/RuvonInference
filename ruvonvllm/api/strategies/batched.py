"""
Batched queue processing strategy (Part 7 - Prefill Batching).

This strategy processes requests in static batches, waiting to collect multiple
requests before processing them together in a single model forward pass.
"""

import time
import asyncio
from typing import Dict, Any, List, TYPE_CHECKING

from .base import QueueStrategy
from ruvonvllm.api.batched_queue import batched_request_queue

if TYPE_CHECKING:
    from ruvonvllm.api.server import CompletionRequest, CompletionResponse


class BatchedQueueStrategy(QueueStrategy):
    """
    Batched processing strategy that collects requests into static batches.

    This strategy implements prefill batching, where requests are collected
    and processed together in fixed-size batches. This provides:
    - Higher throughput than sequential processing
    - Better GPU utilization through batching
    - Static batch composition (no dynamic changes)
    - Predictable batch processing patterns

    Key characteristics:
    - Collects up to max_batch_size requests
    - Waits up to max_wait_time for batch to fill
    - Processes entire batch in single model call
    - All requests in batch start and finish together

    Ideal for:
    - Medium to high throughput scenarios
    - When requests have similar generation lengths
    - Production serving with predictable load
    """

    def __init__(self):
        """Initialize the batched queue strategy."""
        self._queue = batched_request_queue

    async def process_request(
        self, request: "CompletionRequest"
    ) -> "CompletionResponse":
        """
        Process a single request through the batched queue.

        The request is added to the batched queue and will be processed
        when a batch is formed (either when batch is full or timeout occurs).

        Args:
            request: The completion request to process

        Returns:
            CompletionResponse with generated text

        Raises:
            Exception: If request processing fails or times out
        """
        # Add request to batched queue
        request_id = self._queue.add_request(request)

        # Wait for request to complete with timeout
        max_wait_time = 300  # 5 minutes timeout
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            batched_request = self._queue.get_request_status(request_id)

            if batched_request is None:
                raise Exception("Request not found in batched queue")

            if batched_request.status == "completed":
                return batched_request.result

            elif batched_request.status == "failed":
                raise Exception(f"Request failed: {batched_request.error}")

            # Still processing, wait a bit
            await asyncio.sleep(0.1)

        # Request timed out
        raise Exception("Request timeout after 5 minutes")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics for the batched queue.

        Returns:
            Dictionary containing batched queue metrics and status
        """
        stats = self._queue.stats
        stats["mode"] = "batched"
        stats["part"] = self.part_number
        return stats

    def get_recent_completions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recently completed requests from the batched queue.

        Args:
            limit: Maximum number of recent completions to return

        Returns:
            List of recent completion data including prompts and responses
        """
        return self._queue.get_recent_completions(limit)

    @property
    def strategy_name(self) -> str:
        """Get the human-readable name of this strategy."""
        return "Batched"

    @property
    def part_number(self) -> int:
        """Get the part number this strategy represents."""
        return 7

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check for the batched queue strategy.

        Returns:
            Health status information including batch statistics
        """
        base_health = super().health_check()

        # Add strategy-specific health information
        stats = self.get_stats()
        base_health.update(
            {
                "queue_size": stats.get("queue_size", 0),
                "processing_batches": stats.get("processing_batches", 0),
                "total_batches_processed": stats.get("total_batches", 0),
                "max_batch_size": stats.get("max_batch_size", 4),
                "average_batch_size": stats.get("average_batch_size", 0),
                "batch_processor_running": True,  # Since we have a background thread
            }
        )

        return base_health
