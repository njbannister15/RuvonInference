"""
Continuous queue processing strategy (Part 8 - True Continuous Batching).

This strategy processes requests with dynamic batching, where requests can
join and leave batches during generation for optimal GPU utilization.
"""

import time
import asyncio
from typing import Dict, Any, List, TYPE_CHECKING

from ruvonvllm.api.schemas.completions import CompletionRequest

from .base import QueueStrategy
from ruvonvllm.api.continuous_queue import continuous_scheduler

if TYPE_CHECKING:
    from ruvonvllm.api.schemas.completions import CompletionResponse


class ContinuousQueueStrategy(QueueStrategy):
    """
    Continuous processing strategy with dynamic batch scheduling.

    This strategy implements true continuous batching, where requests have
    independent lifecycles and can join/leave batches during generation. This provides:
    - Maximum GPU utilization
    - Optimal throughput for mixed workloads
    - Dynamic batch composition
    - Independent request completion timing

    Key characteristics:
    - Requests join active batches immediately
    - Generation continues with changing batch sizes
    - Requests complete independently when done
    - Optimal for varying generation lengths

    Ideal for:
    - High-throughput production serving
    - Mixed request types and lengths
    - Maximum resource efficiency
    - Real-time serving applications
    """

    def __init__(self):
        """Initialize the continuous queue strategy."""
        self._scheduler = continuous_scheduler

    async def process_request(
        self, request: "CompletionRequest"
    ) -> "CompletionResponse":
        """
        Process a single request through the continuous scheduler.

        The request is added to the continuous scheduler and will be processed
        in dynamic batches that can change composition during generation.

        Args:
            request: The completion request to process

        Returns:
            CompletionResponse with generated text

        Raises:
            Exception: If request processing fails or times out
        """
        # Add request to continuous scheduler
        request_id = self._scheduler.add_request(request)

        # Wait for request to complete with timeout
        max_wait_time = 300  # 5 minutes timeout
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            continuous_request = self._scheduler.get_request_status(request_id)

            if continuous_request is None:
                raise Exception("Request not found in continuous scheduler")

            if continuous_request["status"] == "completed":
                return continuous_request["result"]

            elif continuous_request["status"] == "failed":
                error_msg = continuous_request.get("error", "Unknown error")
                raise Exception(f"Request failed: {error_msg}")

            # Still processing, wait a bit
            await asyncio.sleep(0.1)

        # Request timed out
        raise Exception("Request timeout after 5 minutes")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics for the continuous scheduler.

        Returns:
            Dictionary containing continuous scheduler metrics and status
        """
        stats = self._scheduler.stats
        stats["mode"] = "continuous"
        stats["part"] = self.part_number
        stats["strategy_name"] = self.__class__.__name__
        return stats

    def get_recent_completions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recently completed requests from the continuous scheduler.

        Args:
            limit: Maximum number of recent completions to return

        Returns:
            List of recent completion data including prompts and responses
        """
        return self._scheduler.get_recent_completions(limit)

    @property
    def strategy_name(self) -> str:
        """Get the human-readable name of this strategy."""
        return "Continuous"

    @property
    def part_number(self) -> int:
        """Get the part number this strategy represents."""
        return 8

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check for the continuous queue strategy.

        Returns:
            Health status information including continuous batching statistics
        """
        base_health = super().health_check()

        # Add strategy-specific health information
        stats = self.get_stats()
        base_health.update(
            {
                "waiting_requests": stats.get("waiting_requests", 0),
                "active_requests": stats.get("active_requests", 0),
                "current_batch_size": stats.get("current_batch_size", 0),
                "current_generation_step": stats.get("current_generation_step", 0),
                "max_batch_size": stats.get("max_batch_size", 8),
                "total_generation_steps": stats.get("total_generation_steps", 0),
                "continuous_processor_running": True,  # Since we have a background loop
            }
        )

        return base_health
