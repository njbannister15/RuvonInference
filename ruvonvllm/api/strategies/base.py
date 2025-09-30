"""
Base strategy interface for queue processing strategies.

This module defines the Strategy pattern interface for different queue processing
approaches: sequential, batched (prefill), and continuous batching.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ruvonvllm.api.server import CompletionRequest, CompletionResponse


class QueueStrategy(ABC):
    """
    Abstract base class for queue processing strategies.

    This interface defines the contract that all queue strategies must implement,
    enabling the Strategy pattern for handling different types of request processing:
    - Sequential: One request at a time (Part 6)
    - Batched: Static batches processed together (Part 7)
    - Continuous: Dynamic batches with requests joining/leaving (Part 8)
    """

    @abstractmethod
    async def process_request(
        self, request: "CompletionRequest"
    ) -> "CompletionResponse":
        """
        Process a single completion request using this strategy.

        This is the main interface method that each strategy implements
        according to its specific processing approach.

        Args:
            request: The completion request to process

        Returns:
            CompletionResponse with generated text

        Raises:
            Exception: If processing fails
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics for this queue strategy.

        Returns:
            Dictionary containing strategy-specific metrics and status
        """
        pass

    @abstractmethod
    def get_recent_completions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recently completed requests for monitoring.

        Args:
            limit: Maximum number of recent completions to return

        Returns:
            List of recent completion data
        """
        pass

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """
        Get the human-readable name of this strategy.

        Returns:
            Strategy name (e.g., "Sequential", "Batched", "Continuous")
        """
        pass

    @property
    @abstractmethod
    def part_number(self) -> int:
        """
        Get the part number this strategy represents.

        Returns:
            Part number (6 for Sequential, 7 for Batched, 8 for Continuous)
        """
        pass

    def initialize(self) -> None:
        """
        Initialize the strategy and any background processors.

        This method is called once when the strategy is created to set up
        any necessary background threads or resources.
        """
        pass  # Default implementation does nothing

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check for this strategy.

        This method can be overridden by strategies that need custom health checks.

        Returns:
            Health status information
        """
        return {
            "strategy": self.strategy_name,
            "part": self.part_number,
            "status": "healthy",
        }
