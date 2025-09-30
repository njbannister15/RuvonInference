"""
Factory for creating queue strategy instances.

This module implements the Factory pattern to create appropriate queue strategy
instances based on configuration, making strategy selection clean and testable.
"""

import os
from typing import Dict, Type, Optional

from .base import QueueStrategy
from .sequential import SequentialQueueStrategy
from .batched import BatchedQueueStrategy
from .continuous import ContinuousQueueStrategy


class QueueStrategyFactory:
    """
    Factory class for creating queue strategy instances.

    This factory encapsulates the logic for creating the appropriate queue strategy
    based on configuration parameters. It supports:
    - Environment variable configuration
    - Direct mode specification
    - Default fallback behavior
    - Strategy validation

    The factory implements the Factory pattern, providing a clean interface
    for strategy creation while hiding the complexity of strategy selection.
    """

    # Registry of available strategies
    _strategies: Dict[str, Type[QueueStrategy]] = {
        "sequential": SequentialQueueStrategy,
        "batched": BatchedQueueStrategy,
        "continuous": ContinuousQueueStrategy,
    }

    @classmethod
    def create_strategy(cls, mode: Optional[str] = None) -> QueueStrategy:
        """
        Create a queue strategy instance based on the specified mode.

        Args:
            mode: The queue mode to use. If None, will read from QUEUE_MODE
                  environment variable. Valid values: "sequential", "batched", "continuous"

        Returns:
            QueueStrategy instance for the specified mode

        Raises:
            ValueError: If the specified mode is not supported
            RuntimeError: If strategy creation fails
        """
        # Determine mode from parameter or environment
        if mode is None:
            mode = cls._get_mode_from_environment()

        # Validate mode
        if mode not in cls._strategies:
            valid_modes = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Invalid queue mode: '{mode}'. " f"Valid modes are: {valid_modes}"
            )

        # Create strategy instance
        try:
            strategy_class = cls._strategies[mode]
            strategy = strategy_class()

            # Initialize the strategy (sets up background processors, etc.)
            strategy.initialize()

            return strategy
        except Exception as e:
            raise RuntimeError(f"Failed to create {mode} queue strategy: {e}") from e

    @classmethod
    def get_available_modes(cls) -> list[str]:
        """
        Get list of available queue modes.

        Returns:
            List of supported queue mode names
        """
        return list(cls._strategies.keys())

    @classmethod
    def register_strategy(cls, mode: str, strategy_class: Type[QueueStrategy]) -> None:
        """
        Register a new queue strategy.

        This allows for extending the factory with custom strategies.

        Args:
            mode: The mode name for this strategy
            strategy_class: The strategy class to register

        Raises:
            ValueError: If mode already exists or strategy_class is invalid
        """
        if mode in cls._strategies:
            raise ValueError(f"Strategy mode '{mode}' already registered")

        if not issubclass(strategy_class, QueueStrategy):
            raise ValueError(
                f"Strategy class must inherit from QueueStrategy, "
                f"got {strategy_class.__name__}"
            )

        cls._strategies[mode] = strategy_class

    @classmethod
    def _get_mode_from_environment(cls) -> str:
        """
        Get queue mode from environment variables.

        This method handles the transition from the old USE_BATCHED_QUEUE
        environment variable to the new QUEUE_MODE variable.

        Returns:
            Queue mode string

        Priority:
        1. QUEUE_MODE environment variable
        2. USE_BATCHED_QUEUE for backwards compatibility
        3. Default to "batched"
        """
        # Check new QUEUE_MODE variable first
        queue_mode = os.getenv("QUEUE_MODE")
        if queue_mode:
            return queue_mode.lower()

        # Check legacy USE_BATCHED_QUEUE for backwards compatibility
        use_batched = os.getenv("USE_BATCHED_QUEUE", "True").lower()
        if use_batched == "true":
            return "batched"
        else:
            return "sequential"

    @classmethod
    def create_from_environment(cls) -> QueueStrategy:
        """
        Create a strategy instance based on environment configuration.

        This is a convenience method that reads configuration from environment
        variables and creates the appropriate strategy.

        Returns:
            QueueStrategy instance configured from environment
        """
        return cls.create_strategy()

    @classmethod
    def get_strategy_info(cls, mode: str) -> Dict[str, any]:
        """
        Get information about a specific strategy mode.

        Args:
            mode: The strategy mode to get information about

        Returns:
            Dictionary containing strategy information

        Raises:
            ValueError: If mode is not supported
        """
        if mode not in cls._strategies:
            raise ValueError(f"Unknown strategy mode: {mode}")

        strategy_class = cls._strategies[mode]

        # Create a temporary instance to get strategy properties
        try:
            temp_strategy = strategy_class()
            return {
                "mode": mode,
                "name": temp_strategy.strategy_name,
                "part": temp_strategy.part_number,
                "class": strategy_class.__name__,
                "module": strategy_class.__module__,
            }
        except Exception as e:
            return {
                "mode": mode,
                "name": "Unknown",
                "part": 0,
                "class": strategy_class.__name__,
                "module": strategy_class.__module__,
                "error": str(e),
            }
