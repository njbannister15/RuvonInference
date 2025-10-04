"""
Attention implementation management for different attention implementations.

This module provides a unified interface for loading models with different
attention implementations including standard attention, FlashAttention, and SDPA.
"""

import logging
from enum import Enum
from typing import Optional, List, Dict, Any
import torch
from transformers import GPT2LMHeadModel

logger = logging.getLogger(__name__)


class AttentionImplementation(Enum):
    """
    Enumeration of supported attention implementations.

    Each implementation represents a different attention algorithm with
    varying memory and computational characteristics:

    - EAGER: Standard PyTorch attention (default, most compatible)
    - FLASH_ATTENTION_2: Memory-efficient FlashAttention implementation
    - SDPA: PyTorch's built-in Scaled Dot-Product Attention
    """

    EAGER = "eager"
    FLASH_ATTENTION_2 = "flash_attention_2"
    SDPA = "sdpa"


def get_available_implementations() -> List[AttentionImplementation]:
    """
    Get list of attention implementations available in the current environment.

    Checks for dependencies and hardware requirements for each implementation.

    Returns:
        List of AttentionImplementation enums that are available
    """
    available = [AttentionImplementation.EAGER]  # Always available

    # Check for SDPA support (PyTorch >= 2.0)
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        available.append(AttentionImplementation.SDPA)
        logger.debug("SDPA implementation available")

    # Check for FlashAttention support
    try:
        import flash_attn

        available.append(AttentionImplementation.FLASH_ATTENTION_2)
        logger.debug(
            f"FlashAttention implementation available (version: {flash_attn.__version__})"
        )
    except ImportError:
        logger.debug("FlashAttention not installed - implementation unavailable")

    return available


def load_model_with_attention(
    model_name: str,
    implementation: AttentionImplementation,
    device: str = "cpu",
    torch_dtype: Optional[torch.dtype] = None,
) -> GPT2LMHeadModel:
    """
    Load a GPT-2 model with the specified attention implementation.

    Args:
        model_name: HuggingFace model identifier (e.g., "gpt2", "gpt2-medium")
        implementation: Attention implementation to use
        device: Device to load model on ("cpu", "cuda", etc.)
        torch_dtype: PyTorch data type (FlashAttention requires fp16/bf16)

    Returns:
        Loaded GPT2LMHeadModel with specified attention implementation

    Raises:
        RuntimeError: If implementation is not available or incompatible
        ValueError: If invalid configuration is provided
    """
    # Validate implementation availability
    available_implementations = get_available_implementations()
    if implementation not in available_implementations:
        raise RuntimeError(
            f"Backend {implementation.value} not available. "
            f"Available implementations: {[b.value for b in available_implementations]}"
        )

    # Set appropriate torch_dtype for FlashAttention
    if implementation == AttentionImplementation.FLASH_ATTENTION_2:
        if torch_dtype is None:
            torch_dtype = torch.float16
            logger.info(
                "Using float16 for FlashAttention (required for optimal performance)"
            )
        elif torch_dtype not in [torch.float16, torch.bfloat16]:
            logger.warning(
                f"FlashAttention works best with fp16/bf16, got {torch_dtype}. "
                "Consider using torch.float16 or torch.bfloat16"
            )

    # Load model with attention implementation
    try:
        logger.info(
            f"Loading {model_name} with {implementation.value} attention implementation"
        )

        model_kwargs = {
            "attn_implementation": implementation.value,
        }

        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        model = GPT2LMHeadModel.from_pretrained(model_name, **model_kwargs)
        model = model.to(device)

        logger.info(
            f"Successfully loaded model with {implementation.value} implementation"
        )
        return model

    except Exception as e:
        logger.error(
            f"Failed to load model with {implementation.value} implementation: {e}"
        )

        # Fallback to eager implementation if requested implementation fails
        if implementation != AttentionImplementation.EAGER:
            logger.warning(
                f"Falling back to {AttentionImplementation.EAGER.value} implementation"
            )
            return load_model_with_attention(
                model_name, AttentionImplementation.EAGER, device, torch_dtype
            )
        else:
            raise RuntimeError(f"Failed to load model: {e}")


def get_implementation_info(implementation: AttentionImplementation) -> Dict[str, Any]:
    """
    Get detailed information about an attention implementation.

    Args:
        implementation: Attention implementation to get info for

    Returns:
        Dictionary with implementation information including:
        - name: Backend name
        - description: Human-readable description
        - memory_efficiency: Relative memory efficiency
        - speed_profile: Speed characteristics
        - requirements: Special requirements or dependencies
    """
    info = {
        AttentionImplementation.EAGER: {
            "name": "Standard PyTorch Attention",
            "description": "Default PyTorch attention implementation",
            "memory_efficiency": "O(n²) - Standard quadratic memory usage",
            "speed_profile": "Good for short sequences, memory-bound for long sequences",
            "requirements": "None - always available",
            "best_for": "Compatibility, debugging, short sequences",
        },
        AttentionImplementation.SDPA: {
            "name": "Scaled Dot-Product Attention",
            "description": "PyTorch's optimized attention implementation",
            "memory_efficiency": "O(n²) - Optimized but still quadratic",
            "speed_profile": "Faster than eager, good memory access patterns",
            "requirements": "PyTorch >= 2.0",
            "best_for": "General purpose, good balance of speed and compatibility",
        },
        AttentionImplementation.FLASH_ATTENTION_2: {
            "name": "FlashAttention 2",
            "description": "Memory-efficient attention with block-wise computation",
            "memory_efficiency": "O(n) - Linear memory usage through tiling",
            "speed_profile": "Excellent for long sequences, GPU memory hierarchy optimized",
            "requirements": "flash-attn package, fp16/bf16 dtype, CUDA GPU",
            "best_for": "Long sequences, memory-constrained environments, production serving",
        },
    }

    return info.get(
        implementation, {"name": "Unknown", "description": "No information available"}
    )


def recommend_implementation(
    sequence_length: int, device: str, memory_gb: Optional[float] = None
) -> AttentionImplementation:
    """
    Recommend optimal attention implementation based on use case.

    Args:
        sequence_length: Expected maximum sequence length
        device: Target device ("cpu", "cuda", etc.)
        memory_gb: Available memory in GB (optional)

    Returns:
        Recommended AttentionImplementation
    """
    available = get_available_implementations()

    # For CPU, always use eager (FlashAttention requires CUDA)
    if device == "cpu":
        return AttentionImplementation.EAGER

    # For long sequences, prefer FlashAttention if available
    if (
        sequence_length > 1024
        and AttentionImplementation.FLASH_ATTENTION_2 in available
    ):
        return AttentionImplementation.FLASH_ATTENTION_2

    # For medium sequences, prefer SDPA if available
    if sequence_length > 256 and AttentionImplementation.SDPA in available:
        return AttentionImplementation.SDPA

    # Default to eager for short sequences or limited implementations
    return AttentionImplementation.EAGER
