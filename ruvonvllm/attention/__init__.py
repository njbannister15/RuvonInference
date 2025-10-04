"""
Attention implementation management for RuvonVLLM.

This module provides support for different attention implementations including
standard PyTorch attention, FlashAttention, and SDPA (Scaled Dot-Product Attention).
"""

from .implementations import (
    AttentionImplementation,
    load_model_with_attention,
    get_available_implementations,
    get_implementation_info,
    recommend_implementation,
)

__all__ = [
    "AttentionImplementation",
    "load_model_with_attention",
    "get_available_implementations",
    "get_implementation_info",
    "recommend_implementation",
]
