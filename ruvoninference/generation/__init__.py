"""
Generation capabilities for RuvonInference.

This module contains various generation strategies and capabilities that can be
composed with language models for flexible and modular text generation.
"""

from .batch_generator import BatchGenerator

__all__ = ["BatchGenerator"]
