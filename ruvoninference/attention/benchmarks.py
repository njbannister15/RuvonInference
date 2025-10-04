"""
Performance benchmarking utilities for different attention implementations.

This module provides tools to measure and compare the performance characteristics
of different attention implementations including memory usage, speed, and
scalability with sequence length.
"""

import logging
import time
from typing import List, Optional
from dataclasses import dataclass
import torch

from .implementations import (
    AttentionImplementation,
    load_model_with_attention,
    get_available_implementations,
)
from ..tokenizer.gpt2_tokenizer import GPT2TokenizerWrapper

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """
    Results from a single attention implementation benchmark.

    Attributes:
        implementation: Attention implementation that was tested
        sequence_length: Input sequence length
        generation_length: Number of tokens generated
        total_time: Total generation time in seconds
        tokens_per_second: Generation speed in tokens/second
        memory_allocated_mb: Peak GPU memory allocated in MB
        memory_reserved_mb: Peak GPU memory reserved in MB
        success: Whether the benchmark completed successfully
        error_message: Error message if benchmark failed
    """

    implementation: AttentionImplementation
    sequence_length: int
    generation_length: int
    total_time: float
    tokens_per_second: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    success: bool
    error_message: Optional[str] = None


class AttentionBenchmark:
    """
    Comprehensive benchmarking suite for attention implementations.

    This class provides methods to benchmark different attention implementations
    across various sequence lengths and generation parameters, measuring both
    performance and memory characteristics.
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        """
        Initialize the benchmark suite.

        Args:
            model_name: HuggingFace model identifier to benchmark
            device: Device to run benchmarks on ("cpu", "cuda", etc.)
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = GPT2TokenizerWrapper(model_name)

    def benchmark_implementation(
        self,
        implementation: AttentionImplementation,
        prompt: str,
        max_new_tokens: int = 50,
        warmup_runs: int = 1,
        benchmark_runs: int = 3,
    ) -> BenchmarkResult:
        """
        Benchmark a single attention implementation.

        Args:
            implementation: Attention implementation to benchmark
            prompt: Input text prompt
            max_new_tokens: Number of tokens to generate
            warmup_runs: Number of warmup runs (not measured)
            benchmark_runs: Number of measured benchmark runs

        Returns:
            BenchmarkResult with performance metrics
        """
        try:
            logger.info(f"Benchmarking {implementation.value} implementation")

            # Load model with specified implementation
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            model = load_model_with_attention(
                self.model_name, implementation, self.device, torch_dtype
            )

            # Tokenize input
            input_ids = self.tokenizer.encode(prompt, return_tensors=True)
            sequence_length = input_ids.shape[1]

            # Clear GPU memory stats if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Warmup runs
            for _ in range(warmup_runs):
                with torch.no_grad():
                    model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.tokenizer.eos_token_id,
                    )

            # Clear memory stats after warmup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Benchmark runs
            total_times = []
            for run in range(benchmark_runs):
                start_time = time.time()

                with torch.no_grad():
                    _ = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self.tokenizer.tokenizer.eos_token_id,
                    )

                end_time = time.time()
                total_times.append(end_time - start_time)

            # Calculate metrics
            avg_time = sum(total_times) / len(total_times)
            tokens_per_second = max_new_tokens / avg_time

            # Memory metrics
            memory_allocated_mb = 0
            memory_reserved_mb = 0
            if torch.cuda.is_available():
                memory_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
                memory_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2

            return BenchmarkResult(
                implementation=implementation,
                sequence_length=sequence_length,
                generation_length=max_new_tokens,
                total_time=avg_time,
                tokens_per_second=tokens_per_second,
                memory_allocated_mb=memory_allocated_mb,
                memory_reserved_mb=memory_reserved_mb,
                success=True,
            )

        except Exception as e:
            logger.error(f"Benchmark failed for {implementation.value}: {e}")
            return BenchmarkResult(
                implementation=implementation,
                sequence_length=len(prompt.split()) if prompt else 0,
                generation_length=max_new_tokens,
                total_time=0,
                tokens_per_second=0,
                memory_allocated_mb=0,
                memory_reserved_mb=0,
                success=False,
                error_message=str(e),
            )

    def compare_implementations(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        implementations: Optional[List[AttentionImplementation]] = None,
    ) -> List[BenchmarkResult]:
        """
        Compare multiple attention implementations on the same task.

        Args:
            prompt: Input text prompt
            max_new_tokens: Number of tokens to generate
            implementations: List of implementations to compare (None for all available)

        Returns:
            List of BenchmarkResult objects for each implementation
        """
        if implementations is None:
            implementations = get_available_implementations()

        results = []
        for implementation in implementations:
            result = self.benchmark_implementation(
                implementation, prompt, max_new_tokens
            )
            results.append(result)

        return results

    def sequence_length_scaling(
        self,
        implementation: AttentionImplementation,
        base_prompt: str,
        sequence_lengths: List[int],
        max_new_tokens: int = 20,
    ) -> List[BenchmarkResult]:
        """
        Benchmark how a implementation scales with sequence length.

        Args:
            implementation: Attention implementation to test
            base_prompt: Base prompt to extend
            sequence_lengths: List of sequence lengths to test
            max_new_tokens: Number of tokens to generate for each test

        Returns:
            List of BenchmarkResult objects for each sequence length
        """
        results = []

        for seq_len in sequence_lengths:
            # Create prompt of specified length
            words = base_prompt.split()
            while len(" ".join(words).split()) < seq_len:
                words.extend(base_prompt.split())

            # Trim to exact length
            prompt_words = words[:seq_len]
            prompt = " ".join(prompt_words)

            result = self.benchmark_implementation(
                implementation, prompt, max_new_tokens
            )
            results.append(result)

        return results

    def memory_scaling_analysis(
        self,
        prompt: str,
        max_new_tokens_list: List[int],
        implementation: Optional[AttentionImplementation] = None,
    ) -> List[BenchmarkResult]:
        """
        Analyze memory usage scaling with generation length.

        Args:
            prompt: Input text prompt
            max_new_tokens_list: List of generation lengths to test
            implementation: Backend to test (None for best available)

        Returns:
            List of BenchmarkResult objects for each generation length
        """
        if implementation is None:
            available = get_available_implementations()
            implementation = available[0]  # Use first available

        results = []
        for max_new_tokens in max_new_tokens_list:
            result = self.benchmark_implementation(
                implementation, prompt, max_new_tokens
            )
            results.append(result)

        return results


def format_benchmark_results(results: List[BenchmarkResult]) -> str:
    """
    Format benchmark results into a human-readable string.

    Args:
        results: List of benchmark results to format

    Returns:
        Formatted string with benchmark comparison
    """
    if not results:
        return "No benchmark results to display"

    # Filter successful results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    output = []

    if successful:
        output.append("📊 Benchmark Results\n")
        output.append(
            "Backend".ljust(20)
            + "Time (s)".ljust(12)
            + "Tokens/s".ljust(12)
            + "Memory (MB)"
        )
        output.append("-" * 60)

        for result in successful:
            implementation_name = result.implementation.value
            time_str = f"{result.total_time:.3f}"
            tokens_str = f"{result.tokens_per_second:.1f}"
            memory_str = f"{result.memory_allocated_mb:.1f}"

            output.append(
                implementation_name.ljust(20)
                + time_str.ljust(12)
                + tokens_str.ljust(12)
                + memory_str
            )

        # Find best performer
        if len(successful) > 1:
            fastest = min(successful, key=lambda r: r.total_time)
            output.append(
                f"\n🏆 Fastest: {fastest.implementation.value} ({fastest.total_time:.3f}s)"
            )

            if any(r.memory_allocated_mb > 0 for r in successful):
                most_efficient = min(successful, key=lambda r: r.memory_allocated_mb)
                output.append(
                    f"💾 Most Memory Efficient: {most_efficient.implementation.value} ({most_efficient.memory_allocated_mb:.1f} MB)"
                )

    if failed:
        output.append("\n❌ Failed Benchmarks:")
        for result in failed:
            output.append(f"  {result.implementation.value}: {result.error_message}")

    return "\n".join(output)
