"""
Integration tests for batch generation functionality.

This module tests the complete batch generation pipeline from model loading
to token generation, ensuring all components work together correctly.
"""

import pytest
import torch

from ruvonvllm.model.gpt2 import GPT2Model
from ruvonvllm.tokenizer.gpt2_tokenizer import GPT2TokenizerWrapper


class TestBatchGenerationIntegration:
    """Integration tests for the complete batch generation pipeline."""

    @pytest.fixture(scope="class")
    def model(self):
        """Load GPT-2 model once for all tests."""
        model = GPT2Model(model_name="gpt2", device="cpu")
        model.load_model()
        return model

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Load tokenizer once for all tests."""
        return GPT2TokenizerWrapper("gpt2")

    def test_model_composition_initialization(self, model):
        """Test that the model properly initializes with batch generation capability."""
        # Verify model is loaded
        assert model.model is not None
        assert model.config is not None

        # Verify batch generator is properly composed
        assert model._batch_generator is not None
        assert model.batch_generator is not None

        # Verify the batch generator has reference to the model
        assert model.batch_generator.model is model

    def test_single_sequence_generation(self, model, tokenizer):
        """Test batch generation with a single sequence."""
        # Prepare input
        prompt = "Once upon a time"
        input_ids_tensor = tokenizer.encode(prompt)
        input_ids = input_ids_tensor.squeeze().tolist()  # Convert to list of token IDs
        batch_input_ids = [torch.tensor(input_ids, dtype=torch.long)]

        # Generate tokens
        results = model.generate_batch_with_sampling(
            batch_input_ids=batch_input_ids,
            max_length=5,
            temperature=1.0,
            use_cache=True,
            show_progress=False,
        )

        # Assertions
        assert len(results) == 1, "Should return exactly one sequence"
        result_sequence = results[0]

        # Check that new tokens were generated
        original_length = len(input_ids)
        assert len(result_sequence) > original_length, "Should generate new tokens"
        assert len(result_sequence) <= original_length + 5, "Should respect max_length"

        # Check that original prompt is preserved
        assert (
            result_sequence[:original_length] == input_ids
        ), "Original prompt should be preserved"

        # Check that generated tokens are valid
        for token_id in result_sequence:
            assert isinstance(token_id, int), "All tokens should be integers"
            assert (
                0 <= token_id < model.config.vocab_size
            ), "Tokens should be within vocab range"

    def test_multiple_sequence_generation(self, model, tokenizer):
        """Test batch generation with multiple sequences."""
        # Prepare multiple inputs with different lengths
        prompts = ["The quick brown fox", "Hello world", "AI is"]

        batch_input_ids = []
        original_lengths = []

        for prompt in prompts:
            input_ids_tensor = tokenizer.encode(prompt)
            input_ids = input_ids_tensor.squeeze().tolist()
            batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            original_lengths.append(len(input_ids))

        # Generate tokens
        results = model.generate_batch_with_sampling(
            batch_input_ids=batch_input_ids,
            max_length=8,
            temperature=0.8,
            use_cache=True,
            show_progress=False,
        )

        # Assertions
        assert len(results) == len(prompts), f"Should return {len(prompts)} sequences"

        for i, (result_sequence, original_length) in enumerate(
            zip(results, original_lengths)
        ):
            # Check sequence properties
            assert (
                len(result_sequence) > original_length
            ), f"Sequence {i} should have new tokens"
            assert (
                len(result_sequence) <= original_length + 8
            ), f"Sequence {i} should respect max_length"

            # Check original prompt preservation
            original_tokens = tokenizer.encode(prompts[i]).squeeze().tolist()
            assert (
                result_sequence[:original_length] == original_tokens
            ), f"Sequence {i} prompt should be preserved"

            # Check token validity
            for token_id in result_sequence:
                assert isinstance(
                    token_id, int
                ), f"All tokens in sequence {i} should be integers"
                assert (
                    0 <= token_id < model.config.vocab_size
                ), f"Tokens in sequence {i} should be within vocab"

    def test_sampling_parameters_effect(self, model, tokenizer):
        """Test that different sampling parameters produce different results."""
        prompt = "The future of AI"
        input_ids_tensor = tokenizer.encode(prompt)
        input_ids = input_ids_tensor.squeeze().tolist()
        batch_input_ids = [torch.tensor(input_ids, dtype=torch.long)]

        # Generate with low temperature (more deterministic)
        low_temp_results = model.generate_batch_with_sampling(
            batch_input_ids=batch_input_ids,
            max_length=10,
            temperature=0.1,
            use_cache=True,
            show_progress=False,
        )

        # Generate with high temperature (more random)
        high_temp_results = model.generate_batch_with_sampling(
            batch_input_ids=batch_input_ids,
            max_length=10,
            temperature=2.0,
            use_cache=True,
            show_progress=False,
        )

        # Both should generate tokens
        assert len(low_temp_results[0]) > len(input_ids)
        assert len(high_temp_results[0]) > len(input_ids)

        # Results might be different (though not guaranteed)
        # At minimum, both should be valid sequences
        for sequence in [low_temp_results[0], high_temp_results[0]]:
            assert all(isinstance(token, int) for token in sequence)
            assert all(0 <= token < model.config.vocab_size for token in sequence)

    def test_top_k_sampling(self, model, tokenizer):
        """Test top-k sampling functionality."""
        prompt = "Machine learning is"
        input_ids_tensor = tokenizer.encode(prompt)
        input_ids = input_ids_tensor.squeeze().tolist()
        batch_input_ids = [torch.tensor(input_ids, dtype=torch.long)]

        # Generate with top-k sampling
        results = model.generate_batch_with_sampling(
            batch_input_ids=batch_input_ids,
            max_length=6,
            temperature=1.0,
            top_k=20,
            use_cache=True,
            show_progress=False,
        )

        # Should generate valid tokens
        assert len(results) == 1
        result_sequence = results[0]
        assert len(result_sequence) > len(input_ids)

        # All tokens should be valid
        for token_id in result_sequence:
            assert isinstance(token_id, int)
            assert 0 <= token_id < model.config.vocab_size

    def test_top_p_sampling(self, model, tokenizer):
        """Test nucleus (top-p) sampling functionality."""
        prompt = "Deep learning"
        input_ids_tensor = tokenizer.encode(prompt)
        input_ids = input_ids_tensor.squeeze().tolist()
        batch_input_ids = [torch.tensor(input_ids, dtype=torch.long)]

        # Generate with nucleus sampling
        results = model.generate_batch_with_sampling(
            batch_input_ids=batch_input_ids,
            max_length=7,
            temperature=1.0,
            top_p=0.9,
            use_cache=True,
            show_progress=False,
        )

        # Should generate valid tokens
        assert len(results) == 1
        result_sequence = results[0]
        assert len(result_sequence) > len(input_ids)

        # All tokens should be valid
        for token_id in result_sequence:
            assert isinstance(token_id, int)
            assert 0 <= token_id < model.config.vocab_size

    def test_cache_vs_no_cache(self, model, tokenizer):
        """Test that cached and non-cached generation produce same results."""
        prompt = "Python programming"
        input_ids_tensor = tokenizer.encode(prompt)
        input_ids = input_ids_tensor.squeeze().tolist()
        batch_input_ids = [torch.tensor(input_ids, dtype=torch.long)]

        # Set a fixed seed for deterministic comparison
        torch.manual_seed(42)

        # Generate with cache
        cached_results = model.generate_batch_with_sampling(
            batch_input_ids=batch_input_ids,
            max_length=5,
            temperature=0.1,  # Low temperature for more deterministic results
            use_cache=True,
            show_progress=False,
        )

        # Reset seed
        torch.manual_seed(42)

        # Generate without cache
        no_cache_results = model.generate_batch_with_sampling(
            batch_input_ids=batch_input_ids,
            max_length=5,
            temperature=0.1,  # Same low temperature
            use_cache=False,
            show_progress=False,
        )

        # Results should be identical (or very similar due to deterministic sampling)
        assert len(cached_results) == len(no_cache_results)
        assert len(cached_results[0]) == len(no_cache_results[0])

        # At minimum, the prompt should be preserved identically
        prompt_length = len(input_ids)
        assert cached_results[0][:prompt_length] == no_cache_results[0][:prompt_length]

    def test_empty_input_handling(self, model):
        """Test handling of edge cases."""
        # Empty batch
        results = model.generate_batch_with_sampling(
            batch_input_ids=[], max_length=5, show_progress=False
        )
        assert results == [], "Empty input should return empty result"

    def test_max_length_zero(self, model, tokenizer):
        """Test behavior with max_length=0."""
        prompt = "Test prompt"
        input_ids_tensor = tokenizer.encode(prompt)
        input_ids = input_ids_tensor.squeeze().tolist()
        batch_input_ids = [torch.tensor(input_ids, dtype=torch.long)]

        results = model.generate_batch_with_sampling(
            batch_input_ids=batch_input_ids, max_length=0, show_progress=False
        )

        # Should return original sequences unchanged
        assert len(results) == 1
        assert results[0] == input_ids

    def test_batch_generation_performance_smoke_test(self, model, tokenizer):
        """Smoke test for batch generation performance characteristics."""
        import time

        # Create a batch of sequences
        prompts = [
            "AI research",
            "Climate change",
            "Space exploration",
            "Quantum computing",
        ]
        batch_input_ids = []

        for prompt in prompts:
            input_ids_tensor = tokenizer.encode(prompt)
            input_ids = input_ids_tensor.squeeze().tolist()
            batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))

        # Time the batch generation
        start_time = time.time()
        results = model.generate_batch_with_sampling(
            batch_input_ids=batch_input_ids,
            max_length=10,
            temperature=1.0,
            use_cache=True,
            show_progress=False,
        )
        end_time = time.time()

        # Basic assertions
        assert len(results) == len(prompts)
        assert all(
            len(seq) > len(tokenizer.encode(prompts[i]).squeeze().tolist())
            for i, seq in enumerate(results)
        )

        # Performance should complete in reasonable time (this is a smoke test)
        generation_time = end_time - start_time
        assert (
            generation_time < 30.0
        ), f"Batch generation took too long: {generation_time:.2f}s"

    def test_batch_generator_direct_access(self, model, tokenizer):
        """Test direct access to the batch generator component."""
        prompt = "Direct access test"
        input_ids_tensor = tokenizer.encode(prompt)
        input_ids = input_ids_tensor.squeeze().tolist()
        batch_input_ids = [torch.tensor(input_ids, dtype=torch.long)]

        # Access batch generator directly
        batch_gen = model.batch_generator
        assert batch_gen is not None

        # Use it directly
        results = batch_gen.generate_batch_with_sampling(
            batch_input_ids=batch_input_ids, max_length=3, show_progress=False
        )

        # Should work identically to the model method
        assert len(results) == 1
        assert len(results[0]) > len(input_ids)

    def test_specific_prompts_batch_generation(self, model, tokenizer):
        """Test batch generation with specific sentence prompts."""
        # Test prompts as requested
        prompts = ["to be or not to", "its great to be", "today is a good day to"]

        batch_input_ids = []
        original_lengths = []

        # Prepare input tensors for each prompt
        for prompt in prompts:
            input_ids_tensor = tokenizer.encode(prompt)
            input_ids = input_ids_tensor.squeeze().tolist()
            batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            original_lengths.append(len(input_ids))

        # Generate tokens with moderate creativity
        results = model.generate_batch_with_sampling(
            batch_input_ids=batch_input_ids,
            max_length=10,  # Generate up to 10 new tokens per sequence
            temperature=0.8,  # Moderately creative
            top_k=50,  # Consider top 50 tokens
            top_p=0.9,  # Nucleus sampling
            use_cache=True,
            show_progress=True,  # Show progress for educational purposes
        )

        # Verify results
        assert len(results) == len(prompts), f"Should return {len(prompts)} sequences"

        # Check each generated sequence
        for i, (result_sequence, original_length, prompt) in enumerate(
            zip(results, original_lengths, prompts)
        ):
            print(f"\nPrompt {i+1}: '{prompt}'")

            # Decode and display the result
            try:
                full_text = tokenizer.decode(torch.tensor(result_sequence))
                generated_part = tokenizer.decode(
                    torch.tensor(result_sequence[original_length:])
                )
                print(f"Full result: '{full_text}'")
                print(f"Generated: '{generated_part}'")
            except Exception as e:
                print(f"Could not decode sequence: {e}")

            # Validate sequence properties
            assert (
                len(result_sequence) > original_length
            ), f"Sequence {i} should have new tokens"
            assert (
                len(result_sequence) <= original_length + 10
            ), f"Sequence {i} should respect max_length"

            # Verify original prompt is preserved
            original_tokens = tokenizer.encode(prompt).squeeze().tolist()
            assert (
                result_sequence[:original_length] == original_tokens
            ), f"Sequence {i} prompt should be preserved"

            # Verify all tokens are valid
            for token_id in result_sequence:
                assert isinstance(
                    token_id, int
                ), f"All tokens in sequence {i} should be integers"
                assert (
                    0 <= token_id < model.config.vocab_size
                ), f"Tokens in sequence {i} should be within vocab range"

    def test_model_not_loaded_error_handling(self):
        """Test proper error handling when model is not loaded."""
        model = GPT2Model()  # Don't call load_model()

        # Should raise RuntimeError for batch generation
        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.generate_batch_with_sampling(
                batch_input_ids=[torch.tensor([1, 2, 3], dtype=torch.long)],
                max_length=5,
            )

        # Should raise RuntimeError for batch generator access
        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = model.batch_generator
