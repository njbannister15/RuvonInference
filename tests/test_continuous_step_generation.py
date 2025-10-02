"""
Integration tests for continuous step generation functionality.

This module tests the generate_continuous_step method which is the core of
continuous batching - generating one token at a time for a dynamic batch
of requests that can change composition between steps.
"""

import pytest
import torch
import time

from ruvonvllm.model.gpt2 import GPT2Model
from ruvonvllm.tokenizer.gpt2_tokenizer import GPT2TokenizerWrapper
from ruvonvllm.api.continuous_queue import ContinuousRequest, RequestState


class TestContinuousStepGeneration:
    """Integration tests for the continuous step generation pipeline."""

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

    def create_continuous_request(
        self,
        request_id: str,
        prompt: str,
        tokenizer,
        max_tokens: int = 10,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> ContinuousRequest:
        """
        Helper method to create a ContinuousRequest from a prompt.

        Args:
            request_id: Unique identifier for the request
            prompt: Text prompt to tokenize
            tokenizer: Tokenizer to use for encoding
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter

        Returns:
            ContinuousRequest object ready for generation
        """
        input_ids_tensor = tokenizer.encode(prompt)
        input_tokens = input_ids_tensor.squeeze().tolist()

        return ContinuousRequest(
            id=request_id,
            request_data={"prompt": prompt},
            state=RequestState.ACTIVE,
            created_at=time.time(),
            input_tokens=input_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=True,
        )

    def test_single_request_prefill_step(self, model, tokenizer):
        """Test continuous step generation with a single request (prefill phase)."""
        # Create a single continuous request
        request = self.create_continuous_request(
            request_id="test_001",
            prompt="Once upon a time",
            tokenizer=tokenizer,
            max_tokens=5,
            temperature=0.8,
        )

        active_requests = [request]

        # First step should be prefill (past_key_values=None)
        next_tokens, past_key_values, finished_flags = model.generate_continuous_step(
            active_requests=active_requests,
            past_key_values=None,
            show_progress=True,
        )

        # Assertions for prefill step
        assert len(next_tokens) == 1, "Should return one token for single request"
        assert past_key_values is not None, "Should return cached key-values"
        assert len(finished_flags) == 1, "Should return one finished flag"
        assert isinstance(next_tokens[0], int), "Token should be integer"
        assert 0 <= next_tokens[0] < model.config.vocab_size, "Token should be valid"

        # Check that request was updated
        assert (
            len(request.generated_tokens) == 1
        ), "Request should have one generated token"
        assert (
            request.generated_tokens[0] == next_tokens[0]
        ), "Request should store the generated token"
        assert request.generation_step == 1, "Request should track generation step"

    def test_single_request_decode_steps(self, model, tokenizer):
        """Test continuous step generation with decode steps (using KV cache)."""
        # Create a single continuous request
        request = self.create_continuous_request(
            request_id="test_002",
            prompt="The quick brown fox",
            tokenizer=tokenizer,
            max_tokens=3,
            temperature=0.5,
        )

        active_requests = [request]
        past_key_values = None

        # Generate multiple steps
        generated_sequence = []
        for step in range(3):
            next_tokens, past_key_values, finished_flags = (
                model.generate_continuous_step(
                    active_requests=active_requests,
                    past_key_values=past_key_values,
                    show_progress=True,
                )
            )

            # Basic assertions for each step
            assert len(next_tokens) == 1, f"Step {step}: Should return one token"
            assert past_key_values is not None, f"Step {step}: Should maintain KV cache"
            assert len(finished_flags) == 1, f"Step {step}: Should return one flag"

            generated_sequence.append(next_tokens[0])

            # Check if finished
            if finished_flags[0]:
                print(f"Request finished at step {step + 1}")
                break

        # Final assertions
        assert len(request.generated_tokens) == len(
            generated_sequence
        ), "Request should track all tokens"
        assert (
            request.generated_tokens == generated_sequence
        ), "Request should store correct sequence"

        # Decode and display result
        try:
            full_sequence = request.input_tokens + request.generated_tokens
            full_text = tokenizer.decode(torch.tensor(full_sequence))
            generated_text = tokenizer.decode(torch.tensor(request.generated_tokens))
            print(f"Full result: '{full_text}'")
            print(f"Generated part: '{generated_text}'")
        except Exception as e:
            print(f"Could not decode sequence: {e}")

    def test_multiple_requests_continuous_step(self, model, tokenizer):
        """Test continuous step generation with multiple requests in the same batch."""
        # Create multiple continuous requests with different prompts
        prompts = ["to be or not to", "its great to be", "today is a good day to"]
        requests = []

        for i, prompt in enumerate(prompts):
            request = self.create_continuous_request(
                request_id=f"multi_test_{i:03d}",
                prompt=prompt,
                tokenizer=tokenizer,
                max_tokens=5,
                temperature=0.7,
                top_k=40,
            )
            requests.append(request)

        active_requests = requests.copy()
        past_key_values = None
        step_count = 0

        # Generate tokens until all requests finish or max steps reached
        while active_requests and step_count < 10:
            next_tokens, past_key_values, finished_flags = (
                model.generate_continuous_step(
                    active_requests=active_requests,
                    past_key_values=past_key_values,
                    show_progress=True,
                )
            )

            step_count += 1
            print(f"\nStep {step_count}:")
            print(f"  Active requests: {len(active_requests)}")
            print(f"  Generated tokens: {next_tokens}")
            print(f"  Finished flags: {finished_flags}")

            # Assertions for this step
            assert len(next_tokens) == len(
                active_requests
            ), "Should return token for each active request"
            assert len(finished_flags) == len(
                active_requests
            ), "Should return flag for each active request"
            assert past_key_values is not None, "Should maintain KV cache"

            # Remove finished requests from active list (simulate continuous batching)
            active_requests = [
                req
                for req, finished in zip(active_requests, finished_flags)
                if not finished
            ]

        # Final verification - all original requests should have generated tokens
        for i, request in enumerate(requests):
            print(f"\nRequest {i + 1} ('{prompts[i]}'):")
            print(f"  Generated {len(request.generated_tokens)} tokens")
            print(f"  Generation steps: {request.generation_step}")

            assert (
                len(request.generated_tokens) > 0
            ), f"Request {i} should have generated tokens"
            assert request.generation_step > 0, f"Request {i} should have tracked steps"

            # Decode and display result
            try:
                full_sequence = request.input_tokens + request.generated_tokens
                full_text = tokenizer.decode(torch.tensor(full_sequence))
                generated_text = tokenizer.decode(
                    torch.tensor(request.generated_tokens)
                )
                print(f"  Full result: '{full_text}'")
                print(f"  Generated: '{generated_text}'")
            except Exception as e:
                print(f"  Could not decode: {e}")

    def test_empty_active_requests(self, model):
        """Test that generate_continuous_step handles empty request list gracefully."""
        # Test with empty request list
        next_tokens, past_key_values, finished_flags = model.generate_continuous_step(
            active_requests=[],
            past_key_values=None,
            show_progress=False,
        )

        # Should return empty results
        assert next_tokens == [], "Should return empty token list"
        assert past_key_values is None, "Should return None for KV cache"
        assert finished_flags == [], "Should return empty flags list"

    def test_dynamic_batch_composition(self, model, tokenizer):
        """Test that requests can be added/removed dynamically during generation."""
        # Start with one request
        request1 = self.create_continuous_request(
            request_id="dynamic_001",
            prompt="The future of AI",
            tokenizer=tokenizer,
            max_tokens=3,
            temperature=0.6,
        )

        active_requests = [request1]
        past_key_values = None

        # First step with one request
        next_tokens, past_key_values, finished_flags = model.generate_continuous_step(
            active_requests=active_requests,
            past_key_values=past_key_values,
            show_progress=True,
        )

        print(f"Step 1 - Single request: {next_tokens}")

        # Add a second request (simulating new request joining mid-generation)
        # Note: In real continuous batching, this would require more sophisticated
        # KV cache management, but we test the basic functionality here
        request2 = self.create_continuous_request(
            request_id="dynamic_002",
            prompt="Machine learning is",
            tokenizer=tokenizer,
            max_tokens=3,
            temperature=0.6,
        )

        # For demonstration, we restart KV cache when batch composition changes
        # In production, this would be handled more elegantly
        active_requests = [request1, request2]
        past_key_values = None  # Reset cache for new batch composition

        # Continue generation with both requests
        next_tokens, past_key_values, finished_flags = model.generate_continuous_step(
            active_requests=active_requests,
            past_key_values=past_key_values,
            show_progress=True,
        )

        print(f"Step 2 - Two requests: {next_tokens}")

        # Assertions
        assert len(next_tokens) == 2, "Should handle two requests"
        assert len(finished_flags) == 2, "Should return flags for both"

        # Both requests should have been updated
        assert len(request1.generated_tokens) >= 1, "Request 1 should have tokens"
        assert len(request2.generated_tokens) >= 1, "Request 2 should have tokens"

    def test_sampling_parameters_respected(self, model, tokenizer):
        """Test that individual request sampling parameters are respected."""
        # Create requests with different sampling parameters
        request_deterministic = self.create_continuous_request(
            request_id="deterministic_001",
            prompt="The capital of France is",
            tokenizer=tokenizer,
            max_tokens=3,
            temperature=0.1,  # Very low temperature for deterministic behavior
            top_k=1,  # Only consider top token
        )

        request_creative = self.create_continuous_request(
            request_id="creative_001",
            prompt="The capital of France is",
            tokenizer=tokenizer,
            max_tokens=3,
            temperature=1.5,  # High temperature for more randomness
            top_k=100,  # Consider many tokens
            top_p=0.95,
        )

        active_requests = [request_deterministic, request_creative]

        # Generate one step
        next_tokens, past_key_values, finished_flags = model.generate_continuous_step(
            active_requests=active_requests,
            past_key_values=None,
            show_progress=True,
        )

        print(f"Deterministic (temp=0.1): token {next_tokens[0]}")
        print(f"Creative (temp=1.5): token {next_tokens[1]}")

        # Basic assertions
        assert len(next_tokens) == 2, "Should generate for both requests"
        assert all(
            isinstance(token, int) for token in next_tokens
        ), "All tokens should be integers"
        assert all(
            0 <= token < model.config.vocab_size for token in next_tokens
        ), "All tokens should be valid"

        # Check that requests were updated with their respective parameters
        assert (
            request_deterministic.temperature == 0.1
        ), "Deterministic request should keep its temperature"
        assert (
            request_creative.temperature == 1.5
        ), "Creative request should keep its temperature"
