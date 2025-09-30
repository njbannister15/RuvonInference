"""
Smoke tests for GPT2Model class - Educational & Understanding Focused.

These are NOT correctness tests for GPT-2 itself, but rather smoke tests designed
for understanding how the model works and for people stepping into the codebase.
The purpose is to demonstrate model behavior, tensor shapes, and provide a safe
environment to experiment and learn how language models function.

Think of these as "learning labs" rather than traditional unit tests.
"""

import pytest
import torch

from ruvonvllm.model.gpt2 import GPT2Model


class TestGPT2ModelErrorCases:
    """Demonstrate error handling - helps understand model lifecycle and usage patterns."""

    def test_forward_model_not_loaded(self):
        """Demonstrate what happens when you try to use an unloaded model.

        This helps newcomers understand the model loading lifecycle and
        provides a clear example of proper error handling.
        """
        model = GPT2Model("gpt2")
        input_ids = torch.tensor([[1, 2, 3]])

        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.forward(input_ids)


class TestGPT2ModelSmokeTests:
    """Smoke tests using real models - for learning and understanding model behavior.

    These tests load an actual GPT-2 model to demonstrate:
    - How tensors flow through the model
    - What logits look like in practice
    - How context affects predictions
    - Real examples you can experiment with
    """

    @pytest.fixture(scope="class")
    def real_model(self):
        """Load real GPT-2 model for demonstration and learning.

        This fixture provides a working model that you can experiment with.
        Try changing the model size or device to see how it affects behavior.
        """
        model = GPT2Model("gpt2", device="cpu")
        model.load_model()
        return model

    @pytest.mark.slow
    def test_forward_produces_valid_logits(self, real_model):
        """Demonstrate what model logits look like and their properties.

        This shows you:
        - Exact tensor shapes you'll see in practice
        - How to check for common numerical issues (NaN, inf)
        - How logits convert to probabilities via softmax
        - What a real language model output looks like
        """
        input_ids = torch.tensor([[7454, 2402, 257, 640]])  # "Once upon a time"

        logits = real_model.forward(input_ids)

        # Examine the logits tensor - this is what the model actually outputs
        # Shape: [batch_size=1, sequence_length=4, vocabulary_size=50257]
        assert logits.shape == (1, 4, 50257)
        assert not torch.isnan(logits).any(), "Model produced NaN logits"
        assert not torch.isinf(logits).any(), "Model produced infinite logits"

        # Convert raw logits to probabilities - this is how we get token predictions
        # The softmax function turns unbounded logits into a probability distribution
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        assert torch.allclose(
            probs.sum(), torch.tensor(1.0), atol=1e-6
        ), "Probabilities don't sum to 1"
        assert (probs >= 0).all(), "Found negative probabilities"

    @pytest.mark.slow
    def test_sequence_length_affects_predictions(self, real_model):
        """Demonstrate how context length affects model predictions.

        This is a key concept in language models - longer context provides
        more information for better predictions. You can experiment with
        different contexts to see this effect in action.
        """
        # Compare predictions with different amounts of context
        # This demonstrates the core principle of transformer language models
        short_context = torch.tensor([[7454]])  # "Once"
        long_context = torch.tensor([[7454, 2402, 257]])  # "Once upon a"

        short_logits = real_model.forward(short_context)
        long_logits = real_model.forward(long_context)

        # Convert logits to probabilities to compare distributions
        # This shows how context changes the model's "confidence" in different tokens
        short_probs = torch.softmax(short_logits[0, -1, :], dim=-1)
        long_probs = torch.softmax(long_logits[0, -1, :], dim=-1)

        # Use KL divergence to measure how different the distributions are
        # This quantifies the impact of additional context
        kl_div = torch.sum(short_probs * torch.log(short_probs / (long_probs + 1e-10)))
        assert kl_div > 0.01, "Context didn't meaningfully change predictions"

    @pytest.mark.slow
    def test_model_produces_reasonable_next_tokens(self, real_model):
        """Explore what tokens the model predicts for common phrases.

        This demonstrates:
        - How to get top-k predictions from logits
        - What "reasonable" model behavior looks like
        - How to inspect and understand model outputs
        - Examples you can modify to test your own phrases
        """
        # Try some common English patterns to see what the model predicts
        # Feel free to add your own test cases here to explore model behavior
        test_cases = [
            (torch.tensor([[464]]), "The"),  # "The" should predict common words
            (
                torch.tensor([[1812, 318]]),
                "is",
            ),  # "This is" should predict reasonable continuations
        ]

        for input_ids, context_desc in test_cases:
            logits = real_model.forward(input_ids)

            # Extract the most likely next tokens - this is how you'd implement sampling
            # The top_indices tell you which tokens the model thinks are most likely
            _, top_indices = torch.topk(logits[0, -1, :], 5)

            # Ensure the model outputs are within the expected vocabulary range
            # GPT-2 has 50,257 tokens in its vocabulary
            assert all(
                0 <= idx < 50257 for idx in top_indices
            ), f"Invalid token IDs for '{context_desc}'"

            # Check that the model has clear preferences (not just random guessing)
            # A healthy model should have some tokens much more likely than others
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            max_prob = probs.max().item()
            assert (
                max_prob > 0.01
            ), f"No clear prediction for '{context_desc}' (max prob: {max_prob})"


# Test fixtures for reuse across test files
@pytest.fixture(scope="session")
def sample_token_sequences():
    """Sample token sequences for testing."""
    return {
        "hello_world": torch.tensor([[15496, 995]]),  # "Hello world"
        "once_upon_time": torch.tensor([[7454, 2402, 257, 640]]),  # "Once upon a time"
        "single_token": torch.tensor([[50256]]),  # EOS token
        "longer_sequence": torch.randint(0, 1000, (1, 20)),  # Random 20 tokens
        "batch_sequences": torch.randint(0, 1000, (3, 10)),  # Batch of 3 sequences
    }


@pytest.fixture(scope="session")
def expected_shapes():
    """Expected output shapes for different inputs."""
    return {
        "vocab_size": 50257,
        "single_token": (1, 1, 50257),
        "hello_world": (1, 2, 50257),
        "once_upon_time": (1, 4, 50257),
        "longer_sequence": (1, 20, 50257),
        "batch_sequences": (3, 10, 50257),
    }


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
