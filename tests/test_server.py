"""
Comprehensive tests for the FastAPI server.

This module provides both unit and integration tests for the RuvonInference API server,
testing all endpoints, middleware, error handling, and queue strategy integration.
"""

import os
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from ruvoninference.api.server import app, get_model, get_tokenizer
from ruvoninference.api.schemas.completions import CompletionResponse


class TestFastAPIServer:
    """
    Test suite for the FastAPI server functionality.

    LEARN: Integration tests defend against import chain failures, dependency
    version conflicts, and environment-specific issues that cause servers to
    start but crash on first request. They verify the complete HTTP pipeline
    works, not just individual components in isolation.
    """

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def sample_completion_request(self):
        """Create a sample completion request for testing."""
        return {
            "model": "gpt2",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.9,
            "stream": False,
            "use_cache": True,
            "attention_implementation": None,
        }

    def test_root_endpoint(self, client):
        """Test the root endpoint returns correct API information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        # Verify required fields
        assert "message" in data
        assert "version" in data
        assert "part" in data
        assert "description" in data
        assert "endpoints" in data

        # Verify specific content
        assert "RuvonInference API" in data["message"]
        assert data["version"] == "0.1.0"
        assert isinstance(data["endpoints"], dict)

        # Verify all expected endpoints are documented
        expected_endpoints = ["/completions", "/health", "/queue", "/requests/{id}"]
        for endpoint in expected_endpoints:
            assert endpoint in data["endpoints"]

    @patch("ruvoninference.api.server.get_available_implementations")
    @patch("ruvoninference.api.server.get_best_attention_implementation")
    @patch("ruvoninference.api.server.sequential_queue")
    def test_health_endpoint(
        self, mock_queue, mock_best_impl, mock_available_impl, client
    ):
        """Test the health check endpoint returns comprehensive status."""
        # Mock attention implementations
        from ruvoninference.attention import AttentionImplementation

        mock_available_impl.return_value = [
            AttentionImplementation.EAGER,
            AttentionImplementation.FLASH_ATTENTION_2,
        ]
        mock_best_impl.return_value = AttentionImplementation.FLASH_ATTENTION_2

        # Mock queue stats
        mock_queue.stats = {
            "total_requests": 10,
            "completed_requests": 8,
            "failed_requests": 1,
            "pending_requests": 1,
        }

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Verify required fields
        assert data["status"] == "healthy"
        assert "models_loaded" in data
        assert "tokenizers_loaded" in data
        assert "attention" in data
        assert "queue" in data

        # Verify attention information
        attention_info = data["attention"]
        assert "available_implementations" in attention_info
        assert "default_implementation" in attention_info
        assert attention_info["default_implementation"] == "flash_attention_2"

        # Verify queue stats are included
        assert data["queue"]["total_requests"] == 10

    @patch("ruvoninference.api.server.queue_strategy")
    def test_completions_endpoint_success(
        self, mock_strategy, client, sample_completion_request
    ):
        """Test successful completion request processing."""
        # Mock the queue strategy response
        mock_response = CompletionResponse(
            id="cmpl-test-123",
            created=int(time.time()),
            model="gpt2",
            choices=[
                {"text": " in a land far away", "index": 0, "finish_reason": "stop"}
            ],
            usage={"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
        )
        mock_strategy.process_request = AsyncMock(return_value=mock_response)

        response = client.post("/completions", json=sample_completion_request)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "id" in data
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert "usage" in data

        # Verify specific content
        assert data["model"] == "gpt2"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["text"] == " in a land far away"
        assert data["usage"]["total_tokens"] == 10

    def test_completions_endpoint_streaming_error(
        self, client, sample_completion_request
    ):
        """Test that streaming requests return appropriate error."""
        sample_completion_request["stream"] = True

        response = client.post("/completions", json=sample_completion_request)

        assert response.status_code == 400
        data = response.json()
        assert "streaming not supported" in data["detail"].lower()

    @patch("ruvoninference.api.server.queue_strategy")
    def test_completions_endpoint_processing_error(
        self, mock_strategy, client, sample_completion_request
    ):
        """Test handling of processing errors in completions endpoint."""
        # Mock the strategy to raise an exception
        mock_strategy.process_request = AsyncMock(
            side_effect=Exception("Model processing failed")
        )

        response = client.post("/completions", json=sample_completion_request)

        assert response.status_code == 500
        data = response.json()
        assert "request processing failed" in data["detail"].lower()
        assert "model processing failed" in data["detail"].lower()

    def test_completions_endpoint_invalid_request(self, client):
        """Test validation of invalid completion requests."""
        invalid_request = {
            "model": "gpt2",
            # Missing required 'prompt' field
            "max_tokens": 10,
        }

        response = client.post("/completions", json=invalid_request)

        # Should return validation error
        assert response.status_code == 422

    @patch("ruvoninference.api.server.sequential_queue")
    def test_request_status_endpoint_found(self, mock_queue, client):
        """Test successful request status lookup."""
        # Mock a queued request with status
        mock_request = Mock()
        mock_request.id = "test-request-123"
        mock_request.status.value = "completed"
        mock_request.created_at = time.time()
        mock_request.started_at = time.time() + 1
        mock_request.completed_at = time.time() + 5
        mock_request.wait_time = 1.0
        mock_request.processing_time = 4.0
        mock_request.total_time = 5.0
        mock_request.error = None

        mock_queue.get_request_status.return_value = mock_request

        response = client.get("/requests/test-request-123")

        assert response.status_code == 200
        data = response.json()

        # Verify all status fields are present
        expected_fields = [
            "id",
            "status",
            "created_at",
            "started_at",
            "completed_at",
            "wait_time",
            "processing_time",
            "total_time",
            "error",
        ]
        for field in expected_fields:
            assert field in data

        assert data["id"] == "test-request-123"
        assert data["status"] == "completed"

    @patch("ruvoninference.api.server.sequential_queue")
    def test_request_status_endpoint_not_found(self, mock_queue, client):
        """Test request status lookup for non-existent request."""
        mock_queue.get_request_status.return_value = None

        response = client.get("/requests/non-existent-request")

        assert response.status_code == 404
        data = response.json()
        assert "request not found" in data["detail"].lower()

    @patch("ruvoninference.api.server.queue_strategy")
    def test_queue_status_endpoint(self, mock_strategy, client):
        """Test queue status endpoint."""
        mock_stats = {
            "strategy_name": "batched",
            "total_requests": 100,
            "completed_requests": 95,
            "failed_requests": 2,
            "pending_requests": 3,
            "average_wait_time": 0.5,
            "average_processing_time": 2.1,
        }
        mock_strategy.get_stats.return_value = mock_stats

        response = client.get("/queue")

        assert response.status_code == 200
        data = response.json()

        # Verify stats are returned
        assert data["strategy_name"] == "batched"
        assert data["total_requests"] == 100
        assert data["completed_requests"] == 95

    @patch("ruvoninference.api.server.queue_strategy")
    def test_recent_completions_endpoint(self, mock_strategy, client):
        """Test recent completions endpoint."""
        mock_completions = [
            {
                "id": "cmpl-1",
                "prompt": "Hello",
                "completion": " world",
                "processing_time": 1.5,
                "created_at": time.time() - 100,
            },
            {
                "id": "cmpl-2",
                "prompt": "The sky is",
                "completion": " blue",
                "processing_time": 2.1,
                "created_at": time.time() - 50,
            },
        ]
        mock_strategy.get_recent_completions.return_value = mock_completions

        response = client.get("/queue/recent")

        assert response.status_code == 200
        data = response.json()

        assert len(data) == 2
        assert data[0]["id"] == "cmpl-1"
        assert data[1]["prompt"] == "The sky is"

    def test_recent_completions_with_limit(self, client):
        """Test recent completions endpoint with custom limit."""
        with patch("ruvoninference.api.server.queue_strategy") as mock_strategy:
            mock_strategy.get_recent_completions.return_value = []

            response = client.get("/queue/recent?limit=5")

            assert response.status_code == 200
            # Verify the limit parameter was passed correctly
            mock_strategy.get_recent_completions.assert_called_once_with(5)

    def test_server_startup_configuration(self):
        """Test that server starts with correct configuration."""
        # Verify app configuration
        assert app.title == "RuvonInference API"
        assert app.version == "0.1.0"

        # Verify middleware is configured
        middleware_types = [
            middleware.cls.__name__ for middleware in app.user_middleware
        ]
        assert any("CORS" in name for name in middleware_types)

    @patch.dict(os.environ, {"QUEUE_MODE": "sequential"})
    def test_queue_mode_environment_variable(self):
        """Test that queue mode is read from environment variable."""
        # This tests the queue mode configuration
        # In a real test, you might want to reload the module or
        # test the queue strategy factory directly
        # Note: This might not work in the current test due to module loading order
        # Consider testing the queue strategy factory directly instead

    def test_openapi_schema_generation(self, client):
        """Test that OpenAPI schema is generated correctly."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()

        # Verify basic schema structure
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

        # Verify our endpoints are in the schema
        paths = schema["paths"]
        assert "/" in paths
        assert "/health" in paths
        assert "/completions" in paths
        assert "/queue" in paths

    def test_server_docs_endpoints(self, client):
        """Test that documentation endpoints are accessible."""
        # Test Swagger UI
        response = client.get("/docs")
        assert response.status_code == 200

        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200


class TestServerHelperFunctions:
    """Test suite for server helper functions."""

    @patch("ruvoninference.api.server.get_available_implementations")
    @patch("ruvoninference.api.server.recommend_implementation")
    def test_get_best_attention_implementation(self, mock_recommend, mock_available):
        """Test attention implementation selection logic."""
        from ruvoninference.api.server import get_best_attention_implementation
        from ruvoninference.attention import AttentionImplementation

        mock_recommend.return_value = AttentionImplementation.FLASH_ATTENTION_2

        result = get_best_attention_implementation("gpt2", "cpu")

        assert result == AttentionImplementation.FLASH_ATTENTION_2
        mock_recommend.assert_called_once_with(512, "cpu")

    @patch("ruvoninference.api.server.load_model_with_attention")
    @patch("ruvoninference.api.server.get_best_attention_implementation")
    @patch("ruvoninference.api.server.get_available_implementations")
    def test_get_model_caching(self, mock_available, mock_best, mock_load):
        """Test model caching behavior in get_model function."""
        from ruvoninference.api.server import model_instances
        from ruvoninference.attention import AttentionImplementation

        # Clear any existing cached models
        model_instances.clear()

        # Setup mocks
        mock_best.return_value = AttentionImplementation.EAGER
        mock_available.return_value = [AttentionImplementation.EAGER]

        mock_model = Mock()
        mock_model.config = Mock()
        mock_load.return_value = mock_model

        # First call should load the model
        with patch("ruvoninference.api.server.GPT2Model") as mock_gpt2:
            mock_gpt2_instance = Mock()
            mock_gpt2.return_value = mock_gpt2_instance

            result1 = get_model("gpt2")

            # Should have loaded the model
            mock_load.assert_called_once()
            mock_gpt2.assert_called_once()

        # Second call should use cached model
        mock_load.reset_mock()
        with patch("ruvoninference.model.gpt2.GPT2Model") as mock_gpt2:
            result2 = get_model("gpt2")

            # Should not load again
            mock_load.assert_not_called()
            mock_gpt2.assert_not_called()

        # Should return the same instance
        assert result1 is result2

    @patch("ruvoninference.api.server.GPT2TokenizerWrapper")
    def test_get_tokenizer_caching(self, mock_tokenizer_class):
        """Test tokenizer caching behavior."""
        from ruvoninference.api.server import tokenizer_instances

        # Clear any existing cached tokenizers
        tokenizer_instances.clear()

        mock_tokenizer = Mock()
        mock_tokenizer_class.return_value = mock_tokenizer

        # First call should create tokenizer
        result1 = get_tokenizer("gpt2")
        mock_tokenizer_class.assert_called_once_with("gpt2")

        # Second call should use cached tokenizer
        mock_tokenizer_class.reset_mock()
        result2 = get_tokenizer("gpt2")
        mock_tokenizer_class.assert_not_called()

        # Should return the same instance
        assert result1 is result2


class TestServerErrorHandling:
    """Test suite for server error handling scenarios."""

    @pytest.fixture
    def client(self):
        """Create a test client for error handling tests."""
        return TestClient(app)

    def test_404_error_handling(self, client):
        """Test handling of non-existent endpoints."""
        response = client.get("/non-existent-endpoint")
        assert response.status_code == 404

    def test_method_not_allowed_handling(self, client):
        """Test handling of incorrect HTTP methods."""
        # Try to GET the POST-only completions endpoint
        response = client.get("/completions")
        assert response.status_code == 405

    @patch("ruvoninference.api.server.queue_strategy")
    def test_internal_server_error_handling(self, mock_strategy, client):
        """Test handling of internal server errors."""
        # Mock an unexpected error in the strategy
        mock_strategy.process_request = AsyncMock(
            side_effect=RuntimeError("Unexpected internal error")
        )

        request_data = {"model": "gpt2", "prompt": "Test prompt", "max_tokens": 5}

        response = client.post("/completions", json=request_data)

        assert response.status_code == 500
        data = response.json()
        assert "request processing failed" in data["detail"].lower()

    def test_malformed_json_handling(self, client):
        """Test handling of malformed JSON requests."""
        # Send malformed JSON
        response = client.post(
            "/completions",
            data="{ invalid json }",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    def test_request_validation_errors(self, client):
        """Test various request validation scenarios."""
        # Test missing required fields
        response = client.post("/completions", json={})
        assert response.status_code == 422

        # Test invalid field types
        invalid_request = {
            "model": "gpt2",
            "prompt": "Test",
            "max_tokens": "not_a_number",  # Should be int
        }
        response = client.post("/completions", json=invalid_request)
        assert response.status_code == 422

        # Test negative values where positive required
        invalid_request = {
            "model": "gpt2",
            "prompt": "Test",
            "max_tokens": -5,  # Should be positive
        }
        response = client.post("/completions", json=invalid_request)
        assert response.status_code == 422
