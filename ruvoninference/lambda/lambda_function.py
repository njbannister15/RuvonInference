"""
AWS Lambda function for RuvonInference - Always-on demo deployment.

This Lambda function directly uses the existing GPT2Model and GPT2TokenizerWrapper
classes without any additional abstraction layers.
"""

import json
import logging
import time
import traceback

from ruvoninference.model.gpt2 import GPT2Model
from ruvoninference.tokenizer.gpt2_tokenizer import GPT2TokenizerWrapper

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global instances cache (survives Lambda warm starts)
_model = None
_tokenizer = None
_initialization_time = None


def lambda_handler(event, context):
    """
    AWS Lambda entry point for RuvonInference text generation.

    Uses existing RuvonInference classes directly.
    """
    global _model, _tokenizer, _initialization_time

    start_time = time.time()

    try:
        logger.info(f"Processing request: {json.dumps(event, default=str)}")

        # Parse request body
        if "body" in event:
            if isinstance(event["body"], str):
                try:
                    body = json.loads(event["body"])
                except json.JSONDecodeError:
                    body = {"prompt": event["body"]}
            else:
                body = event["body"] or {}
        else:
            body = event

        # Extract parameters with validation
        prompt = body.get("prompt", "The future of AI is")
        max_tokens = min(int(body.get("max_tokens", 20)), 100)  # Cap for Lambda
        temperature = max(0.1, min(float(body.get("temperature", 1.0)), 2.0))
        top_k = body.get("top_k")
        top_p = body.get("top_p")
        model_name = body.get("model", "gpt2")

        if top_k is not None:
            top_k = max(1, min(int(top_k), 200))
        if top_p is not None:
            top_p = max(0.01, min(float(top_p), 1.0))

        logger.info(
            f"Parameters: prompt='{prompt[:30]}...', max_tokens={max_tokens}, temp={temperature}"
        )

        # Initialize model and tokenizer if needed (cold start)
        if _model is None or _tokenizer is None:
            init_start = time.time()
            logger.info("Cold start: Loading model and tokenizer...")

            _model = GPT2Model(model_name, device="cpu")
            _model.load_model()

            _tokenizer = GPT2TokenizerWrapper(model_name)

            _initialization_time = time.time() - init_start
            logger.info(f"Model and tokenizer loaded in {_initialization_time:.2f}s")
        else:
            logger.info("Warm start: Using cached model and tokenizer")

        # Generate using existing RuvonInference functionality
        generation_start = time.time()

        # Tokenize using existing tokenizer
        input_ids = _tokenizer.encode(prompt, return_tensors=True)

        # Generate using existing model method
        generated_tokens = _model.generate_with_sampling(
            input_ids,
            max_length=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=True,
            show_progress=False,
        )

        # Decode using existing tokenizer
        full_text = _tokenizer.decode(generated_tokens)
        generated_text = full_text[len(prompt) :]

        generation_time = time.time() - generation_start
        total_time = time.time() - start_time

        # Create response
        result = {
            "success": True,
            "data": {
                "prompt": prompt,
                "generated_text": generated_text,
                "full_text": full_text,
                "metadata": {
                    "model": model_name,
                    "prompt_tokens": len(input_ids[0]),
                    "completion_tokens": len(generated_tokens) - len(input_ids[0]),
                    "total_tokens": len(generated_tokens),
                    "generation_time_ms": round(generation_time * 1000, 2),
                    "initialization_time_ms": round(
                        (_initialization_time or 0) * 1000, 2
                    ),
                    "parameters": {
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                    },
                },
                "lambda_metadata": {
                    "function_name": context.function_name,
                    "request_id": context.aws_request_id,
                    "remaining_time_ms": context.get_remaining_time_in_millis(),
                    "total_time_ms": round(total_time * 1000, 2),
                    "memory_limit_mb": context.memory_limit_in_mb,
                    "cold_start": _initialization_time is not None
                    and _initialization_time > 0,
                },
            },
        }

        logger.info(
            f"Generated response in {total_time:.2f}s: {generated_text[:50]}..."
        )

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
            },
            "body": json.dumps(result),
        }

    except Exception as e:
        error_details = {
            "success": False,
            "error": {
                "message": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc(),
            },
            "lambda_metadata": {
                "function_name": getattr(context, "function_name", "unknown"),
                "request_id": getattr(context, "aws_request_id", "unknown"),
                "total_time_ms": round((time.time() - start_time) * 1000, 2),
            },
        }

        logger.error(f"Lambda execution failed: {error_details}")

        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
            },
            "body": json.dumps(error_details),
        }


# For local testing
if __name__ == "__main__":
    # Test the function locally
    test_event = {
        "prompt": "Once upon a time in a land far away",
        "max_tokens": 15,
        "temperature": 0.8,
        "top_k": 40,
    }

    class MockContext:
        function_name = "ruvoninference-demo"
        aws_request_id = "test-request-123"
        memory_limit_in_mb = 1024

        def get_remaining_time_in_millis(self):
            return 30000

    print("Testing Lambda function locally...")
    result = lambda_handler(test_event, MockContext())
    print(json.dumps(result, indent=2))
