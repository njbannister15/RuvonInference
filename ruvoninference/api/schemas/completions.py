from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class CompletionRequest(BaseModel):
    """
    Request model for text completion endpoint.

    This follows OpenAI's completion API structure while supporting our
    advanced sampling capabilities for creative text generation.
    """

    prompt: str = Field(..., description="The text prompt to complete")
    max_tokens: int = Field(
        default=20, ge=1, le=500, description="Maximum tokens to generate"
    )
    model: str = Field(default="gpt2", description="Model to use for generation")
    stream: bool = Field(default=False, description="Whether to stream the response")
    use_cache: bool = Field(
        default=True, description="Whether to use KV-cache optimization"
    )

    # Sampling parameters
    temperature: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Temperature for sampling (0.1=focused, 1.0=balanced, 2.0=creative)",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=1000,
        description="Top-k sampling: only consider k most likely tokens",
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.01,
        le=1.0,
        description="Nucleus sampling: dynamic cutoff based on cumulative probability",
    )

    # Attention implementation parameter
    attention_implementation: Optional[str] = Field(
        default="eager",
        description="Attention implementation: 'eager', 'flash_attention_2', or 'sdpa'",
    )

    @field_validator("attention_implementation")
    @classmethod
    def validate_attention_implementation(cls, v):
        """Convert None to default value to handle explicit null in JSON."""
        if v is None:
            return "eager"
        return v


class CompletionChoice(BaseModel):
    """Single completion choice in the response."""

    text: str
    index: int
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    """
    Response model for text completion.

    Follows OpenAI API structure for compatibility with existing clients.
    """

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Dict[str, int]


class CompletionStreamChunk(BaseModel):
    """Single chunk in a streaming completion response."""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[Dict[str, Any]]
