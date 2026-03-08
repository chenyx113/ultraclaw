"""LLM provider adapters for Ultra-Claw."""

from ultra_claw.services.llm.base import LLMProvider, LLMResponse
from ultra_claw.services.llm.mock_provider import MockProvider

# Optional providers - import only if available
try:
    from ultra_claw.services.llm.openai_provider import OpenAIProvider
except ImportError:
    OpenAIProvider = None  # type: ignore

try:
    from ultra_claw.services.llm.anthropic_provider import AnthropicProvider
except ImportError:
    AnthropicProvider = None  # type: ignore

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "OpenAIProvider",
    "AnthropicProvider",
    "MockProvider",
]
