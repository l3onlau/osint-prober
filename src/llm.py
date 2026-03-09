"""
Shared LLM factory.

All agents import from here to avoid duplicate `get_llm()` definitions
and to keep model/temperature configuration in one place.
"""

from langchain_ollama import ChatOllama

from src.config import config


def get_llm(temperature: float = 0.7, **kwargs) -> ChatOllama:
    """Returns a ChatOllama instance with recommended Qwen3 parameters.

    Qwen3 Reference Parameters for optimal performance:
    - Temperature=0.7 (default, overriding specific agent temps if needed, though they are passed)
    - TopP=0.8, TopK=20, MinP=0.0
    - Context limit (num_ctx) capped at 16384
    - Repeat penalty applied to prevent hallucination loops
    """
    params = {
        "temperature": temperature,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "num_ctx": 16384,  # Max recommended context length
        "num_predict": 4096,  # Sane output limit
        "repeat_penalty": 1.1,  # Slight penalty to break repetition loops
        "repeat_last_n": 64,  # Look back 64 tokens for repetition
    }
    params.update(kwargs)

    return ChatOllama(model=config.ollama_model, **params)


def get_planner_llm() -> ChatOllama:
    """LLM for the Planner agent (creative query generation)."""
    return get_llm(config.planner_temperature)


def get_gatherer_llm() -> ChatOllama:
    """LLM for the Gatherer agent (extraction)."""
    return get_llm(config.gatherer_temperature)


def get_synthesis_llm() -> ChatOllama:
    """LLM for the Synthesis agent (Q&A)."""
    return get_llm(config.synthesis_temperature)


def get_briefing_llm() -> ChatOllama:
    """LLM for the Briefing agent (narrative generation)."""
    return get_llm(config.briefing_temperature)
