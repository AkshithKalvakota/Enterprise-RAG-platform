from langchain_google_genai import ChatGoogleGenerativeAI

from utils import config


def build_llm(model: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        # Gemini 3.x thinks before answering by default; "low" cuts latency a lot
        # for document lookups. Options: minimal, low, medium, high.
        thinking_level=config.LLM_THINKING_LEVEL,
        # Fail fast so the fallback model can take over, instead of retrying for a minute
        max_retries=config.LLM_MAX_ATTEMPTS,
        timeout=config.LLM_TIMEOUT_SECONDS,
    )


def build_llm_with_fallback():
    """
    The primary model; if it is overloaded (503), out of quota (429) or times out,
    the same request is sent to the fallback model instead of failing.
    """
    primary = build_llm(config.LLM_MODEL)
    if config.LLM_FALLBACK_MODEL and config.LLM_FALLBACK_MODEL != config.LLM_MODEL:
        return primary.with_fallbacks([build_llm(config.LLM_FALLBACK_MODEL)])
    return primary


def message_text(response) -> str:
    """Gemini 3 can return several content parts; keep all text parts."""
    content = response.content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)
