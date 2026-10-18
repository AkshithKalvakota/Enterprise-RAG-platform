import re
import time

from utils.logger import setup_logger

logger = setup_logger()

# Text found in Gemini errors that mean "busy, try again shortly"
_BUSY_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "TIMEOUT", "TIMED OUT", "DEADLINE")
# Quota errors say how long to wait, e.g. "Please retry in 6.342s" or "'retryDelay': '6s'"
_SUGGESTED_DELAY = re.compile(r"retry in ([\d.]+)s|retryDelay'?:\s*'?([\d.]+)s", re.IGNORECASE)
MAX_DELAY_SECONDS = 30


def is_model_busy_error(error: Exception) -> bool:
    """True for errors that mean "try again shortly": overloaded, quota exceeded, timeout."""
    text = str(error).upper()
    return any(marker in text for marker in _BUSY_MARKERS)


def _delay_for(error: Exception, attempt: int, base_delay: float) -> float:
    match = _SUGGESTED_DELAY.search(str(error))
    suggested = float(match.group(1) or match.group(2)) if match else 0
    backoff = base_delay * (2 ** (attempt - 1))  # 2s, 4s, 8s, ...
    return min(max(backoff, suggested), MAX_DELAY_SECONDS)


def call_with_retry(fn, *args, attempts: int = 4, base_delay: float = 2, **kwargs):
    """
    Calls fn, retrying only "busy" errors with growing waits (or the wait the API suggests).
    Any other error is raised immediately so real bugs are not hidden.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == attempts or not is_model_busy_error(e):
                raise
            delay = _delay_for(e, attempt, base_delay)
            logger.warning(f"Gemini busy (attempt {attempt}/{attempts}), retrying in {delay:.0f}s: {str(e)[:120]}")
            time.sleep(delay)
