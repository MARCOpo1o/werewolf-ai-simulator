"""Response and memory bandwidth limits (benchmark fairness).

Without limits, a verbose model gets longer public messages (more
persuasion bandwidth) and a larger persistent memory (more effective
context) than a concise model, confounding capability comparisons.

Enforcement is deterministic truncation, never retries: retrying would
cost money and change behavior. Messages are truncated at emission (the
original length is recorded on the event); memory is truncated at
render time only (stored memory stays intact). Structured belief fields
are exempt - they are instrumentation, not gameplay bandwidth.

An unconstrained-language condition can be run later by raising these in
a dedicated, clearly-labeled commit (limits are logged per game).
"""
from typing import Optional

LIMITS_VERSION = 1

PUBLIC_MESSAGE_MAX_CHARS = 800
WOLF_MESSAGE_MAX_CHARS = 800
MEMORY_MAX_CHARS = 2500


def truncate_text(text: str, limit: int) -> tuple[str, Optional[int]]:
    """Returns (possibly-truncated text, original length if truncated)."""
    if text is None or len(text) <= limit:
        return text, None
    return text[:limit], len(text)


def limits_dict() -> dict:
    return {
        "limits_version": LIMITS_VERSION,
        "public_message_max_chars": PUBLIC_MESSAGE_MAX_CHARS,
        "wolf_message_max_chars": WOLF_MESSAGE_MAX_CHARS,
        "memory_max_chars": MEMORY_MAX_CHARS,
    }
