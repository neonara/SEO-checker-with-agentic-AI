"""Errors an audit can end with on purpose."""
from __future__ import annotations


class AuditFailed(RuntimeError):
    """The audit stopped because it could not produce an honest report.

    `code` is a short machine-readable reason ("blocked", "too_few_checks",
    "timeout"). `public_message` is safe to show a visitor as-is: it never
    carries provider error text or anything a model wrote."""

    def __init__(self, code: str, public_message: str):
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message


def is_quota_error(exc: BaseException) -> bool:
    """True if `exc` is the model provider refusing for rate/quota reasons --
    either the SDK's own 429 or the RuntimeError base_agent raises once it
    has run out of fallbacks."""
    if getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return "rate/quota limit" in text or "rate limit" in text or "quota" in text
