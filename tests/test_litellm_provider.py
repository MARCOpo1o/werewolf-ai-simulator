import unittest
from types import SimpleNamespace

try:
    import litellm  # noqa: F401
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False

from werewolf.llm.records import CostSource, ErrorCategory

if HAS_LITELLM:
    from werewolf.llm.litellm_provider import (
        LiteLLMProvider,
        classify_exception,
        _sanitize_error_message,
    )


@unittest.skipUnless(HAS_LITELLM, "litellm not installed")
class ClassifyExceptionTests(unittest.TestCase):
    def test_litellm_exception_classes_map(self):
        cases = {
            litellm.exceptions.RateLimitError: ErrorCategory.RATE_LIMITED,
            litellm.exceptions.AuthenticationError: ErrorCategory.AUTHENTICATION_ERROR,
            litellm.exceptions.ContextWindowExceededError: ErrorCategory.CONTEXT_WINDOW_EXCEEDED,
        }
        for exc_class, expected in cases.items():
            try:
                exc = exc_class(
                    message="x", model="gemini/x", llm_provider="gemini"
                )
            except TypeError:
                self.skipTest(f"{exc_class.__name__} signature changed")
            self.assertEqual(classify_exception(exc), expected)

    def test_unknown_exception_falls_back_by_message(self):
        self.assertEqual(
            classify_exception(Exception("Rate limit exceeded, 429")),
            ErrorCategory.RATE_LIMITED,
        )
        self.assertEqual(
            classify_exception(Exception("something exploded")),
            ErrorCategory.PROVIDER_ERROR,
        )

    def test_error_messages_redact_keys(self):
        msg = _sanitize_error_message(
            Exception("auth failed for key AIzaSyFAKEKEY123 sk-alsofake")
        )
        self.assertNotIn("AIzaSyFAKEKEY123", msg)
        self.assertNotIn("sk-alsofake", msg)
        self.assertIn("[REDACTED]", msg)


@unittest.skipUnless(HAS_LITELLM, "litellm not installed")
class ResultExtractionTests(unittest.TestCase):
    def _fake_response(self, content='{"a": 1}', finish_reason="stop"):
        return SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=120,
                completion_tokens=30,
                total_tokens=150,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=8),
                prompt_tokens_details=SimpleNamespace(cached_tokens=40),
            ),
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )],
            model="gemini-3.1-flash-lite-preview",
            id="resp-123",
        )

    def test_usage_and_estimate_labeling(self):
        result = LiteLLMProvider._result_from_response(
            self._fake_response(), latency_ms=100
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.usage.input_tokens, 120)
        self.assertEqual(result.usage.reasoning_tokens, 8)
        self.assertEqual(result.usage.cached_input_tokens, 40)
        self.assertEqual(result.resolved_model, "gemini-3.1-flash-lite-preview")
        # Cost from a SimpleNamespace can't be computed by litellm's price
        # map -> must be UNAVAILABLE (never provider_reported, never 0.0).
        self.assertIn(
            result.cost.source,
            (CostSource.UNAVAILABLE, CostSource.PRICING_TABLE_ESTIMATE),
        )
        self.assertNotEqual(result.cost.source, CostSource.PROVIDER_REPORTED)
        self.assertIsNone(result.cost.ticks)

    def test_empty_content_flagged(self):
        result = LiteLLMProvider._result_from_response(
            self._fake_response(content=""), latency_ms=10
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error_category, ErrorCategory.EMPTY_RESPONSE)

    def test_length_finish_reason_flagged(self):
        result = LiteLLMProvider._result_from_response(
            self._fake_response(finish_reason="length"), latency_ms=10
        )
        self.assertTrue(result.ok)  # usable text, but flagged
        self.assertEqual(result.error_category, ErrorCategory.MAX_OUTPUT_TOKENS)

    def test_requires_api_key(self):
        with self.assertRaises(ValueError):
            LiteLLMProvider(api_key="")


if __name__ == "__main__":
    unittest.main()
