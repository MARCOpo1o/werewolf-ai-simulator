import json
import unittest

from werewolf.llm.fake_provider import (
    FakeProvider,
    FakeProviderExhausted,
    error_result,
    estimated_cost_result,
    success_result,
)
from werewolf.llm.provider import ModelRequest, Provider
from werewolf.llm.records import CostSource, ErrorCategory


def make_request(**overrides) -> ModelRequest:
    defaults = dict(model="fake-model-1", system_prompt="sys", user_prompt="usr")
    defaults.update(overrides)
    return ModelRequest(**defaults)


class FakeProviderTests(unittest.TestCase):
    def test_satisfies_provider_protocol(self):
        self.assertIsInstance(FakeProvider(), Provider)

    def test_returns_scripted_results_in_order_and_captures_requests(self):
        r1 = success_result({"thought": "a"})
        r2 = error_result(ErrorCategory.RATE_LIMITED)
        provider = FakeProvider([r1, r2])

        out1 = provider.complete(make_request(user_prompt="first"))
        out2 = provider.complete(make_request(user_prompt="second"))

        self.assertIs(out1, r1)
        self.assertIs(out2, r2)
        self.assertEqual(provider.calls_made, 2)
        self.assertEqual(provider.requests[0].user_prompt, "first")
        self.assertEqual(provider.requests[1].user_prompt, "second")

    def test_exhaustion_raises(self):
        provider = FakeProvider([success_result({})])
        provider.complete(make_request())
        with self.assertRaises(FakeProviderExhausted):
            provider.complete(make_request())


class ResultFactoryTests(unittest.TestCase):
    def test_success_result_carries_exact_ticks_and_valid_json(self):
        result = success_result(
            {"thought": "x", "action": {"vote_target": 2}},
            input_tokens=400,
            output_tokens=50,
            reasoning_tokens=120,
            cost_ticks=37_756_000,
        )
        self.assertTrue(result.ok)
        parsed = json.loads(result.text)
        self.assertEqual(parsed["action"]["vote_target"], 2)
        self.assertEqual(result.cost.ticks, 37_756_000)
        self.assertEqual(result.cost.source, CostSource.PROVIDER_REPORTED)
        self.assertEqual(result.usage.total_tokens, 400 + 50 + 120)
        self.assertEqual(result.usage.reasoning_tokens, 120)

    def test_success_result_with_raw_text_for_malformed_json(self):
        result = success_result(text='{"thought": "oops', cost_ticks=1000)
        self.assertTrue(result.ok)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(result.text)
        # Malformed output still cost money.
        self.assertEqual(result.cost.ticks, 1000)

    def test_success_result_without_cost(self):
        result = success_result({}, cost_ticks=None)
        self.assertTrue(result.ok)
        self.assertEqual(result.cost.source, CostSource.UNAVAILABLE)
        self.assertIsNone(result.cost.usd)

    def test_estimated_cost_result(self):
        result = estimated_cost_result({"a": 1}, usd=0.0031)
        self.assertEqual(result.cost.source, CostSource.PRICING_TABLE_ESTIMATE)
        self.assertAlmostEqual(result.cost.usd, 0.0031)
        self.assertIsNone(result.cost.ticks)

    def test_error_result_default_retryability(self):
        retryable = (
            ErrorCategory.RATE_LIMITED,
            ErrorCategory.TIMEOUT,
            ErrorCategory.NETWORK_ERROR,
            ErrorCategory.PROVIDER_ERROR,
        )
        non_retryable = (
            ErrorCategory.AUTHENTICATION_ERROR,
            ErrorCategory.CONTEXT_WINDOW_EXCEEDED,
            ErrorCategory.MISSING_API_KEY,
        )
        for category in retryable:
            self.assertTrue(error_result(category).retryable, category)
        for category in non_retryable:
            self.assertFalse(error_result(category).retryable, category)

    def test_error_result_can_still_carry_billed_cost(self):
        result = error_result(ErrorCategory.MAX_OUTPUT_TOKENS, cost_ticks=999)
        self.assertFalse(result.ok)
        self.assertEqual(result.cost.ticks, 999)
        self.assertEqual(result.cost.source, CostSource.PROVIDER_REPORTED)


if __name__ == "__main__":
    unittest.main()
