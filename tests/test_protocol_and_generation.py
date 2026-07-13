import json
import tempfile
import unittest

from werewolf.engine.game import GameEngine
from werewolf.engine.limits import (
    MEMORY_MAX_CHARS,
    PUBLIC_MESSAGE_MAX_CHARS,
)
from werewolf.llm.fake_provider import FakeProvider, success_result
from werewolf.llm.provider import GenerationConfig, ModelRequest
from werewolf.llm.xai_provider import build_chat_kwargs


def talky_response(text="hello everyone"):
    return {
        "thought": "t",
        "say": {"public": text},
        "action": None,
    }


def run_game(seed=31, text="hello everyone", **engine_kwargs):
    provider = FakeProvider(default=success_result(talky_response(text), cost_ticks=1))
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = GameEngine(
            n_players=4, n_wolves=1, n_seers=0, seed=seed,
            output_dir=tmpdir, api_key="", provider=provider,
            transcript_enabled=False, show_all_channels=False,
            belief_snapshots=False,  # isolate discussion protocol
            **engine_kwargs,
        )
        engine.run()
        with open(engine.logger.filepath, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    return engine, provider, rows


def public_messages(rows, round_=1):
    return [
        e for e in (r["event"] for r in rows if r["type"] == "event")
        if e["type"] == "message" and e["channel"] == "public"
        and e["round"] == round_
    ]


class DiscussionProtocolTests(unittest.TestCase):
    def test_two_cycles_with_reversed_order(self):
        _, _, rows = run_game()
        messages = public_messages(rows)
        cycle1 = [m["speaker_id"] for m in messages
                  if m["payload"]["discussion_cycle"] == 1]
        cycle2 = [m["speaker_id"] for m in messages
                  if m["payload"]["discussion_cycle"] == 2]
        # 4 players, one killed at night -> 3 alive speakers
        self.assertEqual(len(cycle1), 3)
        self.assertEqual(cycle2, list(reversed(cycle1)))
        # positions are logged
        positions = [m["payload"]["speaker_position"] for m in messages
                     if m["payload"]["discussion_cycle"] == 1]
        self.assertEqual(positions, [1, 2, 3])

    def test_order_is_seeded_and_reproducible(self):
        _, _, rows_a = run_game(seed=31)
        _, _, rows_b = run_game(seed=31)
        order_a = [m["speaker_id"] for m in public_messages(rows_a)]
        order_b = [m["speaker_id"] for m in public_messages(rows_b)]
        self.assertEqual(order_a, order_b)

    def test_single_cycle_flag_restores_one_pass(self):
        _, _, rows = run_game(discussion_cycles=1)
        messages = public_messages(rows)
        self.assertEqual(len(messages), 3)  # 3 alive after the night kill
        self.assertTrue(all(
            m["payload"]["discussion_cycle"] == 1 for m in messages
        ))

    def test_invalid_cycles_rejected(self):
        with self.assertRaises(ValueError):
            run_game(discussion_cycles=0)


class BandwidthLimitTests(unittest.TestCase):
    def test_public_message_truncated_with_original_length_recorded(self):
        long_text = "x" * (PUBLIC_MESSAGE_MAX_CHARS + 500)
        _, _, rows = run_game(text=long_text)
        message = public_messages(rows)[0]
        self.assertEqual(len(message["payload"]["text"]), PUBLIC_MESSAGE_MAX_CHARS)
        self.assertEqual(
            message["payload"]["truncated_from"], PUBLIC_MESSAGE_MAX_CHARS + 500
        )

    def test_short_message_not_annotated(self):
        _, _, rows = run_game(text="short")
        message = public_messages(rows)[0]
        self.assertNotIn("truncated_from", message["payload"])

    def test_memory_render_capped_but_storage_intact(self):
        from werewolf.agents.ai_agent import AIAgent
        agent = AIAgent(
            player_id=0, role="villager", team="village",
            provider=FakeProvider(default=success_result(talky_response())),
        )
        agent.memory = {"notes": "y" * (MEMORY_MAX_CHARS * 2)}
        prompt = agent._build_user_prompt(
            {
                "required_action": "speak_public",
                "self": {"id": 0, "role": "villager", "team": "village"},
                "round": 1, "phase": "day_discuss",
                "alive_players": [{"id": 0}, {"id": 1}],
                "recent_events": [], "private_info": {},
            },
            errors=[],
        )
        self.assertIn("[memory truncated", prompt)
        memory_section = prompt.split("=== YOUR MEMORY ===")[1].split("===")[0]
        self.assertLess(len(memory_section), MEMORY_MAX_CHARS + 200)
        # stored memory untouched
        self.assertEqual(len(agent.memory["notes"]), MEMORY_MAX_CHARS * 2)
        # limits are stated to the model
        self.assertIn("LIMITS:", prompt)


class GenerationConfigTests(unittest.TestCase):
    def test_requests_carry_generation_and_records_log_it(self):
        config = GenerationConfig(
            temperature=0.0, top_p=1.0, max_output_tokens=1024,
            reasoning_effort="low", provider_seed=7,
        )
        _, provider, rows = run_game(generation_config=config)
        for request in provider.requests:
            self.assertEqual(request.generation, config)
        llm_calls = [r for r in rows if r["type"] == "llm_call"
                     and r["api_attempted"]]
        for call in llm_calls:
            self.assertEqual(call["requested_generation"]["temperature"], 0.0)
            self.assertEqual(call["requested_generation"]["provider_seed"], 7)
            self.assertEqual(call["schema_version"], 2)
        config_row = next(r for r in rows if r["type"] == "config")
        self.assertEqual(config_row["generation_config"]["temperature"], 0.0)
        self.assertEqual(config_row["discussion_cycles"], 2)
        self.assertEqual(
            config_row["limits"]["public_message_max_chars"],
            PUBLIC_MESSAGE_MAX_CHARS,
        )

    def test_xai_kwargs_mapping(self):
        request = ModelRequest(
            model="grok-4.3", system_prompt="s", user_prompt="u",
            generation=GenerationConfig(
                temperature=0.0, top_p=0.9, max_output_tokens=512,
                reasoning_effort="none", provider_seed=42,
            ),
        )
        kwargs = build_chat_kwargs(request)
        self.assertEqual(kwargs, {
            "model": "grok-4.3", "reasoning_effort": "none",
            "temperature": 0.0, "top_p": 0.9, "max_tokens": 512, "seed": 42,
        })
        # unset fields are not sent at all (provider defaults, logged as None)
        bare = build_chat_kwargs(ModelRequest(
            model="grok-4.3", system_prompt="s", user_prompt="u",
        ))
        self.assertEqual(bare, {"model": "grok-4.3"})

    def test_invalid_param_value_is_identified_for_dropping(self):
        # xai-sdk 1.17.0 raises ValueError('Invalid reasoning effort: none.
        # Must be one of: (\'low\', \'high\')') - the provider must identify
        # reasoning_effort as the offender so it can drop + report it
        # instead of failing every call.
        from werewolf.llm.xai_provider import _unexpected_kwarg_name
        exc = ValueError(
            "Invalid reasoning effort: none. Must be one of: ('low', 'high')"
        )
        kwargs = {"model": "grok-4.3", "reasoning_effort": "none",
                  "temperature": 0.0}
        self.assertEqual(_unexpected_kwarg_name(exc, kwargs), "reasoning_effort")
        # unidentifiable errors are not blamed on a param
        self.assertIsNone(_unexpected_kwarg_name(
            ValueError("something else entirely"), {"temperature": 0.0}
        ))

    def test_litellm_kwargs_mapping(self):
        try:
            from werewolf.llm.litellm_provider import build_completion_kwargs
        except ImportError:
            self.skipTest("litellm not installed")
        request = ModelRequest(
            model="gemini/gemini-3.5-flash", system_prompt="s", user_prompt="u",
            generation=GenerationConfig(
                temperature=0.0, max_output_tokens=512, structured_output=True,
            ),
        )
        kwargs = build_completion_kwargs(request, api_key="k", timeout=10)
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertEqual(kwargs["max_tokens"], 512)
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertNotIn("top_p", kwargs)


if __name__ == "__main__":
    unittest.main()
