"""One-call live smoke test for any registered model alias.

Makes exactly ONE tiny API request (cheapest possible; free-tier for
Gemini) and prints the full accounting record, proving the provider,
key, usage capture, and cost labeling all work.

Usage (from repo root):
    python scripts/smoke_test_model.py gemini_flash_lite
    python scripts/smoke_test_model.py fast          # xAI grok-4.3
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from werewolf.cli.run_game import load_env_file
from werewolf.llm.provider import GenerationConfig, ModelRequest
from werewolf.llm.registry import build_provider, get_api_key, resolve


def main():
    alias = sys.argv[1] if len(sys.argv) > 1 else "gemini_flash_lite"
    load_env_file()

    spec = resolve(alias)
    print(f"alias:     {alias}")
    print(f"provider:  {spec.provider}")
    print(f"model:     {spec.model}")

    if not get_api_key(spec):
        print(f"FAIL: none of {spec.api_key_env} set in environment/.env")
        sys.exit(1)

    provider = build_provider(spec)
    if provider is None:
        print("FAIL: provider could not be built (SDK missing? "
              "run: pip install -r requirements.txt)")
        sys.exit(1)

    print("calling model (1 tiny request)...")
    result = provider.complete(ModelRequest(
        model=spec.model,
        system_prompt="Respond with JSON only.",
        user_prompt='Respond with exactly this JSON: {"hello": "werewolf"}',
        generation=GenerationConfig(reasoning_effort=spec.reasoning_effort),
    ))

    print(f"\nok:              {result.ok}")
    print(f"response text:   {result.text!r}")
    print(f"resolved_model:  {result.resolved_model}")
    print(f"finish_reason:   {result.finish_reason}")
    print(f"latency_ms:      {result.latency_ms}")
    print(f"tokens:          {json.dumps(result.usage.to_json_dict())}")
    print(f"cost:            {json.dumps(result.cost.to_json_dict())}")
    if not result.ok:
        print(f"error:           {result.error_category} {result.error_message}")
        sys.exit(1)
    print("\nSUCCESS - provider, key, usage capture, and cost labeling all work.")


if __name__ == "__main__":
    main()
