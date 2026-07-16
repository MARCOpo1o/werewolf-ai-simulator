"""Versioned prompt profiles pinned into execution contracts.

A prompt profile names the exact prompt behavior a paid experiment ran
with. `baseline_v1` is the current engine prompt set, unchanged; its
identity is the prompt module's content hash plus SHA-256 hashes of
canonical renders of every template (system prompts per role, every
action instruction, and the limits notice). The runner refuses any
profile it does not recognize, so a manifest can never silently execute
with drifted prompts.
"""
from __future__ import annotations

import hashlib

from werewolf.agents.prompts import (
    ACTION_INSTRUCTIONS,
    get_action_instruction,
    get_limits_notice,
    get_prompt_version,
    get_system_prompt,
)

PROMPT_PROFILE_MECHANISM_VERSION = 1

# Fixed placeholder arguments so canonical renders are deterministic.
_CANONICAL_TURN_CONTEXT = {
    "speak_public": {
        "your_position": 1,
        "already_spoken": [],
        "yet_to_speak": [1, 2],
        "discussion_cycle": 1,
        "total_cycles": 2,
    },
    "runoff_vote": {"runoff_candidates": [1, 2]},
}


class PromptProfileError(ValueError):
    pass


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rendered_prompt_hashes() -> dict:
    """SHA-256 of a canonical render of every prompt surface."""
    hashes = {
        "system_werewolf": _sha256_text(
            get_system_prompt("werewolf", 0, wolf_roster=[0, 1])
        ),
        "system_seer": _sha256_text(get_system_prompt("seer", 0)),
        "system_villager": _sha256_text(get_system_prompt("villager", 0)),
        "limits_notice": _sha256_text(get_limits_notice()),
    }
    for action in sorted(ACTION_INSTRUCTIONS):
        hashes[f"action_{action}"] = _sha256_text(
            get_action_instruction(
                action, _CANONICAL_TURN_CONTEXT.get(action)
            )
        )
    return hashes


def resolve_prompt_profile(name: str) -> dict:
    """Return the manifest `prompt_profile` block for a known profile.

    baseline_v1 is exactly the current engine prompt behavior: it pins
    the prompt module content hash and canonical render hashes without
    altering anything the agents send.
    """
    if name != "baseline_v1":
        raise PromptProfileError(
            f"Unknown prompt profile: {name!r} (known: baseline_v1)"
        )
    return {
        "name": "baseline_v1",
        "mechanism_version": PROMPT_PROFILE_MECHANISM_VERSION,
        "prompt_source_version": get_prompt_version(),
        "rendered_prompt_hashes": rendered_prompt_hashes(),
    }


def verify_prompt_profile(pinned: dict) -> list:
    """Compare a manifest-pinned profile against current prompt code.
    Returns human-readable drift descriptions (empty when identical)."""
    if not isinstance(pinned, dict):
        return ["prompt_profile must be an object"]
    try:
        current = resolve_prompt_profile(pinned.get("name"))
    except PromptProfileError as exc:
        return [str(exc)]
    drift = []
    if pinned.get("prompt_source_version") != current["prompt_source_version"]:
        drift.append(
            "prompt source changed since the manifest was created "
            f"(pinned {pinned.get('prompt_source_version')}, "
            f"current {current['prompt_source_version']})"
        )
    pinned_hashes = pinned.get("rendered_prompt_hashes") or {}
    for key, value in current["rendered_prompt_hashes"].items():
        if pinned_hashes.get(key) != value:
            drift.append(f"rendered prompt changed: {key}")
    for key in set(pinned_hashes) - set(current["rendered_prompt_hashes"]):
        drift.append(f"pinned prompt surface no longer exists: {key}")
    return drift
