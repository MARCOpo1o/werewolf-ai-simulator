"""Structured belief snapshots (Research Instrumentation V1).

Players privately report wolf probabilities, voting intent, and (wolves
only) second-order beliefs about how suspicious others are of them.
Snapshots are collected twice per day (pre-discussion via the
`assess_beliefs` action; post-discussion inside the `vote` response) and
logged as moderator-only `belief_snapshot` events with a fixed schema.

Accounting rules mirror the cost ledger philosophy:
- fallbacks/missing data never fabricate probabilities;
- partial data is kept with per-field nulls and an explicit
  invalid_reason, never silently dropped or zero-filled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

BELIEF_SCHEMA_VERSION = 1

CHECKPOINT_PRE = "pre_discussion"
CHECKPOINT_POST = "post_discussion"


def _coerce_id(value) -> Optional[int]:
    from werewolf.engine.validate import _to_int  # lazy: avoid import cycle
    try:
        return _to_int(value)
    except (TypeError, ValueError):
        return None


def coerce_probability(value) -> Optional[float]:
    """Accept 0.65, 1, "0.65"; clamp tiny float drift; reject everything
    else (including percentages like 65 - ambiguity is worse than a null)."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        f = float(value)
    elif isinstance(value, str):
        try:
            f = float(value.strip())
        except (ValueError, AttributeError):
            return None
    else:
        return None
    if not math.isfinite(f):
        return None
    if -0.001 <= f <= 1.001:
        return min(1.0, max(0.0, f))
    return None


def coerce_prob_map(raw, expected_ids: set[int]) -> tuple[dict[int, float], list[str]]:
    """Parse a {player_id: probability} mapping. Returns (parsed, problems).
    Keys tolerate "P2"/"2"/2; unexpected or unparseable entries are
    reported, not guessed."""
    problems: list[str] = []
    out: dict[int, float] = {}
    if not isinstance(raw, dict):
        return out, ["expected an object mapping player ids to probabilities"]
    for key, value in raw.items():
        pid = _coerce_id(key)
        if pid is None:
            problems.append(f"unparseable player key {key!r}")
            continue
        if pid not in expected_ids:
            problems.append(f"unexpected player P{pid}")
            continue
        prob = coerce_probability(value)
        if prob is None:
            problems.append(f"invalid probability for P{pid}: {value!r}")
            continue
        out[pid] = prob
    missing = sorted(expected_ids - set(out))
    if missing:
        problems.append(
            "missing probabilities for: " + ", ".join(f"P{m}" for m in missing)
        )
    return out, problems


@dataclass
class BeliefSnapshot:
    checkpoint: str
    player_id: int
    wolf_probabilities: dict[int, float] = field(default_factory=dict)
    intended_vote: Optional[int] = None
    vote_confidence: Optional[float] = None
    most_influential_recent_speaker: Optional[int] = None
    estimated_suspicion_of_me: Optional[dict[int, float]] = None  # wolves
    valid: bool = True
    invalid_reason: Optional[str] = None  # missing | malformed | partial
    problems: list[str] = field(default_factory=list)

    def to_payload(self) -> dict:
        return {
            "schema_version": BELIEF_SCHEMA_VERSION,
            "checkpoint": self.checkpoint,
            "wolf_probabilities": {
                str(k): v for k, v in self.wolf_probabilities.items()
            },
            "intended_vote": self.intended_vote,
            "vote_confidence": self.vote_confidence,
            "most_influential_recent_speaker": self.most_influential_recent_speaker,
            "estimated_suspicion_of_me": (
                {str(k): v for k, v in self.estimated_suspicion_of_me.items()}
                if self.estimated_suspicion_of_me is not None else None
            ),
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "problems": self.problems,
        }


def parse_belief_snapshot(
    raw,
    checkpoint: str,
    self_id: int,
    alive_ids: list[int],
    is_wolf: bool,
) -> BeliefSnapshot:
    """Parse a model-supplied `beliefs` object. Never raises: returns an
    invalid/partial snapshot with explicit reasons instead."""
    others = set(alive_ids) - {self_id}
    snapshot = BeliefSnapshot(checkpoint=checkpoint, player_id=self_id)

    if raw is None:
        snapshot.valid = False
        snapshot.invalid_reason = "missing"
        return snapshot
    if not isinstance(raw, dict):
        snapshot.valid = False
        snapshot.invalid_reason = "malformed"
        snapshot.problems = ["beliefs is not an object"]
        return snapshot

    probs, problems = coerce_prob_map(raw.get("wolf_probabilities"), others)
    snapshot.wolf_probabilities = probs
    snapshot.problems.extend(problems)

    intended = raw.get("intended_vote")
    if intended is not None:
        pid = _coerce_id(intended)
        if pid in others:
            snapshot.intended_vote = pid
        else:
            snapshot.problems.append(f"invalid intended_vote: {intended!r}")

    confidence = raw.get("vote_confidence")
    if confidence is not None:
        prob = coerce_probability(confidence)
        if prob is not None:
            snapshot.vote_confidence = prob
        else:
            snapshot.problems.append(f"invalid vote_confidence: {confidence!r}")

    speaker = raw.get("most_influential_recent_speaker")
    if speaker is not None:
        pid = _coerce_id(speaker)
        if pid is not None and pid in set(alive_ids):
            snapshot.most_influential_recent_speaker = pid
        else:
            snapshot.problems.append(
                f"invalid most_influential_recent_speaker: {speaker!r}"
            )

    if is_wolf:
        suspicion, sproblems = coerce_prob_map(
            raw.get("estimated_suspicion_of_me"), others
        )
        snapshot.estimated_suspicion_of_me = suspicion
        snapshot.problems.extend(
            f"estimated_suspicion_of_me: {p}" for p in sproblems
        )
        suspicion_complete = len(suspicion) == len(others)
    else:
        suspicion_complete = True

    probs_complete = len(probs) == len(others)
    snapshot.valid = probs_complete and suspicion_complete
    if not snapshot.valid:
        snapshot.invalid_reason = "partial" if (probs or not others) else "malformed"
    return snapshot


def validate_assess_beliefs(observation: dict, response: dict):
    """Strict validation for the dedicated assess_beliefs action (used by
    engine.validate). The vote-embedded snapshot is deliberately NOT
    validated strictly: a legal vote must never be rejected because a
    research field was malformed."""
    self_id = observation["self"]["id"]
    is_wolf = observation["self"]["role"] == "werewolf"
    alive_ids = [p["id"] for p in observation["alive_players"]]

    snapshot = parse_belief_snapshot(
        response.get("beliefs"), CHECKPOINT_PRE, self_id, alive_ids, is_wolf,
    )
    if snapshot.valid:
        return True, None

    detail = "; ".join(snapshot.problems[:4]) or snapshot.invalid_reason
    requirement = (
        'Provide "beliefs" with "wolf_probabilities" covering EVERY alive '
        "player except yourself, values between 0.0 and 1.0"
    )
    if is_wolf:
        requirement += (
            ', and "estimated_suspicion_of_me" covering the same players'
        )
    return False, f"Invalid beliefs ({detail}). {requirement}."
