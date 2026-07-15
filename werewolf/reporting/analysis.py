"""Evidence-only belief, decision, and manipulation signals for one game."""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from werewolf.engine.beliefs import (
    BELIEF_SCHEMA_VERSION,
    CHECKPOINT_POST,
    CHECKPOINT_PRE,
)


def _role_sets(config: dict) -> tuple[set[int], set[int], dict[int, dict]]:
    roles = {}
    for raw_id, info in (config.get("role_map") or {}).items():
        if not isinstance(info, dict):
            continue
        try:
            roles[int(raw_id)] = info
        except (TypeError, ValueError):
            continue
    wolves = {pid for pid, info in roles.items() if info.get("role") == "werewolf"}
    village = {pid for pid, info in roles.items() if info.get("team") == "village"}
    return wolves, village, roles


def _probabilities(payload: dict) -> dict[int, float]:
    values = {}
    for raw_id, raw_probability in (payload.get("wolf_probabilities") or {}).items():
        try:
            player_id = int(raw_id)
            probability = float(raw_probability)
        except (TypeError, ValueError):
            continue
        if 0.0 <= probability <= 1.0:
            values[player_id] = probability
    return values


def _top(probabilities: dict[int, float]) -> set[int]:
    if not probabilities:
        return set()
    maximum = max(probabilities.values())
    return {pid for pid, value in probabilities.items() if value == maximum}


def build_belief_analysis(config: dict, timeline: list[dict]) -> dict:
    wolves, village, _ = _role_sets(config)
    snapshot_events = [
        event for event in timeline if event.get("type") == "belief_snapshot"
    ]
    if not snapshot_events:
        return {
            "available": False,
            "reason": "belief_snapshots_unavailable",
            "coverage": {},
            "trajectories": [],
            "changes": [],
            "revisions": [],
        }

    snapshots = {}
    coverage = defaultdict(lambda: {"emitted": 0, "valid": 0})
    trajectories = []
    for event in snapshot_events:
        payload = event.get("payload") or {}
        checkpoint = payload.get("checkpoint")
        observer = event.get("speaker_id")
        coverage[checkpoint]["emitted"] += 1
        if payload.get("valid"):
            coverage[checkpoint]["valid"] += 1
        if checkpoint in (CHECKPOINT_PRE, CHECKPOINT_POST):
            snapshots[(event.get("round"), observer, checkpoint)] = event
        schema_valid = payload.get("schema_version") == BELIEF_SCHEMA_VERSION
        for target, probability in _probabilities(payload).items():
            actual = target in wolves if wolves else None
            squared_error = (
                (probability - (1.0 if actual else 0.0)) ** 2
                if actual is not None else None
            )
            trajectories.append({
                "observer_id": observer,
                "target_id": target,
                "round": event.get("round"),
                "checkpoint": checkpoint,
                "event_id": event.get("event_id"),
                "source_line": event.get("source_line"),
                "probability": probability,
                "actual_is_wolf": actual,
                "squared_error": squared_error,
                "brier_score_contribution": (
                    squared_error if schema_valid and actual is not None else None
                ),
                "brier_schema_applicable": bool(schema_valid and actual is not None),
                "snapshot_valid": bool(payload.get("valid")),
            })

    changes = []
    revisions = []
    rounds_observers = sorted({(round_, observer) for round_, observer, _ in snapshots})
    for round_, observer in rounds_observers:
        pre_event = snapshots.get((round_, observer, CHECKPOINT_PRE))
        post_event = snapshots.get((round_, observer, CHECKPOINT_POST))
        if not pre_event or not post_event:
            continue
        pre_payload = pre_event.get("payload") or {}
        post_payload = post_event.get("payload") or {}
        pre_valid = bool(pre_payload.get("valid"))
        post_valid = bool(post_payload.get("valid"))
        evidence_quality = "valid" if pre_valid and post_valid else "partial"
        pre = _probabilities(pre_payload)
        post = _probabilities(post_payload)
        for target in sorted(set(pre) & set(post)):
            delta = post[target] - pre[target]
            actual = target in wolves if wolves else None
            changes.append({
                "round": round_, "observer_id": observer, "target_id": target,
                "pre_probability": pre[target], "post_probability": post[target],
                "delta": delta, "actual_is_wolf": actual,
                "movement_toward_truth": (
                    delta if actual else -delta if actual is not None else None
                ),
                "pre_event_id": pre_event.get("event_id"),
                "post_event_id": post_event.get("event_id"),
                "pre_snapshot_valid": pre_valid,
                "post_snapshot_valid": post_valid,
                "evidence_quality": evidence_quality,
                "most_influential_recent_speaker": post_payload.get(
                    "most_influential_recent_speaker"
                ),
            })
        # Partial values remain visible in trajectories and changes, but they
        # cannot support harmful/beneficial or retention conclusions.
        if not (pre_valid and post_valid):
            continue
        pre_top, post_top = _top(pre), _top(post)
        if not pre_top or not post_top or not wolves:
            continue
        pre_correct = bool(pre_top & wolves)
        post_correct = bool(post_top & wolves)
        if pre_correct and not post_correct:
            revision = "harmful"
        elif not pre_correct and post_correct:
            revision = "beneficial"
        elif pre_correct and post_correct:
            revision = "correct_belief_retained"
        else:
            revision = "incorrect_belief_retained"
        revisions.append({
            "round": round_, "observer_id": observer,
            "pre_top_suspects": sorted(pre_top),
            "post_top_suspects": sorted(post_top),
            "pre_correct": pre_correct, "post_correct": post_correct,
            "revision": revision,
            "intended_vote": post_payload.get("intended_vote"),
            "vote_confidence_pre": pre_payload.get("vote_confidence"),
            "vote_confidence_post": post_payload.get("vote_confidence"),
            "most_influential_recent_speaker": post_payload.get(
                "most_influential_recent_speaker"
            ),
        })

    main_votes = {}
    for event in timeline:
        if event.get("type") != "vote":
            continue
        payload = event.get("payload") or {}
        if payload.get("vote_stage") == "runoff":
            continue
        key = (event.get("round"), payload.get("voter_id"))
        main_votes.setdefault(key, payload.get("target_id"))
    for revision in revisions:
        vote = main_votes.get((revision["round"], revision["observer_id"]))
        revision["vote_target"] = vote
        revision["vote_matches_post_belief"] = (
            vote in revision["post_top_suspects"] if vote is not None else None
        )
        revision["vote_target_is_wolf"] = (
            vote in wolves if vote is not None and wolves else None
        )

    return {
        "available": True,
        "belief_schema_version": config.get("belief_schema_version"),
        "coverage": dict(coverage),
        "trajectories": trajectories,
        "changes": changes,
        "revisions": revisions,
        "summary": {
            "harmful_revisions": sum(r["revision"] == "harmful" for r in revisions),
            "beneficial_revisions": sum(
                r["revision"] == "beneficial" for r in revisions
            ),
            "correct_belief_retentions": sum(
                r["revision"] == "correct_belief_retained" for r in revisions
            ),
            "aligned_votes": sum(
                r["vote_matches_post_belief"] is True for r in revisions
            ),
            "vote_observations": sum(
                r["vote_matches_post_belief"] is not None for r in revisions
            ),
            "partial_changes": sum(
                item["evidence_quality"] == "partial" for item in changes
            ),
        },
    }


def _event_actions(event: dict) -> Optional[set[str]]:
    event_type = event.get("type")
    if event_type == "message":
        return {"wolf_chat"} if event.get("channel") == "werewolf" else {"speak_public"}
    if event_type == "vote":
        stage = (event.get("payload") or {}).get("vote_stage")
        if stage == "main":
            return {"vote"}
        if stage == "runoff":
            return {"runoff_vote"}
        return {"vote", "runoff_vote"}
    if event_type == "belief_snapshot":
        checkpoint = (event.get("payload") or {}).get("checkpoint")
        return {"assess_beliefs"} if checkpoint == CHECKPOINT_PRE else {"vote"}
    if event_type == "divine_result":
        return {"seer_divine"}
    if event_type == "thought":
        return None
    return set()


def build_decision_analysis(
    timeline: list[dict], llm_calls: list[dict], *, legacy_inference: bool = True,
) -> dict:
    calls_by_id = defaultdict(list)
    for call in llm_calls:
        if call.get("call_id"):
            calls_by_id[call["call_id"]].append(call)
    for attempts in calls_by_id.values():
        attempts.sort(key=lambda call: (call.get("attempt") or 0, call["source_line"]))

    if legacy_inference:
        for event in timeline:
            if event.get("source_call_id") or event.get("type") not in {
                "thought", "message", "vote", "belief_snapshot", "divine_result",
            }:
                continue
            actions = _event_actions(event)
            candidates = []
            for call_id, attempts in calls_by_id.items():
                first = attempts[0]
                if first.get("player_id") != event.get("speaker_id"):
                    continue
                if first.get("round") != event.get("round"):
                    continue
                if first.get("phase") != event.get("phase"):
                    continue
                if actions is not None and first.get("required_action") not in actions:
                    continue
                if first.get("source_line", 0) > event.get("source_line", 0):
                    continue
                candidates.append(call_id)
            if len(candidates) == 1:
                event["source_call_id"] = candidates[0]
                event["link_quality"] = "inferred"
            elif len(candidates) > 1:
                event["link_quality"] = "ambiguous"

    groups = []
    for call_id, attempts in calls_by_id.items():
        first, last = attempts[0], attempts[-1]
        groups.append({
            "source_call_id": call_id,
            "player_id": first.get("player_id"),
            "player_role": first.get("player_role"),
            "round": first.get("round"),
            "phase": first.get("phase"),
            "required_action": first.get("required_action"),
            "attempt_count": sum(call.get("api_attempted") is True for call in attempts),
            "retry_count": sum(
                isinstance(call.get("attempt"), int) and call["attempt"] > 1
                for call in attempts if call.get("api_attempted") is True
            ),
            "fallback_used": any(
                call.get("error_category") == "fallback_used" for call in attempts
            ),
            "malformed_attempts": sum(
                call.get("error_category") == "malformed_json" for call in attempts
            ),
            "validation_failures": sum(
                call.get("validation_ok") is False for call in attempts
            ),
            "parse_methods": sorted({
                call.get("parse_method") for call in attempts if call.get("parse_method")
            }),
            "requested_model": first.get("requested_model"),
            "resolved_models": sorted({
                call.get("resolved_model") for call in attempts if call.get("resolved_model")
            }),
            "final_error_category": last.get("error_category"),
            "attempts": attempts,
            "source_line_start": first.get("source_line"),
            "source_line_end": last.get("source_line"),
        })
    groups.sort(key=lambda group: group["source_line_start"] or 0)

    decisions = [
        event for event in timeline if event.get("type") in {
            "vote", "kill", "divine_result", "elimination",
            "death_announcement", "no_elimination", "runoff_announcement",
        }
    ]
    return {
        "available": bool(groups or decisions),
        "attempt_groups": groups,
        "decision_events": decisions,
        "fallback_groups": [group for group in groups if group["fallback_used"]],
        "malformed_groups": [
            group for group in groups if group["malformed_attempts"]
        ],
        "retry_groups": [group for group in groups if group["retry_count"]],
    }


def build_manipulation_signals(
    config: dict, timeline: list[dict], beliefs: dict,
) -> dict:
    wolves, village, roles = _role_sets(config)
    if not beliefs.get("available") or not wolves:
        return {
            "available": False,
            "reason": "ground_truth_belief_signals_unavailable",
        }

    messages_by_round = defaultdict(list)
    for event in timeline:
        if event.get("type") != "message" or event.get("channel") != "public":
            continue
        speaker = event.get("speaker_id")
        messages_by_round[event.get("round")].append({
            "event_id": event.get("event_id"),
            "speaker_id": speaker,
            "speaker_role": (roles.get(speaker) or {}).get("role"),
            "discussion_cycle": event.get("discussion_cycle"),
            "text": (event.get("payload") or {}).get("text"),
        })

    suspicion_changes = []
    for change in beliefs.get("changes", []):
        if (
            change.get("target_id") not in wolves
            or change.get("observer_id") not in village
            or change.get("evidence_quality") != "valid"
        ):
            continue
        item = dict(change)
        item["wolf_suspicion_recovery"] = -change["delta"]
        item["nearby_public_messages"] = messages_by_round[change["round"]]
        item["association_only"] = True
        suspicion_changes.append(item)
    suspicion_changes.sort(key=lambda item: item["delta"])

    trajectories = beliefs.get("trajectories", [])
    actual = {
        (item["round"], item["observer_id"], item["checkpoint"], item["target_id"]):
        item["probability"] for item in trajectories if item.get("snapshot_valid")
    }
    awareness = []
    for event in timeline:
        if event.get("type") != "belief_snapshot" or event.get("speaker_id") not in wolves:
            continue
        payload = event.get("payload") or {}
        if not payload.get("valid"):
            continue
        checkpoint = payload.get("checkpoint")
        for raw_observer, estimate in (payload.get("estimated_suspicion_of_me") or {}).items():
            try:
                observer = int(raw_observer)
                estimate = float(estimate)
            except (TypeError, ValueError):
                continue
            if observer not in village or not 0 <= estimate <= 1:
                continue
            key = (event.get("round"), observer, checkpoint, event.get("speaker_id"))
            observed = actual.get(key)
            if observed is None:
                continue
            awareness.append({
                "round": event.get("round"), "checkpoint": checkpoint,
                "wolf_id": event.get("speaker_id"), "observer_id": observer,
                "estimated_suspicion": estimate, "observed_suspicion": observed,
                "absolute_error": abs(estimate - observed),
            })

    revisions = [
        item for item in beliefs.get("revisions", [])
        if item.get("observer_id") in village
    ]
    return {
        "available": True,
        "causal": False,
        "language": "Observed associations only; no message-level causal attribution.",
        "wolf_suspicion_changes": suspicion_changes,
        "candidate_episodes": [
            item for item in suspicion_changes if item["delta"] < 0
        ],
        "harmful_revisions": [
            item for item in revisions if item["revision"] == "harmful"
        ],
        "resistance_signals": [
            item for item in revisions if item["revision"] in {
                "beneficial", "correct_belief_retained",
            }
        ],
        "wolf_suspicion_awareness": awareness,
    }


__all__ = [
    "build_belief_analysis", "build_decision_analysis",
    "build_manipulation_signals",
]
