"""Minimal execution-side completion verifier for crash reconciliation.

When a session crashes between a game finishing and its terminal
journal record being appended, resume must decide whether the paid game
is recoverable WITHOUT importing the PR 2 report builder: analysis code
evolves freely and must never gate paid-execution decisions. This
module is deliberately tiny, self-contained, and covered by
`execution_runtime_hash`.

Recoverable completion requires all of:

1. The canonical game JSONL parses sufficiently for terminal
   verification (config + terminal records).
2. The config's game ID matches the journaled trial_started game ID.
3. Exactly one usable terminal outcome record exists.
4. The winner and terminal state satisfy the configured victory
   predicate.
5. The terminal usage_summary close record exists (required by the
   log schema this engine writes).

PR 2 may later classify a recovered game dirty or analytically
ineligible; that never changes the paid attempt's completion state.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from werewolf.experiments.canonical import sha256_bytes

VERIFIER_VERSION = 1

_WINNERS = ("wolf", "village")


def _parse_rows(data: bytes) -> tuple:
    """Parse complete JSON lines; unparseable lines are counted, not
    fatal — sufficiency is judged by what the checks need."""
    rows, unparseable = [], 0
    for line in data.split(b"\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            unparseable += 1
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            unparseable += 1
    return rows, unparseable


def _usable_outcome(row: dict) -> bool:
    return (
        row.get("winner") in _WINNERS
        and isinstance(row.get("rounds"), int)
        and isinstance(row.get("remaining"), list)
        and all(isinstance(pid, int) for pid in row["remaining"])
    )


def _usable_abort(row: dict) -> bool:
    return (
        row.get("abort_schema_version") == 1
        and isinstance(row.get("reason"), str)
        and bool(row["reason"].strip())
        and isinstance(row.get("round"), int)
        and isinstance(row.get("phase"), str)
    )


def _victory_predicate_holds(
    winner: str, remaining: list, role_map: dict
) -> Optional[str]:
    """Re-evaluate the win condition from the terminal state. Returns a
    reason string when the predicate does NOT hold."""
    if not isinstance(role_map, dict) or not role_map:
        return "config carries no role_map to evaluate the victory predicate"
    roles = {}
    for pid_text, entry in role_map.items():
        try:
            pid = int(pid_text)
        except (TypeError, ValueError):
            return f"role_map has a non-integer player id: {pid_text!r}"
        if not isinstance(entry, dict) or "role" not in entry:
            return f"role_map entry for player {pid_text} is malformed"
        roles[pid] = entry["role"]
    unknown = [pid for pid in remaining if pid not in roles]
    if unknown:
        return f"terminal survivors {unknown} are not in the role_map"
    wolves = sum(1 for pid in remaining if roles[pid] == "werewolf")
    villagers = len(remaining) - wolves
    if winner == "village":
        if wolves != 0:
            return (
                f"winner is village but {wolves} wolves remain alive"
            )
    else:
        if wolves == 0 or wolves < villagers:
            return (
                f"winner is wolf but terminal state has {wolves} wolves "
                f"vs {villagers} village-team survivors"
            )
    return None


def verify_terminal_completion(
    data: bytes, *, expected_game_id: str, game_rules: Optional[dict] = None
) -> dict:
    """Run the recoverable-completion checks over raw game log bytes."""
    rows, unparseable = _parse_rows(data)
    checks: dict = {}
    reasons = []

    configs = [r for r in rows if r.get("type") == "config"]
    config = configs[0] if configs else {}
    checks["config_present"] = len(configs) == 1
    if len(configs) != 1:
        reasons.append(
            f"expected exactly one config record, found {len(configs)}"
        )

    checks["game_id_match"] = config.get("game_id") == expected_game_id
    if not checks["game_id_match"]:
        reasons.append(
            f"game_id mismatch: log has {config.get('game_id')!r}, "
            f"journal expected {expected_game_id!r}"
        )

    if game_rules and config:
        mismatched = [
            key for key in ("n_players", "n_wolves", "n_seers")
            if key in game_rules and config.get(key) != game_rules[key]
        ]
        checks["game_rules_match"] = not mismatched
        if mismatched:
            reasons.append(
                "game rules differ from the execution contract: "
                + ", ".join(
                    f"{key}={config.get(key)!r} (expected "
                    f"{game_rules[key]!r})" for key in mismatched
                )
            )
    else:
        checks["game_rules_match"] = game_rules is None or bool(config)

    outcomes = [r for r in rows if r.get("type") == "outcome"]
    usable = [r for r in outcomes if _usable_outcome(r)]
    checks["single_usable_outcome"] = len(usable) == 1 and len(outcomes) == 1
    if not checks["single_usable_outcome"]:
        reasons.append(
            f"expected exactly one usable terminal outcome, found "
            f"{len(usable)} usable of {len(outcomes)} outcome records"
        )
    outcome = usable[0] if len(usable) == 1 else None

    if outcome is not None and checks["config_present"]:
        predicate_failure = _victory_predicate_holds(
            outcome["winner"], outcome["remaining"],
            config.get("role_map"),
        )
        checks["victory_predicate"] = predicate_failure is None
        if predicate_failure:
            reasons.append(predicate_failure)
    else:
        checks["victory_predicate"] = False
        if outcome is None:
            pass  # already reported above
        else:
            reasons.append("victory predicate cannot be evaluated "
                           "without a config record")

    checks["usage_summary_present"] = any(
        r.get("type") == "usage_summary" for r in rows
    )
    if not checks["usage_summary_present"]:
        reasons.append("terminal usage_summary close record is missing")

    aborts = [r for r in rows if r.get("type") == "abort"]
    usable_aborts = [r for r in aborts if _usable_abort(r)]
    common_terminal_checks = (
        checks["config_present"]
        and checks["game_id_match"]
        and checks["game_rules_match"]
        and checks["usage_summary_present"]
    )
    terminal_abort = None
    if len(aborts) == 1 and len(usable_aborts) == 1 \
            and common_terminal_checks and not outcomes:
        abort = usable_aborts[0]
        terminal_abort = {
            "reason": abort["reason"],
            "classification": (
                "interrupted"
                if abort["reason"] == "operator_interrupt"
                else "failed"
            ),
        }

    complete = all(checks.values())
    return {
        "verifier_version": VERIFIER_VERSION,
        "complete": complete,
        "checks": checks,
        "reasons": reasons,
        "unparseable_lines": unparseable,
        "terminal_abort": terminal_abort,
        "outcome": (
            {
                "winner": outcome["winner"],
                "rounds": outcome["rounds"],
                "remaining": outcome["remaining"],
            }
            if complete and outcome is not None else None
        ),
    }


def reconcile_attempt_source(
    game_log_path, *, expected_game_id: str,
    game_rules: Optional[dict] = None,
) -> dict:
    """Single-read source capture for crash reconciliation: hash and
    verify the SAME bytes, so the recorded hash always describes exactly
    what was verified."""
    path = Path(game_log_path)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return {
            "source_status": "missing_game_log",
            "recorded_game_sha256": None,
            "verification": None,
        }
    return {
        "source_status": "recorded",
        "recorded_game_sha256": sha256_bytes(data),
        "verification": verify_terminal_completion(
            data, expected_game_id=expected_game_id, game_rules=game_rules,
        ),
    }
