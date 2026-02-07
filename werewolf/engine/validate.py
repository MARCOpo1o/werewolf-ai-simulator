from typing import Tuple, Optional


def _to_int(value) -> int:
    """Convert string or int to int. AI sometimes returns strings."""
    if isinstance(value, str):
        return int(value)
    return value


def validate_action(
    observation: dict,
    response: dict,
    game_state
) -> Tuple[bool, Optional[str]]:
    required_action = observation["required_action"]
    action = response.get("action")
    say = response.get("say")

    player_id = observation["self"]["id"]
    player_role = observation["self"]["role"]
    alive_ids = {p["id"] for p in observation["alive_players"]}

    if say:
        valid_say, error = validate_say(say, player_role, required_action)
        if not valid_say:
            return False, error

    if required_action == "wolf_chat":
        return True, None

    if required_action == "speak_public":
        return True, None

    if required_action == "choose_wolf_kill":
        if not action or "kill_target" not in action:
            return False, "Missing kill_target in action"
        target = _to_int(action["kill_target"])
        if target not in alive_ids:
            return False, f"Kill target {target} is not alive"
        wolf_ids = set(observation["private_info"].get("wolf_roster", []))
        if target in wolf_ids:
            return False, f"Cannot kill fellow wolf {target}"
        return True, None

    if required_action == "seer_divine":
        if not action or "divine_target" not in action:
            return False, "Missing divine_target in action"
        target = _to_int(action["divine_target"])
        if target not in alive_ids:
            return False, f"Divine target {target} is not alive"
        if target == player_id:
            return False, "Cannot divine yourself"
        return True, None

    if required_action == "vote":
        if not action or "vote_target" not in action:
            return False, "Missing vote_target in action"
        target = _to_int(action["vote_target"])
        if target not in alive_ids:
            return False, f"Vote target {target} is not alive"
        if target == player_id:
            return False, "Cannot vote for yourself"
        return True, None

    if required_action == "runoff_vote":
        if not action or "vote_target" not in action:
            return False, "Missing vote_target in action"
        target = _to_int(action["vote_target"])
        tc = observation.get("turn_context") or {}
        runoff_candidates = set(tc.get("runoff_candidates", []))
        if target not in runoff_candidates:
            return False, f"Vote target {target} is not a runoff candidate. Valid: {sorted(runoff_candidates)}"
        if target == player_id:
            return False, "Cannot vote for yourself"
        return True, None

    return False, f"Unknown required_action: {required_action}"


def validate_say(
    say: dict,
    player_role: str,
    required_action: str
) -> Tuple[bool, Optional[str]]:
    allowed_channels = {"public"}
    if player_role == "werewolf":
        allowed_channels.add("werewolf")

    for channel in say.keys():
        if channel not in allowed_channels:
            return False, f"Player with role {player_role} cannot send to channel {channel}"

    if required_action == "wolf_chat":
        if "public" in say:
            return False, "Cannot send public messages during wolf chat"

    return True, None


def get_fallback_action(observation: dict, rng) -> dict:
    required_action = observation["required_action"]
    player_id = observation["self"]["id"]
    alive_ids = [p["id"] for p in observation["alive_players"]]

    if required_action == "wolf_chat":
        return {"thought": "[fallback] No comment", "say": None, "action": None}

    if required_action == "speak_public":
        return {"thought": "[fallback] Staying quiet", "say": {"public": "..."}, "action": None}

    if required_action == "choose_wolf_kill":
        wolf_ids = set(observation["private_info"].get("wolf_roster", []))
        valid_targets = [pid for pid in alive_ids if pid not in wolf_ids]
        if valid_targets:
            target = rng.choice(valid_targets)
        else:
            target = alive_ids[0] if alive_ids else 0
        return {
            "thought": "[fallback] Random kill target",
            "say": None,
            "action": {"kill_target": target}
        }

    if required_action == "seer_divine":
        valid_targets = [pid for pid in alive_ids if pid != player_id]
        if valid_targets:
            target = rng.choice(valid_targets)
        else:
            target = alive_ids[0] if alive_ids else 0
        return {
            "thought": "[fallback] Random divine target",
            "say": None,
            "action": {"divine_target": target}
        }

    if required_action == "vote":
        valid_targets = [pid for pid in alive_ids if pid != player_id]
        if valid_targets:
            target = rng.choice(valid_targets)
        else:
            target = alive_ids[0] if alive_ids else 0
        return {
            "thought": "[fallback] Random vote target",
            "say": None,
            "action": {"vote_target": target}
        }

    if required_action == "runoff_vote":
        tc = observation.get("turn_context") or {}
        runoff_candidates = tc.get("runoff_candidates", [])
        valid_targets = [pid for pid in runoff_candidates if pid != player_id]
        if valid_targets:
            target = rng.choice(valid_targets)
        else:
            target = runoff_candidates[0] if runoff_candidates else 0
        return {
            "thought": "[fallback] Random runoff vote target",
            "say": None,
            "action": {"vote_target": target}
        }

    return {"thought": "[fallback] Unknown action", "say": None, "action": None}
