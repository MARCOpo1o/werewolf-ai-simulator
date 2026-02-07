from typing import Optional


CHANNEL_VISIBILITY = {
    "public": {"werewolf", "seer", "villager"},
    "werewolf": {"werewolf"},
    "seer_private": {"seer"},
    "moderator_only": set()
}


def can_see_channel(role: str, channel: str) -> bool:
    return role in CHANNEL_VISIBILITY.get(channel, set())


def filter_events_for_player(
    events: list[dict],
    player_role: str,
    since_idx: int = 0,
    max_events: int = 50
) -> list[dict]:
    visible = []
    for event in events[since_idx:]:
        channel = event.get("channel", "public")
        if can_see_channel(player_role, channel):
            visible.append(event)

    if len(visible) > max_events:
        visible = visible[-max_events:]

    return visible


def build_observation(
    game_state,
    player_id: int,
    required_action: str,
    turn_context: dict = None
) -> dict:
    player = game_state.players[player_id]

    recent_events = filter_events_for_player(
        game_state.events,
        player.role,
        since_idx=player.last_seen_event_idx
    )

    alive_players = [{"id": p.id} for p in game_state.get_alive_players()]

    private_info = {}
    if player.role == "werewolf":
        private_info["wolf_roster"] = game_state.get_wolf_ids()
    elif player.role == "seer":
        last_divine = get_last_divine_result(game_state.events, player_id)
        if last_divine:
            private_info["last_divine_result"] = last_divine

    observation = {
        "required_action": required_action,
        "self": player.to_self_dict(),
        "round": game_state.round,
        "phase": game_state.phase,
        "alive_players": alive_players,
        "recent_events": recent_events,
        "private_info": private_info
    }

    if turn_context is not None:
        observation["turn_context"] = turn_context

    return observation


def get_last_divine_result(events: list[dict], seer_id: int) -> Optional[dict]:
    for event in reversed(events):
        if event.get("type") == "divine_result" and event.get("speaker_id") == seer_id:
            return event.get("payload")
    return None


def update_player_seen_index(game_state, player_id: int):
    game_state.players[player_id].last_seen_event_idx = len(game_state.events)
