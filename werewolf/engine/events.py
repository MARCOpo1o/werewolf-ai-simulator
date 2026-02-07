import time
from typing import Optional


def create_event(
    game_state,
    event_type: str,
    channel: str,
    payload: dict,
    speaker_id: Optional[int] = None
) -> dict:
    event = {
        "id": game_state.next_event_id(),
        "t": time.time(),
        "round": game_state.round,
        "phase": game_state.phase,
        "type": event_type,
        "channel": channel,
        "speaker_id": speaker_id,
        "payload": payload
    }
    game_state.events.append(event)
    return event


def create_message_event(
    game_state,
    channel: str,
    speaker_id: int,
    text: str
) -> dict:
    return create_event(
        game_state,
        event_type="message",
        channel=channel,
        payload={"text": text},
        speaker_id=speaker_id
    )


def create_thought_event(
    game_state,
    speaker_id: int,
    thought: str
) -> dict:
    return create_event(
        game_state,
        event_type="thought",
        channel="moderator_only",
        payload={"thought": thought},
        speaker_id=speaker_id
    )


def create_kill_event(
    game_state,
    victim_id: int,
    kill_votes: dict[int, int]
) -> dict:
    return create_event(
        game_state,
        event_type="kill",
        channel="moderator_only",
        payload={"victim_id": victim_id, "votes": kill_votes}
    )


def create_death_announcement_event(
    game_state,
    victim_id: int,
    victim_role: str,
    cause: str  # "wolf_kill" | "vote_elimination"
) -> dict:
    if cause == "wolf_kill":
        return create_event(
            game_state,
            event_type="death_announcement",
            channel="public",
            payload={"victim_id": victim_id, "cause": cause}
        )
    else:
        return create_event(
            game_state,
            event_type="death_announcement",
            channel="public",
            payload={"victim_id": victim_id, "victim_role": victim_role, "cause": cause}
        )


def create_divine_result_event(
    game_state,
    seer_id: int,
    target_id: int,
    is_werewolf: bool
) -> dict:
    return create_event(
        game_state,
        event_type="divine_result",
        channel="seer_private",
        payload={"target_id": target_id, "is_werewolf": is_werewolf},
        speaker_id=seer_id
    )


def create_vote_event(
    game_state,
    voter_id: int,
    target_id: int
) -> dict:
    return create_event(
        game_state,
        event_type="vote",
        channel="public",
        payload={"voter_id": voter_id, "target_id": target_id},
        speaker_id=voter_id
    )


def create_elimination_event(
    game_state,
    eliminated_id: int,
    eliminated_role: str,
    vote_counts: dict[int, int]
) -> dict:
    return create_event(
        game_state,
        event_type="elimination",
        channel="public",
        payload={
            "eliminated_id": eliminated_id,
            "eliminated_role": eliminated_role,
            "vote_counts": vote_counts
        }
    )


def create_phase_event(game_state, new_phase: str) -> dict:
    return create_event(
        game_state,
        event_type="phase_change",
        channel="public",
        payload={"new_phase": new_phase}
    )


def create_win_event(game_state, winner: str, remaining_ids: list[int]) -> dict:
    return create_event(
        game_state,
        event_type="win",
        channel="public",
        payload={"winner": winner, "remaining": remaining_ids}
    )


def create_game_status_event(
    game_state,
    alive_wolves: int,
    alive_villagers: int
) -> dict:
    return create_event(
        game_state,
        event_type="game_status",
        channel="public",
        payload={"alive_wolves": alive_wolves, "alive_villagers": alive_villagers}
    )
