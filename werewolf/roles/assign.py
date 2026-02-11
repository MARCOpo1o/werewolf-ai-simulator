import random
from werewolf.engine.state import PlayerState


def assign_roles(
    n_players: int,
    n_wolves: int,
    rng: random.Random,
    n_seers: int = 1
) -> dict[int, PlayerState]:
    if n_wolves >= n_players:
        raise ValueError("Number of wolves must be less than total players")
    if n_wolves < 1:
        raise ValueError("Need at least 1 wolf")
    if n_players < 3:
        raise ValueError("Need at least 3 players")
    if n_seers not in (0, 1):
        raise ValueError("Number of seers must be 0 or 1")
    if n_players - n_wolves - n_seers < 1:
        raise ValueError("Need at least 1 villager")

    player_ids = list(range(n_players))
    rng.shuffle(player_ids)

    wolf_ids = set(player_ids[:n_wolves])
    seer_id = player_ids[n_wolves] if n_seers == 1 else None

    players = {}
    for pid in range(n_players):
        if pid in wolf_ids:
            role = "werewolf"
            team = "wolf"
        elif seer_id is not None and pid == seer_id:
            role = "seer"
            team = "village"
        else:
            role = "villager"
            team = "village"

        players[pid] = PlayerState(
            id=pid,
            alive=True,
            role=role,
            team=team
        )

    return players
