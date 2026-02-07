import random
from werewolf.engine.state import PlayerState


def assign_roles(
    n_players: int,
    n_wolves: int,
    rng: random.Random
) -> dict[int, PlayerState]:
    if n_wolves >= n_players:
        raise ValueError("Number of wolves must be less than total players")
    if n_wolves < 1:
        raise ValueError("Need at least 1 wolf")
    if n_players < 3:
        raise ValueError("Need at least 3 players")

    player_ids = list(range(n_players))
    rng.shuffle(player_ids)

    wolf_ids = set(player_ids[:n_wolves])
    seer_id = player_ids[n_wolves]

    players = {}
    for pid in range(n_players):
        if pid in wolf_ids:
            role = "werewolf"
            team = "wolf"
        elif pid == seer_id:
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
