from dataclasses import dataclass, field
from typing import Optional
import random
import time
import uuid


@dataclass
class PlayerState:
    id: int
    alive: bool
    role: str  # "werewolf" | "seer" | "villager"
    team: str  # "wolf" | "village"
    last_seen_event_idx: int = 0

    def to_public_dict(self) -> dict:
        return {"id": self.id}

    def to_self_dict(self) -> dict:
        return {"id": self.id, "role": self.role, "team": self.team}


@dataclass
class GameState:
    seed: int
    round: int
    phase: str  # "night_wolf_chat" | "night_wolf_kill" | "night_seer" | "day_announce" | "day_discuss" | "day_vote"
    players: dict[int, PlayerState]
    events: list[dict] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    rng: random.Random = field(default_factory=random.Random)
    winner: Optional[str] = None  # "wolf" | "village" | None
    game_id: str = ""

    def __post_init__(self):
        if not self.game_id:
            ts_ms = int(time.time() * 1000)
            suffix = uuid.uuid4().hex[:8]
            self.game_id = f"game_{self.seed}_{ts_ms}_{suffix}"

    def get_alive_players(self) -> list[PlayerState]:
        return sorted(
            [p for p in self.players.values() if p.alive],
            key=lambda p: p.id
        )

    def get_alive_wolves(self) -> list[PlayerState]:
        return [p for p in self.get_alive_players() if p.role == "werewolf"]

    def get_alive_villagers(self) -> list[PlayerState]:
        return [p for p in self.get_alive_players() if p.team == "village"]

    def get_wolf_ids(self) -> list[int]:
        return [p.id for p in self.players.values() if p.role == "werewolf"]

    def get_seer(self) -> Optional[PlayerState]:
        for p in self.players.values():
            if p.role == "seer":
                return p
        return None

    def check_win_condition(self) -> Optional[str]:
        alive_wolves = len(self.get_alive_wolves())
        alive_villagers = len(self.get_alive_villagers())

        if alive_wolves == 0:
            return "village"
        if alive_wolves >= alive_villagers:
            return "wolf"
        return None

    def next_event_id(self) -> int:
        return len(self.events)
