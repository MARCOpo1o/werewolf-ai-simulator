import random
import unittest

from werewolf.engine.state import GameState
from werewolf.roles.assign import assign_roles


class RoleAssignmentTests(unittest.TestCase):
    def test_assign_roles_without_seer(self):
        players = assign_roles(n_players=7, n_wolves=2, n_seers=0, rng=random.Random(1))
        seers = [p for p in players.values() if p.role == "seer"]
        self.assertEqual(len(seers), 0)

    def test_assign_roles_with_single_seer(self):
        players = assign_roles(n_players=7, n_wolves=2, n_seers=1, rng=random.Random(1))
        seers = [p for p in players.values() if p.role == "seer"]
        self.assertEqual(len(seers), 1)

    def test_assign_roles_rejects_invalid_seer_count(self):
        with self.assertRaises(ValueError):
            assign_roles(n_players=7, n_wolves=2, n_seers=2, rng=random.Random(1))

    def test_assign_roles_requires_at_least_one_villager(self):
        with self.assertRaises(ValueError):
            assign_roles(n_players=3, n_wolves=2, n_seers=1, rng=random.Random(1))


class GameIdTests(unittest.TestCase):
    def test_game_id_is_unique_across_rapid_initialization(self):
        ids = set()
        for _ in range(200):
            state = GameState(seed=42, round=0, phase="setup", players={})
            ids.add(state.game_id)
        self.assertEqual(len(ids), 200)


if __name__ == "__main__":
    unittest.main()
