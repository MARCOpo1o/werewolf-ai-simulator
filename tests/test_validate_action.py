import math
import unittest

from werewolf.engine.validate import _to_int, validate_action


class ToIntTests(unittest.TestCase):

    # --- valid inputs that must succeed ---

    def test_plain_int(self):
        self.assertEqual(_to_int(0), 0)
        self.assertEqual(_to_int(5), 5)
        self.assertEqual(_to_int(99), 99)

    def test_numeric_string(self):
        self.assertEqual(_to_int("0"), 0)
        self.assertEqual(_to_int("5"), 5)
        self.assertEqual(_to_int("12"), 12)

    def test_p_prefix_uppercase(self):
        self.assertEqual(_to_int("P0"), 0)
        self.assertEqual(_to_int("P5"), 5)
        self.assertEqual(_to_int("P99"), 99)

    def test_p_prefix_lowercase(self):
        self.assertEqual(_to_int("p0"), 0)
        self.assertEqual(_to_int("p5"), 5)

    def test_player_word_prefix(self):
        self.assertEqual(_to_int("Player0"), 0)
        self.assertEqual(_to_int("Player 5"), 5)
        self.assertEqual(_to_int("player3"), 3)
        self.assertEqual(_to_int("PLAYER 12"), 12)

    def test_whitespace_around_value(self):
        self.assertEqual(_to_int(" 3 "), 3)
        self.assertEqual(_to_int(" P0 "), 0)
        self.assertEqual(_to_int("  Player 5  "), 5)

    def test_float_whole_number(self):
        self.assertEqual(_to_int(3.0), 3)
        self.assertEqual(_to_int(0.0), 0)

    def test_float_string_whole_number(self):
        self.assertEqual(_to_int("3.0"), 3)
        self.assertEqual(_to_int("0.0"), 0)

    def test_negative_int(self):
        self.assertEqual(_to_int(-1), -1)
        self.assertEqual(_to_int("-1"), -1)

    # --- invalid inputs that must raise ValueError ---

    def test_bool_rejected(self):
        with self.assertRaises(ValueError):
            _to_int(True)
        with self.assertRaises(ValueError):
            _to_int(False)

    def test_none_rejected(self):
        with self.assertRaises(ValueError):
            _to_int(None)

    def test_empty_string_rejected(self):
        with self.assertRaises(ValueError):
            _to_int("")
        with self.assertRaises(ValueError):
            _to_int("   ")

    def test_bare_p_rejected(self):
        with self.assertRaises(ValueError):
            _to_int("P")
        with self.assertRaises(ValueError):
            _to_int("p")

    def test_garbage_string_rejected(self):
        with self.assertRaises(ValueError):
            _to_int("not_a_number")
        with self.assertRaises(ValueError):
            _to_int("abc")
        with self.assertRaises(ValueError):
            _to_int("kill P3")

    def test_fractional_float_rejected(self):
        with self.assertRaises(ValueError):
            _to_int(3.5)
        with self.assertRaises(ValueError):
            _to_int("3.5")

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            _to_int(float("nan"))

    def test_inf_rejected(self):
        with self.assertRaises(ValueError):
            _to_int(float("inf"))

    def test_list_rejected(self):
        with self.assertRaises(ValueError):
            _to_int([3])

    def test_dict_rejected(self):
        with self.assertRaises(ValueError):
            _to_int({"id": 3})

    def test_p_prefix_with_trailing_junk_rejected(self):
        with self.assertRaises(ValueError):
            _to_int("P2x")
        with self.assertRaises(ValueError):
            _to_int("Player abc")


class ValidateActionTests(unittest.TestCase):

    def _vote_obs(self, alive=None):
        if alive is None:
            alive = [0, 1, 2]
        return {
            "required_action": "vote",
            "self": {"id": 1, "role": "villager"},
            "alive_players": [{"id": i} for i in alive],
            "private_info": {},
        }

    def _kill_obs(self, alive=None, wolf_roster=None):
        if alive is None:
            alive = [0, 1, 2]
        if wolf_roster is None:
            wolf_roster = [1]
        return {
            "required_action": "choose_wolf_kill",
            "self": {"id": 1, "role": "werewolf"},
            "alive_players": [{"id": i} for i in alive],
            "private_info": {"wolf_roster": wolf_roster},
        }

    def _divine_obs(self, alive=None):
        if alive is None:
            alive = [0, 1, 2]
        return {
            "required_action": "seer_divine",
            "self": {"id": 1, "role": "seer"},
            "alive_players": [{"id": i} for i in alive],
            "private_info": {},
        }

    # --- vote normalization ---

    def test_vote_int(self):
        r = {"action": {"vote_target": 2}, "say": None}
        ok, err = validate_action(self._vote_obs(), r, None)
        self.assertTrue(ok, err)
        self.assertEqual(r["action"]["vote_target"], 2)

    def test_vote_string(self):
        r = {"action": {"vote_target": "2"}, "say": None}
        ok, err = validate_action(self._vote_obs(), r, None)
        self.assertTrue(ok, err)
        self.assertEqual(r["action"]["vote_target"], 2)

    def test_vote_p_prefix(self):
        r = {"action": {"vote_target": "P2"}, "say": None}
        ok, err = validate_action(self._vote_obs(), r, None)
        self.assertTrue(ok, err)
        self.assertEqual(r["action"]["vote_target"], 2)

    def test_vote_player_prefix(self):
        r = {"action": {"vote_target": "Player 2"}, "say": None}
        ok, err = validate_action(self._vote_obs(), r, None)
        self.assertTrue(ok, err)
        self.assertEqual(r["action"]["vote_target"], 2)

    def test_vote_float(self):
        r = {"action": {"vote_target": 2.0}, "say": None}
        ok, err = validate_action(self._vote_obs(), r, None)
        self.assertTrue(ok, err)
        self.assertEqual(r["action"]["vote_target"], 2)

    def test_vote_invalid_rejected(self):
        r = {"action": {"vote_target": "garbage"}, "say": None}
        ok, _ = validate_action(self._vote_obs(), r, None)
        self.assertFalse(ok)

    def test_vote_self_rejected(self):
        r = {"action": {"vote_target": 1}, "say": None}
        ok, err = validate_action(self._vote_obs(), r, None)
        self.assertFalse(ok)
        self.assertIn("Cannot vote for yourself", err)

    def test_vote_dead_player_rejected(self):
        r = {"action": {"vote_target": 9}, "say": None}
        ok, err = validate_action(self._vote_obs(), r, None)
        self.assertFalse(ok)
        self.assertIn("not alive", err)

    # --- kill normalization ---

    def test_kill_int(self):
        r = {"action": {"kill_target": 0}, "say": None}
        ok, err = validate_action(self._kill_obs(), r, None)
        self.assertTrue(ok, err)

    def test_kill_p_prefix(self):
        r = {"action": {"kill_target": "P0"}, "say": None}
        ok, err = validate_action(self._kill_obs(), r, None)
        self.assertTrue(ok, err)
        self.assertEqual(r["action"]["kill_target"], 0)

    def test_kill_float(self):
        r = {"action": {"kill_target": 0.0}, "say": None}
        ok, err = validate_action(self._kill_obs(), r, None)
        self.assertTrue(ok, err)
        self.assertEqual(r["action"]["kill_target"], 0)

    def test_kill_fellow_wolf_rejected(self):
        r = {"action": {"kill_target": 1}, "say": None}
        ok, err = validate_action(self._kill_obs(), r, None)
        self.assertFalse(ok)
        self.assertIn("Cannot kill fellow wolf", err)

    # --- divine normalization ---

    def test_divine_string(self):
        r = {"action": {"divine_target": "2"}, "say": None}
        ok, err = validate_action(self._divine_obs(), r, None)
        self.assertTrue(ok, err)
        self.assertEqual(r["action"]["divine_target"], 2)

    def test_divine_self_rejected(self):
        r = {"action": {"divine_target": 1}, "say": None}
        ok, err = validate_action(self._divine_obs(), r, None)
        self.assertFalse(ok)
        self.assertIn("Cannot divine yourself", err)

    # --- runoff normalization ---

    def test_runoff_p_prefix_in_candidates(self):
        obs = {
            "required_action": "runoff_vote",
            "self": {"id": 1, "role": "villager"},
            "alive_players": [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}],
            "private_info": {},
            "turn_context": {"runoff_candidates": ["P2", 3]},
        }
        r = {"action": {"vote_target": "3"}, "say": None}
        ok, err = validate_action(obs, r, None)
        self.assertTrue(ok, err)
        self.assertEqual(r["action"]["vote_target"], 3)


if __name__ == "__main__":
    unittest.main()
