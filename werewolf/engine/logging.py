import json
import os
from typing import TextIO


class JSONLLogger:
    def __init__(self, output_dir: str, game_id: str):
        os.makedirs(output_dir, exist_ok=True)
        self.filepath = os.path.join(output_dir, f"{game_id}.jsonl")
        self.file: TextIO = open(self.filepath, "w")

    def log_config(self, config: dict):
        self._write({"type": "config", **config})

    def log_event(self, event: dict):
        self._write({"type": "event", "event": event})

    def log_outcome(self, winner: str, rounds: int, remaining: list[int]):
        self._write({
            "type": "outcome",
            "winner": winner,
            "rounds": rounds,
            "remaining": remaining
        })

    def _write(self, obj: dict):
        self.file.write(json.dumps(obj) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class ConsoleTranscript:
    def __init__(self, show_all: bool = True):
        self.show_all = show_all

    def print_phase_header(self, round_num: int, phase: str):
        if "night" in phase:
            print(f"\n{'='*60}")
            print(f"  NIGHT {round_num}")
            print(f"{'='*60}")
        elif phase == "day_announce":
            print(f"\n{'='*60}")
            print(f"  DAY {round_num}")
            print(f"{'='*60}")

    def print_event(self, event: dict, players: dict):
        event_type = event.get("type")
        channel = event.get("channel", "public")
        speaker_id = event.get("speaker_id")
        payload = event.get("payload", {})

        if not self.show_all and channel == "moderator_only":
            return

        if event_type == "message":
            text = payload.get("text", "")
            role = players[speaker_id].role if speaker_id in players else "?"
            role_tag = f"[{role.upper()}]" if role == "werewolf" else ""
            
            if channel == "werewolf":
                print(f"\n  [WOLF CHAT] P{speaker_id} {role_tag}:")
                print(f"    \"{text}\"")
            else:
                print(f"\n  [SPEAKS] P{speaker_id} {role_tag}:")
                print(f"    \"{text}\"")

        elif event_type == "thought":
            thought = payload.get("thought", "")
            role = players[speaker_id].role if speaker_id in players else "?"
            role_tag = f"({role})"
            print(f"\n  [THINKS] P{speaker_id} {role_tag}:")
            for line in self._wrap_text(thought, 70):
                print(f"    {line}")

        elif event_type == "death_announcement":
            victim_id = payload.get("victim_id")
            victim_role = payload.get("victim_role")
            cause = payload.get("cause")
            print()
            if cause == "wolf_kill":
                print(f"  >>> P{victim_id} was KILLED during the night!")
            else:
                print(f"  >>> P{victim_id} was ELIMINATED by vote! They were a {victim_role.upper()}.")

        elif event_type == "divine_result":
            target_id = payload.get("target_id")
            is_werewolf = payload.get("is_werewolf")
            result = "WEREWOLF" if is_werewolf else "NOT WEREWOLF"
            print(f"\n  [SEER DIVINE] P{speaker_id} checked P{target_id} --> {result}")

        elif event_type == "vote":
            voter_id = payload.get("voter_id")
            target_id = payload.get("target_id")
            print(f"  [VOTE] P{voter_id} --> P{target_id}")

        elif event_type == "elimination":
            vote_counts = payload.get("vote_counts", {})
            votes_str = ", ".join(f"P{k}:{v}" for k, v in sorted(vote_counts.items()))
            print(f"\n  Final votes: {votes_str}")

        elif event_type == "game_status":
            wolves = payload.get("alive_wolves")
            villagers = payload.get("alive_villagers")
            print(f"\n  [STATUS] Wolves alive: {wolves} | Villagers alive: {villagers}")

        elif event_type == "win":
            winner = payload.get("winner")
            remaining = payload.get("remaining", [])
            print(f"\n{'='*60}")
            print(f"  GAME OVER - {winner.upper()} WINS!")
            print(f"  Survivors: {', '.join(f'P{p}' for p in remaining)}")
            print(f"{'='*60}")

    def _wrap_text(self, text: str, width: int) -> list[str]:
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return lines if lines else [""]

    def print_role_reveal(self, players: dict):
        print("\n" + "="*60)
        print("  ROLE ASSIGNMENT (Secret - only moderator sees this)")
        print("="*60)
        wolves = []
        seer = None
        villagers = []
        for pid in sorted(players.keys()):
            p = players[pid]
            if p.role == "werewolf":
                wolves.append(f"P{pid}")
            elif p.role == "seer":
                seer = f"P{pid}"
            else:
                villagers.append(f"P{pid}")
        print(f"  Werewolves: {', '.join(wolves)}")
        if seer:
            print(f"  Seer:       {seer}")
        print(f"  Villagers:  {', '.join(villagers)}")
        print("="*60)
