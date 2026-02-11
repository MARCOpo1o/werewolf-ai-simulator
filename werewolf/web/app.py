import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template

from werewolf.engine.game import GameEngine

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

app = Flask(__name__)

game_engine: GameEngine | None = None


def get_api_key() -> str:
    return os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY", "")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def get_state():
    if game_engine is None:
        return jsonify({"error": "No game in progress", "game": None})
    return jsonify({"game": game_engine.get_state_dict()})


@app.route("/api/new", methods=["POST"])
def new_game():
    global game_engine

    data = request.get_json() or {}
    n_players = data.get("n_players", 7)
    n_wolves = data.get("n_wolves", 2)
    n_seers = data.get("n_seers", 1)
    seed = data.get("seed", 42)
    model = data.get("model", "grok-4-1-fast")

    api_key = get_api_key()
    if not api_key:
        return jsonify({"error": "Set GROK_API_KEY or XAI_API_KEY in .env"}), 400
    if n_seers not in (0, 1):
        return jsonify({"error": "n_seers must be 0 or 1"}), 400
    if n_wolves >= n_players:
        return jsonify({"error": "n_wolves must be less than n_players"}), 400
    if n_wolves + n_seers >= n_players:
        return jsonify({"error": "Need at least one villager"}), 400

    game_engine = GameEngine(
        n_players=n_players,
        n_wolves=n_wolves,
        n_seers=n_seers,
        seed=seed,
        api_key=api_key,
        model=model,
        show_all_channels=True,
        show_prompts=False
    )

    return jsonify({"game": game_engine.get_state_dict()})


@app.route("/api/advance", methods=["POST"])
def advance_phase():
    global game_engine

    if game_engine is None:
        return jsonify({"error": "No game in progress"}), 400

    result = game_engine.run_next_phase()
    return jsonify({
        "result": result,
        "game": game_engine.get_state_dict()
    })


def main():
    app.run(debug=True, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
