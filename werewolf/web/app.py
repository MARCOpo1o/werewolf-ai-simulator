from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template

from werewolf.engine.game import GameEngine
from werewolf.llm.registry import build_provider, get_api_key, resolve

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

app = Flask(__name__)

game_engine: GameEngine | None = None


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
    model = data.get("model", "grok-4.3")

    # Resolve aliases (fast/reasoning/gemini_flash_lite/...) and full model
    # IDs to the right provider and key env vars.
    spec = resolve(model)
    api_key = get_api_key(spec)
    if not api_key:
        env_names = " or ".join(spec.api_key_env) or "an API key"
        return jsonify({"error": f"Set {env_names} in .env for model {model}"}), 400
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
        model=spec.model,
        provider=build_provider(spec, api_key=api_key),
        model_alias=spec.alias,
        reasoning_effort=spec.reasoning_effort,
        show_all_channels=True,
        show_prompts=False
    )

    return jsonify({"game": game_engine.get_state_dict()})


@app.route("/api/usage")
def get_usage():
    """Live usage/cost summary for the current game (see UsageLedger)."""
    if game_engine is None:
        return jsonify({"error": "No game in progress", "usage": None})
    return jsonify({
        "game_id": game_engine.state.game_id,
        "usage": game_engine.ledger.game_summary(),
    })


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
