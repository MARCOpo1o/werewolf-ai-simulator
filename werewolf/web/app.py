from pathlib import Path
import threading

from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template, send_file
from werkzeug.exceptions import BadRequest, UnsupportedMediaType

from werewolf.engine.game import GameEngine
from werewolf.llm.registry import get_api_key, selectable_models
from werewolf.reporting.privacy import build_public_report
from werewolf.reporting.repository import (
    GameRepository,
    InvalidCursor,
    InvalidGameId,
)
from werewolf.reporting.service import load_full_report
from werewolf.web.services import (
    RequestValidationError,
    create_engine_from_payload,
    health_check,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

app = Flask(__name__)

game_engine: GameEngine | None = None
_game_lock = threading.RLock()
game_repository = GameRepository(Path("outputs/games"))


def _request_json_object():
    try:
        data = request.get_json(silent=False)
    except (BadRequest, UnsupportedMediaType):
        return None, (jsonify({
            "error": "Invalid JSON",
            "errors": {"request": {
                "code": "invalid_json",
                "message": "Request body must contain valid JSON.",
            }},
        }), 400)
    if not isinstance(data, dict):
        return None, (jsonify({
            "error": "Invalid request",
            "errors": {"request": {
                "code": "invalid_type",
                "message": "Request body must be a JSON object.",
            }},
        }), 400)
    return data, None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/games")
def game_history():
    return render_template("history.html")


@app.route("/games/<game_id>")
def game_report_page(game_id: str):
    try:
        validate = game_repository.log_path(game_id)
    except InvalidGameId:
        return render_template("report.html", game_id=None), 404
    if not validate.exists():
        return render_template("report.html", game_id=None), 404
    return render_template("report.html", game_id=game_id)


@app.route("/api/state")
def get_state():
    with _game_lock:
        if game_engine is None:
            return jsonify({"error": "No game in progress", "game": None})
        return jsonify({"game": game_engine.get_state_dict()})


@app.route("/api/models")
def get_models():
    return jsonify({
        "models": [
            {
                "alias": spec.alias,
                "display_name": spec.display_name,
                "family": spec.family,
                "description": spec.description,
                "provider": spec.provider,
                "requested_model": spec.model,
                "registry_reasoning_default": spec.reasoning_effort,
                "speed_tier": spec.speed_tier,
                "cost_tier": spec.cost_tier,
                "tags": list(spec.tags),
                "experimental": spec.experimental,
                "key_configured": bool(get_api_key(spec)),
            }
            for spec in selectable_models()
        ]
    })


@app.route("/api/models/<alias>/health-check", methods=["POST"])
def check_model(alias: str):
    data, error_response = _request_json_object()
    if error_response is not None:
        return error_response
    body, status_code = health_check(alias, data)
    return jsonify(body), status_code


@app.route("/api/new", methods=["POST"])
def new_game():
    global game_engine
    data, error_response = _request_json_object()
    if error_response is not None:
        return error_response
    try:
        new_engine = create_engine_from_payload(data)
    except RequestValidationError as exc:
        return jsonify({"error": "Invalid game configuration", "errors": exc.errors}), 400
    except Exception:
        app.logger.exception("Game construction failed")
        return jsonify({
            "error": "Game could not be created",
            "errors": {"request": {
                "code": "engine_initialization_failed",
                "message": "The game engine could not be initialized.",
            }},
        }), 500

    try:
        state = new_engine.get_state_dict()
    except Exception:
        try:
            new_engine.close()
        except Exception:
            app.logger.exception("Failed to close unusable new game")
        app.logger.exception("New game state could not be serialized")
        return jsonify({
            "error": "Game could not be created",
            "errors": {"request": {
                "code": "state_serialization_failed",
                "message": "The new game state could not be serialized.",
            }},
        }), 500

    with _game_lock:
        old_engine = game_engine
        game_engine = new_engine
    if old_engine is not None:
        try:
            old_engine.close()
        except Exception:
            app.logger.exception("Failed to close previous game")
    return jsonify({
        "game": state,
        "links": {
            "history": "/games",
            "report": f"/games/{state['game_id']}",
        },
    })


@app.route("/api/usage")
def get_usage():
    """Live usage/cost summary for the current game (see UsageLedger)."""
    with _game_lock:
        if game_engine is None:
            return jsonify({"error": "No game in progress", "usage": None})
        return jsonify({
            "game_id": game_engine.state.game_id,
            "usage": game_engine.ledger.game_summary(),
        })


@app.route("/api/advance", methods=["POST"])
def advance_phase():
    global game_engine

    with _game_lock:
        if game_engine is None:
            return jsonify({"error": "No game in progress"}), 400

        result = game_engine.run_next_phase()
        if result.get("done"):
            try:
                game_repository.refresh_game(game_engine.state.game_id)
            except Exception:
                app.logger.exception("Completed game index refresh failed")
        return jsonify({
            "result": result,
            "game": game_engine.get_state_dict(),
            "links": {
                "history": "/games",
                "report": f"/games/{game_engine.state.game_id}",
            },
        })


@app.route("/api/games")
def list_games():
    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400
    cursor = request.args.get("cursor")
    with _game_lock:
        active_game_id = game_engine.state.game_id if game_engine else None
    try:
        return jsonify(game_repository.list_games(
            limit=limit, cursor=cursor, active_game_id=active_game_id,
        ))
    except InvalidCursor as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/games/<game_id>/report")
def get_game_report(game_id: str):
    raw_private = request.args.get("include_private", "false").lower()
    if raw_private not in {"true", "false"}:
        return jsonify({"error": "include_private must be true or false"}), 400
    include_private = raw_private == "true"
    with _game_lock:
        active_game_id = game_engine.state.game_id if game_engine else None
    try:
        report = load_full_report(
            game_repository, game_id, active_game_id=active_game_id,
        )
    except InvalidGameId:
        report = None
    if report is None:
        return jsonify({"error": "Game not found"}), 404
    if include_private:
        report["privacy"] = {
            "include_private": True,
            "spoiler_protection_only": True,
        }
        response = jsonify(report)
        response.headers["Cache-Control"] = "no-store"
        return response
    return jsonify(build_public_report(report))


@app.route("/api/games/<game_id>/raw")
def download_game_log(game_id: str):
    try:
        path = game_repository.log_path(game_id)
    except InvalidGameId:
        return jsonify({"error": "Game not found"}), 404
    if not path.exists():
        return jsonify({"error": "Game not found"}), 404
    response = send_file(
        path.resolve(), mimetype="application/x-ndjson", as_attachment=True,
        download_name=path.name,
    )
    response.headers["Cache-Control"] = "no-store"
    return response


def main():
    app.run(debug=True, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
