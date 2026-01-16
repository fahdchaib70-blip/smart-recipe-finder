import os
import logging

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from marshmallow import Schema, fields, ValidationError
from dotenv import load_dotenv

from search import search_recipes
from generate_response import generate_response

# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Rate limiting (basic abuse protection)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
)


# ---------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------
class SearchSchema(Schema):
    """
    Validate incoming search request payload.

    Expected JSON:
      - query: non-empty string
      - top_k: positive integer (optional)
    """
    query = fields.Str(required=True, validate=lambda s: len(s.strip()) > 0)
    top_k = fields.Int(load_default=5, validate=lambda n: n > 0)


search_schema = SearchSchema()


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.route("/search", methods=["POST"])
@limiter.limit("10 per minute")
def search_endpoint():
    """
    Main endpoint:
      - validates request JSON
      - retrieves similar recipes via vector search
      - generates a final natural-language response
      - returns recipes + placeholder video links
    """
    # 1) Validate user input
    try:
        payload = request.get_json(silent=True) or {}
        data = search_schema.load(payload)
    except ValidationError as err:
        return jsonify(err.messages), 400

    user_query = data["query"].strip()
    top_k = data["top_k"]  # default handled by schema

    # 2) Search recipes
    try:
        recipes, recipe_embeddings, query_embedding = search_recipes(user_query, top_k=top_k)
    except Exception as e:
        logger.exception("Recipe search failed: %s", e)
        return jsonify({"error": "Erreur interne lors de la recherche des recettes."}), 500

    if not recipes:
        return jsonify(
            {
                "query": user_query,
                "response": "No relevant recipes found for your query.",
                "recipes": [],
                "videos": {},
            }
        ), 200

    # 3) Generate AI response
    try:
        response_text = generate_response(user_query, recipes, recipe_embeddings, query_embedding)
    except Exception as e:
        logger.exception("Response generation failed: %s", e)
        response_text = "Erreur lors de la génération de la réponse."

    # 4) Build placeholder video links (kept as in your current behavior)
    recipe_videos = {
        recipe.get("id", f"recipe_{idx}"): f"https://www.youtube.com/watch?v={recipe.get('id', '')}"
        for idx, recipe in enumerate(recipes)
    }

    return jsonify(
        {
            "query": user_query,
            "response": response_text,
            "recipes": recipes,
            "videos": recipe_videos,
        }
    ), 200


# ---------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------
@app.errorhandler(400)
def bad_request(_error):
    return jsonify({"error": "Bad Request"}), 400


@app.errorhandler(500)
def internal_error(_error):
    return jsonify({"error": "Internal Server Error"}), 500


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
