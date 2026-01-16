import os
import logging
from typing import Any, Dict, List, Tuple

import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------
# Environment & logging
# ---------------------------------------------------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini / Google GenAI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Model & generation parameters (kept identical to your current behavior)
_GEMINI_MODEL_NAME = "models/gemini-1.5-flash"
_GEN_CONFIG = {"temperature": 0.7, "max_output_tokens": 300}


def generate_response(
    query: str,
    recipes: List[Dict[str, Any]],
    recipe_embeddings: List[List[float]],
    query_embedding: List[float],
) -> str:
    """
    Generate a natural-language answer using the user query and the top recipes.

    The function:
      - computes cosine similarity between query_embedding and each recipe embedding
      - keeps the best recipes (top 3)
      - builds a prompt for Gemini
      - appends a "Video Links" section at the end of the generated text

    Args:
        query: User query text.
        recipes: Recipe metadata list (each item is a dict).
        recipe_embeddings: Embeddings for the retrieved recipes (same order as recipes).
        query_embedding: Embedding vector for the query.

    Returns:
        Generated answer (string). If no results or API issue, returns a user-friendly message.
    """
    if not recipes or not recipe_embeddings:
        return "Aucune recette pertinente trouvée."

    # -----------------------------------------------------------------
    # 1) Rank recipes by cosine similarity
    # -----------------------------------------------------------------
    similarity_scores = cosine_similarity([query_embedding], recipe_embeddings)[0]
    ranked = sorted(zip(recipes, similarity_scores), key=lambda x: x[1], reverse=True)

    # -----------------------------------------------------------------
    # 2) Build prompt
    # -----------------------------------------------------------------
    prompt_lines: List[str] = [f"User query: {query}", "", "Here are some recipes to consider:"]
    recipe_links: Dict[str, str] = {}  # Store recipe_id -> video link (placeholder)

    for recipe, _score in ranked[:3]:
        recipe_id = recipe.get("id", "unknown_id")
        title = (recipe.get("title", "") or "").strip() or "Unnamed Recipe"
        ingredients = recipe.get("ingredients", "")
        directions = (recipe.get("directions", "") or "").strip()

        # Keep the same truncation behavior
        short_directions = directions[:200] if len(directions) > 200 else directions

        # Placeholder YouTube link, same behavior as your code
        youtube_link = f"https://www.youtube.com/watch?v={recipe_id}"
        recipe_links[recipe_id] = youtube_link

        prompt_lines.append(f"- {title} (Video: {youtube_link})")
        prompt_lines.append(f"  Ingredients: {ingredients}")
        prompt_lines.append(f"  Steps: {short_directions}...")
        prompt_lines.append("")

    prompt_lines.append(
        "Please provide a helpful response considering the following user preferences:\n"
        "1. Main preferences: dietary restrictions, cuisine type, or cooking style.\n"
        "2. Ease of preparation: simple to prepare, minimal ingredients.\n"
        "3. Flavor and enjoyment: balanced and enjoyable dishes.\n"
        "4. Additional criteria: health focus, specific cuisines.\n"
        "Additionally, compare the recipes and explain why other options may be less suitable.\n"
        "Suggest a complementary dish or side if applicable.\n\n"
        "Please limit your response to approximately 200 words."
    )

    input_text = "\n".join(prompt_lines)

    # -----------------------------------------------------------------
    # 3) Call Gemini
    # -----------------------------------------------------------------
    try:
        model = genai.GenerativeModel(model_name=_GEMINI_MODEL_NAME)
        response = model.generate_content(
            contents=[input_text],
            generation_config=_GEN_CONFIG,
        )

        if response and getattr(response, "candidates", None):
            candidate = response.candidates[0]
            generated_text = "".join(part.text for part in candidate.content.parts)

            # Keep same behavior: append Video Links block at the end
            generated_text += "\n\nVideo Links:\n" + "\n".join(
                [f"{rid}: {link}" for rid, link in recipe_links.items()]
            )
            return generated_text.strip()

        return "Aucune réponse reçue du modèle."

    except Exception as e:
        logger.error("Erreur inattendue : %s", e)
        return "Une erreur inattendue est survenue lors de la génération de la réponse."
