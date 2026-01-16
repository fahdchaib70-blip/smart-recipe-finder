import os
import re
import logging
from typing import List, Dict, Any

import chromadb
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------
# Environment & logging
# ---------------------------------------------------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# MongoDB setup
# ---------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)

db = client["smart_recipe_db"]
recipes_collection = db["recipes"]

# ---------------------------------------------------------------------
# Embedding model + ChromaDB setup
# ---------------------------------------------------------------------
_EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

_CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
_chroma_client = chromadb.PersistentClient(path=_CHROMA_PATH)
collection = _chroma_client.get_or_create_collection(name="recipes_embeddings")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def normalize_text(text_list: List[str]) -> List[str]:
    """
    Clean and normalize a list of strings:
      - lowercase
      - strip spaces
      - collapse multiple spaces
      - remove special characters (keeps letters/numbers/underscore + spaces)

    Args:
        text_list: list of strings

    Returns:
        Normalized list of strings
    """
    normalized: List[str] = []
    for t in text_list:
        t = (t or "").strip().lower()
        t = re.sub(r"\s+", " ", t)        # collapse multiple spaces
        t = re.sub(r"[^\w\s]", "", t)     # remove special characters
        normalized.append(t)
    return normalized


def vectorize_text(title: str, ingredients: List[str], directions: List[str]):
    """
    Convert a recipe (title + ingredients + directions) into a single embedding vector.
    """
    text = f"{title} {' '.join(ingredients)} {' '.join(directions)}"
    return _EMBEDDING_MODEL.encode([text])[0]


def process_batch(batch: List[Dict[str, Any]]) -> None:
    """
    Process a list of MongoDB recipe documents:
      - normalize ingredients/directions
      - build embeddings
      - store everything in ChromaDB
    """
    ids: List[str] = []
    embeddings = []
    metadatas: List[Dict[str, Any]] = []

    for recipe in batch:
        try:
            title = (recipe.get("name", "") or "").strip() or "Unnamed Recipe"
            ingredients = recipe.get("ingredients", []) or []
            directions = recipe.get("directions", []) or []

            # Normalize lists
            ingredients = normalize_text(ingredients)
            directions = normalize_text(directions)

            # Skip incomplete recipes
            if not ingredients or not directions:
                continue

            ingredients_str = ", ".join(ingredients)
            directions_str = ". ".join(directions)

            embedding = vectorize_text(title, ingredients, directions)

            ids.append(str(recipe["_id"]))
            metadatas.append(
                {
                    "title": title,
                    "ingredients": ingredients_str,
                    "directions": directions_str,
                }
            )
            embeddings.append(embedding)

        except Exception as e:
            logger.warning("Erreur lors du traitement de la recette %s: %s", recipe.get("_id"), e)

    # Push to Chroma only if we have data
    if ids:
        collection.add(
            documents=["" for _ in ids],  # kept identical (empty documents)
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )


def main() -> None:
    """
    Load recipes from MongoDB and index them into ChromaDB in batches.
    """
    limit = int(os.getenv("RECIPE_LIMIT", "5000"))
    batch_size = int(os.getenv("BATCH_SIZE", "100"))

    recipes = list(
        recipes_collection.find(
            {"directions": {"$exists": True, "$ne": []}},
            {"_id": 1, "name": 1, "ingredients": 1, "directions": 1},
        ).limit(limit)
    )

    total_batches = (len(recipes) + batch_size - 1) // batch_size
    logger.info("Indexing %d recipes (batch_size=%d, batches=%d).", len(recipes), batch_size, total_batches)

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch = recipes[start:end]

        logger.info("Processing batch %d/%d...", batch_idx + 1, total_batches)
        try:
            process_batch(batch)
            logger.info("Finished batch %d/%d.", batch_idx + 1, total_batches)
        except Exception as e:
            logger.error("Error processing batch %d/%d: %s", batch_idx + 1, total_batches, e)

    logger.info("All batches processed successfully!")


if __name__ == "__main__":
    main()
