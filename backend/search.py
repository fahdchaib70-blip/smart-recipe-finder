import os
import logging
from typing import List, Tuple, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Global resources (loaded once)
# ---------------------------------------------------------------------
# Embedding model used to convert user queries to vectors
_EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Persistent Chroma client + collection (vector store)
_CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
_CHROMA_CLIENT = chromadb.PersistentClient(path=_CHROMA_PATH)
_COLLECTION = _CHROMA_CLIENT.get_or_create_collection(name="recipes_embeddings")


def search_recipes(query: str, top_k: int = 3) -> Tuple[List[Dict[str, Any]], List[List[float]], List[float]]:
    """
    Search for the most relevant recipes given a user query.

    The function:
      1) encodes the query into an embedding vector,
      2) queries ChromaDB for the top-k nearest recipes,
      3) returns recipe metadata + corresponding embeddings + query embedding.

    Args:
        query: User text query (must be non-empty).
        top_k: Number of recipes to return (default: 3).

    Returns:
        A tuple (recipes_meta, recipes_embeddings, query_embedding) where:
          - recipes_meta: list of recipe metadata dicts
          - recipes_embeddings: list of embeddings for the retrieved recipes
          - query_embedding: embedding vector for the input query

    Raises:
        ValueError: if query is empty or only whitespace.
    """
    if query is None or not str(query).strip():
        raise ValueError("La requête ne peut pas être vide.")

    # Encode query -> vector
    query_embedding = _EMBEDDING_MODEL.encode([query])[0]
    logger.info("Query embedding generated (dim=%s)", len(query_embedding))

    # Vector search in Chroma
    results = _COLLECTION.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["embeddings", "metadatas", "documents"],
    )

    # Extract results
    recipes_meta: List[Dict[str, Any]] = []
    recipes_embeddings: List[List[float]] = []

    metadatas = results.get("metadatas", [[]])[0]
    embeddings = results.get("embeddings", [[]])[0]

    for idx, meta in enumerate(metadatas):
        # Ensure each item has an id field (use existing if present)
        meta["id"] = meta.get("id", f"recipe_{idx}")
        recipes_meta.append(meta)

        # Collect the corresponding embedding
        if idx < len(embeddings):
            recipes_embeddings.append(embeddings[idx])

    logger.info("Retrieved %d recipe(s) from ChromaDB.", len(recipes_meta))
    for r in recipes_meta:
        logger.info("Recipe hit: %s", r)

    return recipes_meta, recipes_embeddings, query_embedding
