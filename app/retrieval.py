"""
Retrieval module: loads catalog.json, builds a FAISS index on first use,
and provides semantic search over SHL assessments.
"""
import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Lazy imports — loaded once at startup
_index = None
_catalog: List[Dict[str, Any]] = []
_embeddings_model = None


def _get_catalog_path() -> Path:
    here = Path(__file__).parent.parent
    return here / "data" / "catalog.json"


def _load_catalog() -> List[Dict[str, Any]]:
    path = _get_catalog_path()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_text(item: Dict[str, Any]) -> str:
    """Combine fields into a single string for embedding."""
    types = ", ".join(item.get("test_type", []))
    levels = ", ".join(item.get("job_levels", []))
    keywords = ", ".join(item.get("keywords", []))
    return (
        f"{item['name']}. "
        f"Type: {types}. "
        f"Description: {item['description']} "
        f"Job levels: {levels}. "
        f"Keywords: {keywords}."
    )


def initialize():
    """Build the FAISS index. Called once at server startup."""
    global _index, _catalog, _embeddings_model

    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError:
        raise RuntimeError("sentence-transformers and faiss-cpu are required.")

    _catalog = _load_catalog()
    _embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [_build_text(item) for item in _catalog]
    embeddings = _embeddings_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    _index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    _index.add(embeddings)

    print(f"[retrieval] Indexed {len(_catalog)} assessments.")


def search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Return top_k catalog items for a query string."""
    global _index, _catalog, _embeddings_model

    if _index is None:
        initialize()

    import faiss

    q_emb = _embeddings_model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_emb)

    distances, indices = _index.search(q_emb, top_k)
    results = []
    for idx in indices[0]:
        if idx < len(_catalog):
            results.append(_catalog[idx])
    return results


def get_by_names(names: List[str]) -> List[Dict[str, Any]]:
    """Fetch specific assessments by name (case-insensitive partial match)."""
    global _catalog
    if not _catalog:
        _catalog = _load_catalog()

    found = []
    for name in names:
        name_lower = name.lower().strip()
        for item in _catalog:
            if name_lower in item["name"].lower() or item["name"].lower() in name_lower:
                found.append(item)
                break
    return found


def get_all() -> List[Dict[str, Any]]:
    global _catalog
    if not _catalog:
        _catalog = _load_catalog()
    return _catalog
