"""
Retrieval module: loads catalog.json, builds a TF-IDF index on first use,
and provides keyword search over SHL assessments.
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Lazy imports — loaded once at startup
_vectorizer = None
_tfidf_matrix = None
_catalog: List[Dict[str, Any]] = []
_texts: List[str] = []


def _get_catalog_path() -> Path:
    here = Path(__file__).parent.parent
    return here / "data" / "catalog.json"


def _load_catalog() -> List[Dict[str, Any]]:
    path = _get_catalog_path()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_text(item: Dict[str, Any]) -> str:
    """Combine fields into a single string for TF-IDF."""
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
    """Build the TF-IDF index. Called once at server startup."""
    global _vectorizer, _tfidf_matrix, _catalog, _texts

    _catalog = _load_catalog()
    _texts = [_build_text(item) for item in _catalog]

    _vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(1, 2)
    )
    _tfidf_matrix = _vectorizer.fit_transform(_texts)

    print(f"[retrieval] Indexed {len(_catalog)} assessments with TF-IDF.")


def search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Return top_k catalog items for a query string using TF-IDF."""
    global _vectorizer, _tfidf_matrix, _catalog, _texts

    if _vectorizer is None:
        initialize()

    # Transform query to TF-IDF
    query_vec = _vectorizer.transform([query])

    # Compute cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_vec, _tfidf_matrix).flatten()

    # Get top-k indices
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
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
