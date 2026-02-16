"""Query preprocessing: normalization, expansion, and Hebrew handling."""

import re


def normalize_hebrew(text: str) -> str:
    """Normalize Hebrew text for consistent retrieval.

    Handles:
    - Remove niqqud (vowel marks)
    - Normalize final-form letters (sofiot)
    - Normalize common spelling variations
    """
    # Remove niqqud (Hebrew diacritical marks)
    text = re.sub(r"[\u0591-\u05C7]", "", text)

    # Normalize final-form letters to regular form
    finals_map = {
        "\u05da": "\u05db",  # ך -> כ
        "\u05dd": "\u05de",  # ם -> מ
        "\u05df": "\u05e0",  # ן -> נ
        "\u05e3": "\u05e4",  # ף -> פ
        "\u05e5": "\u05e6",  # ץ -> צ
    }
    for final, regular in finals_map.items():
        text = text.replace(final, regular)

    return text


def clean_query(query: str) -> str:
    """Basic query cleaning."""
    # Remove excessive whitespace
    query = re.sub(r"\s+", " ", query).strip()
    # Remove trailing punctuation that might hurt retrieval
    query = query.rstrip("?!.。")
    return query


def process_query(query: str) -> str:
    """Full query preprocessing pipeline."""
    query = clean_query(query)
    # Don't normalize Hebrew for the actual embedding query
    # (the model handles it), but use normalized form for BM25
    return query


def get_normalized_query(query: str) -> str:
    """Get a normalized version for BM25/sparse search."""
    query = clean_query(query)
    query = normalize_hebrew(query)
    return query.lower()
