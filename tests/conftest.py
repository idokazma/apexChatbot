"""Shared test fixtures and mock modules for missing heavy dependencies."""

import sys
import types
from unittest.mock import MagicMock

# Mock heavy dependencies that aren't installed in test environment.
# These modules are imported transitively by source modules but aren't needed
# for the unit-level logic under test.

_MOCK_MODULES = [
    "sentence_transformers",
    "chromadb",
    "langdetect",
    "playwright",
    "playwright.async_api",
    "anthropic",
    "docling",
    "gradio",
    "tqdm",
]

for mod_name in _MOCK_MODULES:
    if mod_name not in sys.modules:
        mock_mod = types.ModuleType(mod_name)
        # Add common attributes so `from X import Y` works
        mock_mod.__dict__.update({
            "SentenceTransformer": MagicMock,
            "CrossEncoder": MagicMock,
            "detect": lambda text, *a, **kw: "he",  # langdetect.detect stub
            "async_playwright": MagicMock,
            "Anthropic": MagicMock,
            "PersistentClient": MagicMock,
            "ClientAPI": MagicMock,
            "tqdm": lambda x, **kw: x,
        })
        sys.modules[mod_name] = mock_mod
