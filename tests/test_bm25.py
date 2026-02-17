"""Tests for retrieval.bm25_search tokenizer and BM25Index."""

from retrieval.bm25_search import BM25Index, _tokenize


class TestTokenize:
    def test_english(self):
        tokens = _tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_hebrew(self):
        tokens = _tokenize("שלום עולם")
        assert len(tokens) == 2
        assert "שלום" in tokens
        assert "עולם" in tokens

    def test_mixed_language(self):
        tokens = _tokenize("Hello שלום World עולם")
        assert len(tokens) == 4

    def test_punctuation_removed(self):
        tokens = _tokenize("Hello, World! How are you?")
        assert "hello" in tokens
        assert "world" in tokens
        assert "," not in " ".join(tokens)

    def test_lowercased(self):
        tokens = _tokenize("HELLO WORLD")
        assert tokens == ["hello", "world"]

    def test_empty_string(self):
        # After regex substitution, empty string splits to []
        assert _tokenize("") == []

    def test_preserves_hebrew_chars(self):
        tokens = _tokenize("ביטוח-רכב")
        # Hyphen removed, but Hebrew chars preserved
        assert any("ביטוח" in t for t in tokens)


class TestBM25Index:
    def test_not_built_initially(self):
        idx = BM25Index()
        assert idx.is_built is False

    def test_search_returns_empty_when_not_built(self):
        idx = BM25Index()
        assert idx.search("hello") == []

    def test_search_empty_tokens(self):
        idx = BM25Index()
        assert idx.search("") == []
