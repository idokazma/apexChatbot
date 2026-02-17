"""Tests for retrieval.query_processor."""

from retrieval.query_processor import (
    clean_query,
    get_normalized_query,
    normalize_hebrew,
    process_query,
)


class TestNormalizeHebrew:
    def test_removes_niqqud(self):
        # שָׁלוֹם -> שלום (shalom with niqqud -> without)
        text_with_niqqud = "שָׁלוֹם"
        result = normalize_hebrew(text_with_niqqud)
        # After removing niqqud marks, only base letters remain
        assert "\u05B8" not in result  # kamatz removed
        assert "\u05C1" not in result  # shin dot removed

    def test_normalizes_final_letters(self):
        # ך -> כ, ם -> מ, ן -> נ, ף -> פ, ץ -> צ
        text = "ך ם ן ף ץ"
        result = normalize_hebrew(text)
        assert "ך" not in result
        assert "ם" not in result
        assert "ן" not in result
        assert "ף" not in result
        assert "ץ" not in result
        assert "כ" in result
        assert "מ" in result
        assert "נ" in result
        assert "פ" in result
        assert "צ" in result

    def test_preserves_regular_letters(self):
        # Note: normalize_hebrew converts ALL final-form letters to regular,
        # including word-final ם→מ and ן→נ. This is intended for search normalization.
        text = "שלום עולם"
        result = normalize_hebrew(text)
        # ם (final mem) -> מ (regular mem)
        assert result == "שלומ עולמ"

    def test_english_passthrough(self):
        text = "hello world"
        assert normalize_hebrew(text) == "hello world"


class TestCleanQuery:
    def test_strips_whitespace(self):
        assert clean_query("  hello  world  ") == "hello world"

    def test_collapses_multiple_spaces(self):
        assert clean_query("hello   world") == "hello world"

    def test_removes_trailing_punctuation(self):
        assert clean_query("what is insurance?") == "what is insurance"
        assert clean_query("tell me more!") == "tell me more"
        assert clean_query("hello.") == "hello"

    def test_preserves_internal_punctuation(self):
        result = clean_query("what is the cost? of insurance?")
        assert result == "what is the cost? of insurance"

    def test_empty_string(self):
        assert clean_query("") == ""

    def test_only_whitespace(self):
        assert clean_query("   ") == ""


class TestProcessQuery:
    def test_cleans_query(self):
        result = process_query("  what is insurance?  ")
        assert result == "what is insurance"

    def test_preserves_hebrew(self):
        result = process_query("מה זה ביטוח רכב?")
        assert "ביטוח" in result
        assert "רכב" in result


class TestGetNormalizedQuery:
    def test_lowercases(self):
        result = get_normalized_query("HELLO WORLD")
        assert result == "hello world"

    def test_normalizes_hebrew_finals(self):
        result = get_normalized_query("שלום")
        # ם at end gets normalized to מ
        assert "ם" not in result

    def test_cleans_and_normalizes(self):
        result = get_normalized_query("  What is insurance?  ")
        assert result == "what is insurance"
