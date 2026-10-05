import pytest

from mcp_tool_router.router import (
    EXACT,
    FUZZY,
    PREFIX,
    STOPWORDS,
    match_strength,
    tokenize,
)


class TestTokenize:
    def test_lowercases_and_splits_on_punctuation(self):
        assert tokenize("Top-Selling ITEMS, today!") == ["top", "selling", "items", "today"]

    def test_splits_snake_case(self):
        assert tokenize("get_sales_by_category") == ["sales", "category"]

    def test_splits_camel_and_pascal_case(self):
        assert tokenize("getSalesByCategory") == ["sales", "category"]
        assert tokenize("HTTPServerError") == ["http", "server", "error"]

    def test_drops_stopwords(self):
        assert tokenize("what are the sales for my store") == ["sales", "store"]

    def test_drops_single_characters(self):
        assert tokenize("a b c sales x") == ["sales"]

    def test_keeps_multi_digit_numbers(self):
        assert tokenize("last 30 days 7") == ["last", "30", "days"]

    def test_unicode_is_normalized(self):
        # Full-width letters and the "fi" ligature normalize to plain ASCII words.
        assert tokenize("ｓａｌｅｓ ﬁling") == ["sales", "filing"]  # noqa: RUF001

    def test_words_with_combining_marks_stay_whole(self):
        # Devanagari vowel signs are combining marks; they must not split words.
        assert tokenize("बिक्री रिपोर्ट") == ["बिक्री", "रिपोर्ट"]
        assert tokenize("café naïve") == ["café", "naïve"]

    @pytest.mark.parametrize("text", ["", "   ", "the a of", "!!! ??? ..."])
    def test_empty_or_filler_only(self, text):
        assert tokenize(text) == []

    def test_stopwords_are_lowercase_words(self):
        assert all(word == word.casefold() and word.isalpha() for word in STOPWORDS)


class TestMatchStrength:
    def test_exact(self):
        assert match_strength("sales", "sales") == EXACT

    @pytest.mark.parametrize(
        ("query", "doc"), [("sale", "sales"), ("sales", "sale"), ("cat", "category")]
    )
    def test_prefix_either_direction(self, query, doc):
        assert match_strength(query, doc) == PREFIX

    def test_prefix_needs_three_characters(self):
        assert match_strength("ca", "category") == 0.0

    @pytest.mark.parametrize(
        ("query", "doc"), [("inventroy", "inventory"), ("categry", "category")]
    )
    def test_fuzzy_catches_typos(self, query, doc):
        assert match_strength(query, doc) == FUZZY

    def test_fuzzy_skips_short_words(self):
        # "taxe"/"taxi" are similar but too short to trust a fuzzy match.
        assert match_strength("taxe", "taxi") == 0.0

    def test_fuzzy_rejects_unrelated_words(self):
        assert match_strength("weather", "employee") == 0.0

    def test_fuzzy_rejects_very_different_lengths(self):
        assert match_strength("inventory", "inventoryreconciliation"[:5] + "zzzzzzzzzzzz") == 0.0

    def test_order_of_strengths(self):
        assert EXACT > PREFIX > FUZZY > 0

    def test_exact_mode_disables_prefix_and_fuzzy(self):
        assert match_strength("sales", "sales", "exact") == EXACT
        assert match_strength("sale", "sales", "exact") == 0.0
        assert match_strength("inventroy", "inventory", "exact") == 0.0

    def test_prefix_mode_disables_fuzzy_only(self):
        assert match_strength("sale", "sales", "prefix") == PREFIX
        assert match_strength("inventroy", "inventory", "prefix") == 0.0
