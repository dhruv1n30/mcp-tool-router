"""Property-based tests: invariants that must hold for any catalog and query."""

from hypothesis import given, settings
from hypothesis import strategies as st

from mcp_tool_router import ToolRouter, tokenize

WORDS = st.text(alphabet="abcdefghijklmnop", min_size=2, max_size=8)
TOOL = st.builds(
    lambda name, words: {"name": name, "description": " ".join(words)},
    WORDS,
    st.lists(WORDS, max_size=6),
)
CATALOG = st.lists(TOOL, max_size=12, unique_by=lambda t: t["name"])
QUERY = st.lists(WORDS, max_size=5).map(" ".join)


@st.composite
def routed(draw):
    catalog = draw(CATALOG)
    pinned = (
        draw(st.lists(st.sampled_from([t["name"] for t in catalog]), unique=True))
        if catalog
        else []
    )
    router = ToolRouter(
        catalog,
        max_tools=draw(st.integers(1, 5)),
        always_include=pinned,
        match=draw(st.sampled_from(["exact", "prefix", "fuzzy"])),
        on_no_match=draw(st.sampled_from(["all", "pinned"])),
    )
    return router, draw(QUERY)


@settings(max_examples=200, deadline=None)
@given(routed())
def test_selection_is_an_ordered_subset_of_the_catalog(case):
    router, query = case
    positions = [
        next(i for i, t in enumerate(router.tools) if t is s) for s in router.select(query)
    ]
    assert positions == sorted(set(positions))


@settings(max_examples=200, deadline=None)
@given(routed())
def test_selection_size_is_bounded_unless_failing_open(case):
    router, query = case
    selected = router.select(query)
    if len(selected) != len(router.tools) or router.on_no_match == "pinned":
        assert len(selected) <= router.max_tools + len(router.always_include)


@settings(max_examples=200, deadline=None)
@given(routed())
def test_pinned_tools_are_always_selected(case):
    router, query = case
    selected = {t["name"] for t in router.select(query)}
    assert router.always_include <= selected


@settings(max_examples=200, deadline=None)
@given(routed())
def test_rank_scores_are_positive_descending_and_unique(case):
    router, query = case
    ranking = router.rank(query)
    scores = [score for _, score in ranking]
    assert all(score > 0 for score in scores)
    assert scores == sorted(scores, reverse=True)
    assert len({name for name, _ in ranking}) == len(ranking)


@settings(max_examples=200, deadline=None)
@given(CATALOG, QUERY)
def test_routing_is_deterministic(catalog, query):
    assert ToolRouter(catalog).rank(query) == ToolRouter(catalog).rank(query)


@settings(max_examples=200, deadline=None)
@given(CATALOG, QUERY)
def test_looser_matching_never_loses_a_tool(catalog, query):
    found = {
        mode: {n for n, _ in ToolRouter(catalog, match=mode).rank(query)}
        for mode in ("exact", "prefix", "fuzzy")
    }
    assert found["exact"] <= found["prefix"] <= found["fuzzy"]


@settings(max_examples=200, deadline=None)
@given(CATALOG)
def test_a_tools_own_name_finds_it(catalog):
    for tool in catalog:
        if tokenize(tool["name"]):
            assert tool["name"] in {n for n, _ in ToolRouter(catalog).rank(tool["name"])}


@settings(max_examples=300, deadline=None)
@given(st.text())
def test_tokenize_is_idempotent(text):
    words = tokenize(text)
    assert tokenize(" ".join(words)) == words
