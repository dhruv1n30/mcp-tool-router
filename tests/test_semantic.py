"""Semantic routing: embeddings fused with BM25 by Reciprocal Rank Fusion."""

import sys
import types
import zlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mcp_tool_router import ToolRouter, embedders, tokenize
from mcp_tool_router.cli import main

# Tools whose descriptions never say "revenue", "money" or "merchandise".
TOOLS = [
    {"name": "get_sales_data", "description": "Sales totals and order counts for a date range."},
    {"name": "get_inventory_value", "description": "Total value of stock on hand."},
    {"name": "get_employee_hours", "description": "Hours worked per employee, from the timesheet."},
    {"name": "get_weather_forecast", "description": "Weather forecast for a city."},
    {"name": "send_email", "description": "Send an email message to a recipient."},
]

CONCEPTS = [
    {"sales", "revenue", "income", "earnings", "money", "turnover", "order"},
    {"stock", "inventory", "merchandise", "goods"},
    {"hours", "worked", "overtime", "timesheet", "shift"},
    {"weather", "rain", "forecast", "temperature"},
    {"email", "mail", "notify", "message", "send"},
]


class ConceptEmbedder:
    """Deterministic stand-in for a real model: one dimension per concept."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts, *, query):
        self.calls.append((list(texts), query))
        return [
            [sum(word in concept for word in tokenize(t)) for concept in CONCEPTS] for t in texts
        ]


def names(tools):
    return [t["name"] for t in tools]


def ranked(router, query, **kwargs):
    return [name for name, _ in router.rank(query, **kwargs)]


class TestMeaning:
    def test_keywords_alone_cannot_find_revenue(self):
        assert ToolRouter(TOOLS).rank("what is revenue in last 5 days") == []

    def test_embeddings_find_revenue(self):
        router = ToolRouter(TOOLS, max_tools=1, embed=ConceptEmbedder())
        assert names(router.select("what is revenue in last 5 days")) == ["get_sales_data"]

    @pytest.mark.parametrize(
        ("question", "tool"),
        [
            ("how much money did we make", "get_sales_data"),
            ("what is our merchandise worth", "get_inventory_value"),
            ("who did overtime", "get_employee_hours"),
            ("will it rain tomorrow", "get_weather_forecast"),
            ("notify the manager by mail", "send_email"),
        ],
    )
    def test_meaning_only_questions(self, question, tool):
        assert ranked(ToolRouter(TOOLS, embed=ConceptEmbedder()), question)[0] == tool


class TestFusion:
    def test_keyword_and_meaning_agreeing_scores_both_lists(self):
        ranking = ToolRouter(TOOLS, embed=ConceptEmbedder()).rank("sales")
        assert ranking[0] == ("get_sales_data", pytest.approx(2 / 61))

    def test_a_tool_found_by_both_beats_one_found_by_either(self):
        # "stock" is a keyword hit for inventory; "revenue" is a meaning hit for sales.
        # "sales revenue" hits sales both ways, so it must lead.
        ranking = ranked(ToolRouter(TOOLS, embed=ConceptEmbedder()), "sales revenue stock")
        assert ranking[0] == "get_sales_data"
        assert set(ranking[:2]) == {"get_sales_data", "get_inventory_value"}

    def test_rrf_k_changes_the_scores(self):
        ranking = ToolRouter(TOOLS, embed=ConceptEmbedder(), rrf_k=10).rank("sales")
        assert ranking[0][1] == pytest.approx(2 / 11)

    def test_semantic_candidates_are_capped_at_the_budget(self):
        # Every concept word appears, so every tool is semantically related.
        question = "revenue merchandise overtime rain mail"
        assert len(ToolRouter(TOOLS, max_tools=2, embed=ConceptEmbedder()).rank(question)) == 2
        pinned = ToolRouter(
            TOOLS, max_tools=2, always_include=["send_email"], embed=ConceptEmbedder()
        )
        assert len(pinned.rank(question)) == 3

    def test_unrelated_meaning_is_not_a_match(self):
        # No concept words: zero vector, zero similarity, nothing proposed.
        router = ToolRouter(TOOLS, embed=ConceptEmbedder())
        assert router.rank("quantum chromodynamics") == []
        assert router.select("quantum chromodynamics") == TOOLS  # fails open

    def test_min_similarity_floor(self):
        question = "revenue and something about rain"
        low = ToolRouter(TOOLS, embed=ConceptEmbedder(), min_similarity=0.1)
        high = ToolRouter(TOOLS, embed=ConceptEmbedder(), min_similarity=0.99)
        assert set(ranked(low, question)) == {"get_sales_data", "get_weather_forecast"}
        assert ranked(high, question) == []

    def test_history_reaches_the_embedder(self):
        embed = ConceptEmbedder()
        router = ToolRouter(TOOLS, embed=embed)
        assert ranked(router, "same again please", history=["our revenue"])[0] == "get_sales_data"
        assert embed.calls[-1] == (["same again please our revenue"], True)


class TestEmbedderCalls:
    def test_catalog_is_embedded_once_lazily_as_documents(self):
        embed = ConceptEmbedder()
        router = ToolRouter(TOOLS, embed=embed)
        assert embed.calls == []  # nothing at construction
        router.rank("revenue")
        router.rank("stock")
        document_calls = [texts for texts, query in embed.calls if not query]
        assert len(document_calls) == 1 and len(document_calls[0]) == len(TOOLS)
        assert "get sales data Sales totals" in document_calls[0][0]  # readable, not snake_case

    def test_repeated_questions_reuse_the_query_vector(self):
        embed = ConceptEmbedder()
        router = ToolRouter(TOOLS, embed=embed)
        for _ in range(3):
            router.select("revenue")
        router.select("stock")
        assert [texts for texts, query in embed.calls if query] == [["revenue"], ["stock"]]

    def test_query_vector_cache_is_bounded(self):
        router = ToolRouter(TOOLS, embed=ConceptEmbedder())
        for i in range(300):
            router.rank(f"revenue {i:03d}")
        assert len(router._query_vectors) == 256

    def test_filler_only_question_never_calls_the_embedder(self):
        embed = ConceptEmbedder()
        ToolRouter(TOOLS, embed=embed).rank("what is the")
        assert embed.calls == []

    def test_empty_catalog_never_calls_the_embedder(self):
        embed = ConceptEmbedder()
        assert ToolRouter([], embed=embed).rank("revenue") == []
        assert embed.calls == []

    def test_lexical_only_router_never_needs_an_embedder(self):
        assert ToolRouter(TOOLS).embed is None


class TestBadEmbedders:
    def test_not_callable(self):
        with pytest.raises(TypeError, match="callable"):
            ToolRouter(TOOLS, embed="openai")

    def test_wrong_number_of_vectors(self):
        router = ToolRouter(TOOLS, embed=lambda texts, *, query: [[1.0, 0.0]])
        with pytest.raises(ValueError, match="returned 1 vectors for 5 texts"):
            router.rank("sales")

    def test_ragged_vectors(self):
        def embed(texts, *, query):
            return [[1.0] * (i + 1) for i in range(len(texts))]

        with pytest.raises(ValueError, match="different lengths"):
            ToolRouter(TOOLS, embed=embed).rank("sales")

    def test_query_and_document_dimensions_differ(self):
        def embed(texts, *, query):
            return [[1.0, 0.0, 0.0] if query else [1.0, 0.0] for _ in texts]

        with pytest.raises(ValueError, match="query and tool vectors"):
            ToolRouter(TOOLS, embed=embed).rank("sales")

    def test_numpy_like_vectors_are_accepted(self):
        array = pytest.importorskip("numpy").array

        def embed(texts, *, query):
            return array([[1.0, 0.0]] * len(texts))

        assert ranked(ToolRouter(TOOLS, embed=embed), "sales")[0] == "get_sales_data"


# Property tests: the routing invariants still hold with an embedder in play.
WORDS = st.text(alphabet="abcdefghijklmnop", min_size=2, max_size=8)
CATALOG = st.lists(
    st.builds(
        lambda n, d: {"name": n, "description": " ".join(d)}, WORDS, st.lists(WORDS, max_size=6)
    ),
    max_size=10,
    unique_by=lambda t: t["name"],
)


def hashing_embedder(texts, *, query):
    vectors = []
    for text in texts:
        vector = [0.0] * 16
        for word in tokenize(text):
            vector[zlib.crc32(word.encode()) % 16] += 1
        vectors.append(vector)
    return vectors


@settings(max_examples=200, deadline=None)
@given(CATALOG, st.lists(WORDS, max_size=5).map(" ".join), st.integers(1, 4))
def test_invariants_with_embeddings(catalog, query, max_tools):
    router = ToolRouter(catalog, max_tools=max_tools, embed=hashing_embedder)
    ranking = router.rank(query)
    scores = [score for _, score in ranking]
    assert scores == sorted(scores, reverse=True) and all(s > 0 for s in scores)
    assert {n for n, _ in ToolRouter(catalog).rank(query)} <= {n for n, _ in ranking}
    selected = router.select(query)
    positions = [next(i for i, t in enumerate(catalog) if t is s) for s in selected]
    assert positions == sorted(set(positions))
    if ranking:
        assert len(selected) <= max_tools


class TestFastembedEmbedder:
    def test_uses_query_and_passage_embedding(self, monkeypatch):
        seen = {}

        class Vector(list):
            def tolist(self):
                return list(self)

        class TextEmbedding:
            def __init__(self, model, **options):
                seen["init"] = (model, options)

            def query_embed(self, texts):
                seen["query"] = texts
                return (Vector([1.0, 0.0]) for _ in texts)

            def passage_embed(self, texts):
                seen["passage"] = texts
                return (Vector([0.0, 1.0]) for _ in texts)

        monkeypatch.setitem(
            sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=TextEmbedding)
        )
        embed = embedders.fastembed_embedder(threads=2)
        assert seen["init"] == (embedders.DEFAULT_MODEL, {"threads": 2})
        assert embed(("a", "b"), query=True) == [[1.0, 0.0], [1.0, 0.0]]
        assert embed(["c"], query=False) == [[0.0, 1.0]]
        assert (seen["query"], seen["passage"]) == (["a", "b"], ["c"])

    def test_missing_fastembed_raises_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "fastembed", None)
        with pytest.raises(ImportError):
            embedders.fastembed_embedder()


class TestCli:
    @pytest.fixture
    def files(self, tmp_path):
        import json

        tools = tmp_path / "tools.json"
        tools.write_text(json.dumps(TOOLS), encoding="utf-8")
        cases = tmp_path / "cases.csv"
        cases.write_text(
            "question,expected\nwhat is revenue in last 5 days,get_sales_data\n"
            "how much money did we make,get_sales_data\n",
            encoding="utf-8",
        )
        return tools, cases

    @pytest.fixture
    def fake_model(self, monkeypatch):
        models = []

        def factory(model=embedders.DEFAULT_MODEL, **options):
            models.append(model)
            return ConceptEmbedder()

        monkeypatch.setattr(embedders, "fastembed_embedder", factory)
        return models

    def test_semantic_flag_turns_fail_open_into_real_routing(self, files, fake_model, capsys):
        tools, cases = files
        base = ["eval", "--tools", str(tools), "--cases", str(cases), "--max-tools", "1"]
        # Keywords alone: "recall" is perfect only because nothing matched and the
        # whole catalog was sent. The no-match and payload lines expose it.
        assert main(base) == 0
        out = capsys.readouterr().out
        assert "no match         2 questions" in out and "0% smaller" in out
        assert main([*base, "--semantic"]) == 0
        out = capsys.readouterr().out
        assert "recall           100.0%" in out and "tools per query  1.0 of 5" in out
        assert "no match         0 questions" in out
        assert fake_model == [embedders.DEFAULT_MODEL]

    def test_semantic_flag_takes_a_model_name(self, files, fake_model):
        tools, cases = files
        main(
            [
                "eval",
                "--tools",
                str(tools),
                "--cases",
                str(cases),
                "--semantic",
                "BAAI/bge-small-en-v1.5",
            ]
        )
        assert fake_model == ["BAAI/bge-small-en-v1.5"]

    def test_min_similarity_flag(self, files, fake_model, capsys):
        tools, cases = files
        args = ["eval", "--tools", str(tools), "--cases", str(cases), "--semantic"]
        main([*args, "--min-similarity", "0.999", "--max-tools", "1"])
        assert "no match         0" in capsys.readouterr().out  # 1.0 cosine passes the floor

    def test_missing_fastembed_explains_the_extra(self, files, monkeypatch, capsys):
        tools, cases = files
        monkeypatch.setitem(sys.modules, "fastembed", None)
        assert main(["eval", "--tools", str(tools), "--cases", str(cases), "--semantic"]) == 2
        assert "mcp-tool-router[semantic]" in capsys.readouterr().err

    def test_serve_passes_the_embedder_to_the_transform(self, tmp_path, fake_model, monkeypatch):
        pytest.importorskip("fastmcp")
        from fastmcp.server.server import FastMCP

        from mcp_tool_router.transform import RouterSearchTransform

        proxies = []
        monkeypatch.setattr(FastMCP, "run", lambda self, **kw: proxies.append(self))
        config = tmp_path / "mcp.json"
        config.write_text('{"mcpServers": {"x": {"command": "x"}}}', encoding="utf-8")
        assert (
            main(["serve", "--config", str(config), "--semantic", "--min-similarity", "0.3"]) == 0
        )
        (transform,) = [t for t in proxies[0].transforms if isinstance(t, RouterSearchTransform)]
        assert isinstance(transform._embed, ConceptEmbedder)
        assert transform._min_similarity == 0.3


async def test_fastmcp_transform_searches_by_meaning():
    pytest.importorskip("fastmcp")
    from fastmcp import Client, FastMCP

    from mcp_tool_router import RouterSearchTransform

    mcp = FastMCP("semantic")

    @mcp.tool
    def get_sales_data(start_date: str) -> str:
        """Sales totals and order counts for a date range."""
        return "ok"

    @mcp.tool
    def get_weather_forecast(city: str) -> str:
        """Weather forecast for a city."""
        return "ok"

    mcp.add_transform(RouterSearchTransform(embed=ConceptEmbedder()))
    async with Client(mcp) as client:
        result = await client.call_tool("search_tools", {"query": "what is revenue in last 5 days"})
        assert [tool["name"] for tool in result.data] == ["get_sales_data"]


@pytest.mark.model
@pytest.mark.parametrize(
    ("question", "tool"),
    [
        ("what is revenue in last 5 day", "get_sales_data"),
        ("how much money did we make this week", "get_sales_data"),
        ("what are our earnings today", "get_sales_data"),
        ("how much is our merchandise worth", "get_inventory_value"),
        ("who worked overtime", "get_employee_hours"),
        ("is it going to rain in Pune", "get_weather_forecast"),
        ("notify the manager by mail", "send_email"),
    ],
)
def test_real_model_routes_by_meaning(question, tool, real_router):
    assert tool in names(real_router.select(question))


@pytest.fixture(scope="module")
def real_router():
    pytest.importorskip("fastembed")
    return ToolRouter(TOOLS, max_tools=2, embed=embedders.fastembed_embedder())
