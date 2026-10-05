"""Per-query tool routing: rank a tool catalog against a question.

Ranking is BM25 over the tools' words, optionally fused with embedding
similarity for semantic matches ("revenue" finding a sales tool). Stdlib only:
embeddings come from a function you pass in.

Accepts tools in MCP, OpenAI, Anthropic and Gemini shapes (dicts or objects) and
returns the caller's own tool objects, so the result can be passed straight back
to whichever SDK the tools came from.
"""

from __future__ import annotations

import difflib
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

MatchMode = Literal["exact", "prefix", "fuzzy"]
NoMatchPolicy = Literal["all", "pinned"]


class Embedder(Protocol):
    """Turns texts into vectors. ``query`` is True for questions, False for tool
    descriptions; models that embed the two differently (BGE, Arctic, E5) use it,
    others can ignore it."""

    def __call__(self, texts: Sequence[str], *, query: bool) -> Sequence[Sequence[float]]: ...


# Filler words that appear in questions but carry no routing signal.
STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are as at be been before being
    below between both but by can could did do does doing down during each few for
    from further had has have having he her here hers him his how i if in into is it
    its just me more most my no nor not now of off on once only or other our ours out
    over own please same she should show so some such tell than that the their theirs
    them then there these they this those through to too under until up us very was
    we were what when where which while who whom why will with would you your yours
    give want need know let lets like get find see
    """.split()  # noqa: SIM905 - a word block reads better than 130 quoted strings
)

# Strength of each kind of term match. Exact beats prefix beats fuzzy, so a tool
# that spells the user's word exactly outranks one that only resembles it.
EXACT, PREFIX, FUZZY = 1.0, 0.9, 0.7
MIN_PREFIX_LEN = 3
MIN_FUZZY_LEN = 5
FUZZY_RATIO = 0.8

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _is_word_char(ch: str) -> bool:
    # Combining marks (category M*) belong to their word: re's \w excludes them,
    # which would shred Devanagari, Bengali, Tamil etc. into single letters.
    return ch.isalnum() or unicodedata.category(ch).startswith("M")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase content words.

    Handles snake_case and camelCase identifiers, normalizes Unicode (NFKC +
    casefold), keeps combining marks inside words, and drops stopwords and
    single characters.
    """
    text = _CAMEL.sub(" ", unicodedata.normalize("NFKC", text)).casefold()
    words = "".join(ch if _is_word_char(ch) else " " for ch in text).split()
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


def match_strength(query_word: str, doc_word: str, mode: MatchMode = "fuzzy") -> float:
    """How strongly a query word matches a document word (0.0 = no match)."""
    if query_word == doc_word:
        return EXACT
    if mode == "exact":
        return 0.0
    if min(len(query_word), len(doc_word)) >= MIN_PREFIX_LEN and (
        doc_word.startswith(query_word) or query_word.startswith(doc_word)
    ):
        return PREFIX
    if mode == "prefix" or min(len(query_word), len(doc_word)) < MIN_FUZZY_LEN:
        return 0.0
    # ratio >= 0.8 is impossible when one word is under 2/3 the other's length,
    # so skip the expensive SequenceMatcher for those pairs.
    if 3 * min(len(query_word), len(doc_word)) < 2 * max(len(query_word), len(doc_word)):
        return 0.0
    matcher = difflib.SequenceMatcher(None, query_word, doc_word)
    if matcher.real_quick_ratio() < FUZZY_RATIO or matcher.quick_ratio() < FUZZY_RATIO:
        return 0.0
    return FUZZY if matcher.ratio() >= FUZZY_RATIO else 0.0


def _field(obj: Any, *keys: str) -> Any:
    """First non-None value among keys, read from a mapping or as attributes."""
    for key in keys:
        value = obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)
        if value is not None:
            return value
    return None


def describe_tool(tool: Any) -> tuple[str, str]:
    """Return (name, searchable text) for a tool in any supported shape.

    Supported: MCP (``inputSchema``), Anthropic (``input_schema``), OpenAI chat
    (``{"type": "function", "function": {...}}``), OpenAI responses / Gemini
    (``parameters``), as dicts or objects such as ``mcp.types.Tool``. The text
    covers the name, description, top-level parameter names, their descriptions
    and any enum values (so enum-dispatch tools like ``get_metric(metric="tax")``
    are findable by their enum members).
    """
    if not isinstance(tool, Mapping) and not hasattr(tool, "name"):
        raise TypeError(f"Unsupported tool type: {type(tool).__name__}")
    spec = _field(tool, "function") if isinstance(tool, Mapping) else None
    spec = spec if isinstance(spec, Mapping) else tool
    name = _field(spec, "name")
    if not isinstance(name, str) or not name:
        raise TypeError(f"Tool has no name: {tool!r}")

    parts = [name, _field(spec, "description") or ""]
    # input_schema first: MCP SDK v2 objects warn when the old inputSchema name is read.
    schema = _field(spec, "input_schema", "inputSchema", "parameters") or {}
    properties = _field(schema, "properties") or {}
    for param_name, param in properties.items():
        parts.append(str(param_name))
        parts.append(str(_field(param, "description") or ""))
        enum = _field(param, "enum") or ()
        parts.extend(str(value) for value in enum)
    return name, " ".join(parts)


@dataclass(frozen=True)
class _Doc:
    name: str
    words: frozenset[str]
    text: str  # natural-language form, for embedding


def _unit_vectors(vectors: Sequence[Sequence[float]], expected: int) -> list[list[float]]:
    """Validate embedder output and scale every vector to length 1."""
    vectors = [[float(x) for x in vector] for vector in vectors]
    if len(vectors) != expected:
        raise ValueError(f"embedder returned {len(vectors)} vectors for {expected} texts")
    if len({len(vector) for vector in vectors}) > 1:
        raise ValueError("embedder returned vectors of different lengths")
    units = []
    for vector in vectors:
        norm = math.sqrt(sum(x * x for x in vector))
        units.append([x / norm for x in vector] if norm else vector)
    return units


class ToolRouter:
    """Select the tools relevant to a question before calling an LLM.

    Example::

        router = ToolRouter(tools, max_tools=8, always_include=["get_user_info"])
        relevant = router.select("top selling items last week")
        # pass `relevant` to the model instead of `tools`

    Args:
        tools: The full catalog, in any shape ``describe_tool`` accepts.
        max_tools: Most non-pinned tools returned per query.
        always_include: Tool names returned on every query, outside the cap.
        match: ``"exact"`` words only, ``"prefix"`` also matches "sale"/"sales",
            ``"fuzzy"`` also tolerates typos like "inventroy".
        on_no_match: What ``select`` returns when nothing matches: ``"all"``
            fails open to the whole catalog (safe), ``"pinned"`` returns only
            ``always_include`` (cheap).
        embed: Optional :class:`Embedder` for semantic matching. Tool vectors are
            computed once, on the first query. Rankings are then fused with
            Reciprocal Rank Fusion, so keyword hits and meaning-only matches both
            count. See ``mcp_tool_router.embedders`` for a local default.
        min_similarity: With ``embed``, ignore semantic matches below this cosine
            similarity. ``None`` (default) keeps the nearest tools with any
            positive similarity, so semantic routing nearly always proposes
            something. Set a floor (tune it per model with ``eval``) to let
            unrelated questions fall through to ``on_no_match`` instead.
        k1, b: Standard BM25 parameters.
        rrf_k: Reciprocal Rank Fusion constant (60 is the usual choice).
    """

    def __init__(
        self,
        tools: Iterable[Any],
        *,
        max_tools: int = 12,
        always_include: Iterable[str] = (),
        match: MatchMode = "fuzzy",
        on_no_match: NoMatchPolicy = "all",
        embed: Embedder | None = None,
        min_similarity: float | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        rrf_k: int = 60,
    ) -> None:
        if max_tools < 1:
            raise ValueError("max_tools must be at least 1")
        if match not in ("exact", "prefix", "fuzzy"):
            raise ValueError(f"match must be 'exact', 'prefix' or 'fuzzy', not {match!r}")
        if on_no_match not in ("all", "pinned"):
            raise ValueError(f"on_no_match must be 'all' or 'pinned', not {on_no_match!r}")
        if embed is not None and not callable(embed):
            raise TypeError("embed must be callable")

        self.tools: list[Any] = list(tools)
        self.max_tools = max_tools
        self.match: MatchMode = match
        self.on_no_match: NoMatchPolicy = on_no_match
        self.embed = embed
        self.min_similarity = min_similarity
        self.k1, self.b, self.rrf_k = k1, b, rrf_k
        self._vectors: list[list[float]] | None = None
        self._query_vectors: dict[str, list[float]] = {}

        self._docs: list[_Doc] = []
        for tool in self.tools:
            name, text = describe_tool(tool)
            natural = " ".join(_CAMEL.sub(" ", text).replace("_", " ").split())
            self._docs.append(_Doc(name, frozenset(tokenize(text)), natural))
        names = [doc.name for doc in self._docs]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"Duplicate tool names: {duplicates}")

        self.always_include = frozenset(always_include)
        unknown = sorted(self.always_include - set(names))
        if unknown:
            raise ValueError(f"always_include names not in the catalog: {unknown}")

        self._vocabulary = frozenset().union(*(doc.words for doc in self._docs))
        self._avg_len = (
            sum(len(doc.words) for doc in self._docs) / len(self._docs) if self._docs else 0.0
        )
        # ponytail: unbounded per-router cache of query-word expansions; fine for a
        # catalog's lifetime, add an LRU bound if routers live for millions of queries.
        self._expansions: dict[str, dict[str, float]] = {}

    def _expand(self, query_word: str) -> dict[str, float]:
        """Vocabulary words matching a query word, with their match strength."""
        if query_word not in self._expansions:
            matches = {}
            for doc_word in self._vocabulary:
                strength = match_strength(query_word, doc_word, self.match)
                if strength:
                    matches[doc_word] = strength
            self._expansions[query_word] = matches
        return self._expansions[query_word]

    def _lexical_scores(self, query_words: set[str]) -> list[float]:
        """BM25 score of every tool, with prefix/fuzzy-tolerant term matching."""
        n = len(self._docs)
        scores = [0.0] * n
        for word in query_words:
            matches = self._expand(word)
            if not matches:
                continue
            strengths = [
                max((matches.get(w, 0.0) for w in doc.words), default=0.0) for doc in self._docs
            ]
            doc_freq = sum(1 for s in strengths if s)
            idf = math.log(1 + (n - doc_freq + 0.5) / (doc_freq + 0.5))
            for i, strength in enumerate(strengths):
                if strength:
                    norm = 1 - self.b + self.b * len(self._docs[i].words) / self._avg_len
                    scores[i] += strength * idf * (self.k1 + 1) / (1 + self.k1 * norm)
        return scores

    def _query_vector(self, text: str) -> list[float]:
        if text not in self._query_vectors:
            assert self.embed is not None
            (vector,) = _unit_vectors(self.embed([text], query=True), 1)
            if len(self._query_vectors) >= 256:  # bounded: drop the oldest entry
                del self._query_vectors[next(iter(self._query_vectors))]
            self._query_vectors[text] = vector
        return self._query_vectors[text]

    def _semantic_order(self, text: str) -> list[int]:
        """Tool indices nearest the question, best first, capped at the selection budget."""
        assert self.embed is not None
        if self._vectors is None:
            texts = [doc.text for doc in self._docs]
            self._vectors = _unit_vectors(self.embed(texts, query=False), len(texts))
        query = self._query_vector(text)
        if len(query) != len(self._vectors[0]):
            raise ValueError("embedder returned query and tool vectors of different lengths")
        similarity = [sum(q * t for q, t in zip(query, v, strict=True)) for v in self._vectors]
        order = sorted(range(len(similarity)), key=lambda i: -similarity[i])
        if self.min_similarity is None:
            # Unrelated or opposite meaning is not a match, even when it is the nearest.
            order = [i for i in order if similarity[i] > 0]
        else:
            order = [i for i in order if similarity[i] >= self.min_similarity]
        return order[: self.max_tools + len(self.always_include)]

    def rank(self, query: str, history: Sequence[str] = ()) -> list[tuple[str, float]]:
        """Relevant tools, best first (ties keep catalog order).

        Without ``embed``: every tool with a positive BM25 score, scored by BM25.
        With ``embed``: the BM25 ranking fused with the nearest tools by meaning,
        scored by Reciprocal Rank Fusion.

        ``history`` is earlier user messages; they are folded into the query so
        follow-ups ("and last month?") keep the earlier topic's tools. A question
        with no content words at all ranks nothing. Pinned tools are scored like
        any other.
        """
        text = " ".join([query, *history])
        query_words = set(tokenize(text))
        n = len(self._docs)
        if not query_words or not n:
            return []
        lexical = self._lexical_scores(query_words)
        lexical_order = sorted((i for i in range(n) if lexical[i] > 0), key=lambda i: -lexical[i])
        if self.embed is None:
            return [(self._docs[i].name, lexical[i]) for i in lexical_order]

        # Reciprocal Rank Fusion uses rank positions only, so BM25 scores and cosine
        # similarities never need to be put on the same scale.
        fused = [0.0] * n
        for order in (lexical_order, self._semantic_order(text)):
            for position, i in enumerate(order, 1):
                fused[i] += 1 / (self.rrf_k + position)
        order = sorted((i for i in range(n) if fused[i] > 0), key=lambda i: -fused[i])
        return [(self._docs[i].name, fused[i]) for i in order]

    def select(self, query: str, history: Sequence[str] = ()) -> list[Any]:
        """The pinned tools plus the top ``max_tools`` ranked tools, in catalog order.

        Keeping catalog order (not rank order) means the same tool set always
        serializes identically, which keeps LLM prompt caching effective.
        """
        ranking = self.rank(query, history)
        if not ranking and self.on_no_match == "all":
            return list(self.tools)
        ranked = [name for name, _ in ranking if name not in self.always_include]
        keep = self.always_include | set(ranked[: self.max_tools])
        return [tool for tool, doc in zip(self.tools, self._docs, strict=True) if doc.name in keep]
