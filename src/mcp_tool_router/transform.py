"""FastMCP integration: a drop-in replacement for ``BM25SearchTransform``.

Same ``search_tools`` + ``call_tool`` interface as FastMCP's built-in search
transforms, ranked by :class:`ToolRouter` (prefix and typo-tolerant matching,
stopwords, enum values indexed). Requires the ``fastmcp`` extra.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Annotated, Any

from fastmcp.server.context import Context
from fastmcp.server.transforms.search.base import BaseSearchTransform, SearchResultSerializer
from fastmcp.tools.base import Tool

from mcp_tool_router.router import Embedder, MatchMode, ToolRouter


def _fingerprint(tools: Sequence[Tool]) -> str:
    payload = [[t.name, t.description or "", t.parameters] for t in tools]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class RouterSearchTransform(BaseSearchTransform):
    """Collapse a server's tool list into ``search_tools`` + ``call_tool``.

    Example::

        from fastmcp import FastMCP
        from mcp_tool_router import RouterSearchTransform

        mcp = FastMCP("Server")
        mcp.add_transform(RouterSearchTransform(max_results=8))

    Args:
        max_results: Most tools returned per search.
        always_visible: Tool names kept in ``list_tools`` next to the search tools.
        match: Term matching mode, see :class:`ToolRouter`.
        embed: Optional embedder for semantic search, see :class:`ToolRouter`.
        min_similarity: Cosine floor for semantic matches, see :class:`ToolRouter`.
    """

    def __init__(
        self,
        *,
        max_results: int = 5,
        always_visible: list[str] | None = None,
        match: MatchMode = "fuzzy",
        embed: Embedder | None = None,
        min_similarity: float | None = None,
        search_tool_name: str = "search_tools",
        call_tool_name: str = "call_tool",
        search_result_serializer: SearchResultSerializer | None = None,
    ) -> None:
        if max_results < 1:
            raise ValueError("max_results must be at least 1")
        super().__init__(
            max_results=max_results,
            always_visible=always_visible,
            search_tool_name=search_tool_name,
            call_tool_name=call_tool_name,
            search_result_serializer=search_result_serializer,
        )
        self._match: MatchMode = match
        self._embed = embed
        self._min_similarity = min_similarity
        self._router: ToolRouter | None = None
        self._router_key = ""

    def _make_search_tool(self) -> Tool:
        transform = self

        async def search_tools(
            query: Annotated[str, "What you want to do, in plain words"],
            ctx: Context = None,  # type: ignore[assignment]
        ) -> str | list[dict[str, Any]]:
            """Find the tools that can help with a task.

            Returns matching tool definitions, best first, in the same format as
            list_tools. Call them with call_tool.
            """
            hidden = await transform._get_visible_tools(ctx)
            return await transform._render_results(await transform._search(hidden, query))

        return Tool.from_function(fn=search_tools, name=self._search_tool_name)

    async def _search(self, tools: Sequence[Tool], query: str) -> Sequence[Tool]:
        key = _fingerprint(tools)
        if self._router is None or key != self._router_key:
            # Rebuilt only when the catalog changes, so the word-match cache and
            # the tool embeddings survive between searches.
            self._router = ToolRouter(
                tools,
                max_tools=self._max_results,
                match=self._match,
                embed=self._embed,
                min_similarity=self._min_similarity,
            )
            self._router_key = key
        by_name = {tool.name: tool for tool in tools}
        return [by_name[name] for name, _ in self._router.rank(query)[: self._max_results]]
