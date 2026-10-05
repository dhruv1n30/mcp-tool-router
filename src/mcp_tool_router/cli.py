"""Command line: ``mcp-tool-router eval`` and ``mcp-tool-router serve``.

Exit codes: 0 success, 1 recall below ``--min-recall``, 2 bad input or usage.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mcp_tool_router import embedders
from mcp_tool_router.evaluate import evaluate, load_cases, load_tools
from mcp_tool_router.router import Embedder, ToolRouter

FASTMCP_HINT = "this command needs FastMCP: pip install 'mcp-tool-router[fastmcp]'"
SEMANTIC_HINT = "--semantic needs fastembed: pip install 'mcp-tool-router[semantic]'"


def _names(value: str) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


def _load_config(path: str) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("mcpServers"), dict):
        raise ValueError(f"{path}: expected an MCP config with an 'mcpServers' object")
    return config


async def _fetch_tools(config: dict[str, Any]) -> list[dict[str, Any]]:
    from fastmcp import Client

    async with Client(config) as client:
        tools = await client.list_tools()
    return [tool.model_dump(mode="json", exclude_none=True) for tool in tools]


def _embedder(args: argparse.Namespace) -> Embedder | None:
    if args.semantic is None:
        return None
    try:
        return embedders.fastembed_embedder(args.semantic)
    except ImportError:
        raise ValueError(SEMANTIC_HINT) from None


def _add_semantic_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--semantic",
        nargs="?",
        const=embedders.DEFAULT_MODEL,
        metavar="MODEL",
        help=f"also match by meaning (local fastembed model, default {embedders.DEFAULT_MODEL})",
    )
    parser.add_argument("--min-similarity", type=float, help="cosine floor for semantic matches")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-tool-router", description="Route LLM requests to the relevant MCP tools."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("eval", help="measure recall and payload savings on labelled questions")
    source = ev.add_mutually_exclusive_group(required=True)
    source.add_argument("--tools", help="JSON file with the tool catalog")
    source.add_argument("--config", help="MCP config (mcpServers) to fetch tools from live")
    ev.add_argument("--cases", required=True, help="CSV or JSONL with question,expected")
    ev.add_argument("--max-tools", type=int, default=12)
    ev.add_argument("--always-include", type=_names, default=[], help="comma-separated names")
    ev.add_argument("--match", choices=["exact", "prefix", "fuzzy"], default="fuzzy")
    ev.add_argument("--min-recall", type=float, help="exit 1 if recall is below this (0-1)")
    ev.add_argument("--show-misses", type=int, default=10)
    _add_semantic_options(ev)

    sv = sub.add_parser("serve", help="proxy MCP servers behind search_tools + call_tool")
    sv.add_argument("--config", required=True, help="MCP config (mcpServers) to proxy")
    sv.add_argument("--max-results", type=int, default=8)
    sv.add_argument("--always-visible", type=_names, default=[], help="comma-separated names")
    sv.add_argument("--match", choices=["exact", "prefix", "fuzzy"], default="fuzzy")
    sv.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    _add_semantic_options(sv)
    return parser


def _run_eval(args: argparse.Namespace) -> int:
    if args.config:
        try:
            tools = asyncio.run(_fetch_tools(_load_config(args.config)))
        except ImportError:
            raise ValueError(FASTMCP_HINT) from None
    else:
        tools = load_tools(args.tools)
    router = ToolRouter(
        tools,
        max_tools=args.max_tools,
        always_include=args.always_include,
        match=args.match,
        embed=_embedder(args),
        min_similarity=args.min_similarity,
    )
    report = evaluate(router, load_cases(args.cases))
    print(report.format(args.show_misses))
    if args.min_recall is not None and report.recall < args.min_recall:
        print(f"FAIL: recall {report.recall:.1%} is below {args.min_recall:.1%}", file=sys.stderr)
        return 1
    return 0


def _run_serve(args: argparse.Namespace) -> int:
    try:
        from fastmcp.server import create_proxy

        from mcp_tool_router.transform import RouterSearchTransform
    except ImportError:
        raise ValueError(FASTMCP_HINT) from None
    proxy = create_proxy(_load_config(args.config), name="mcp-tool-router")
    proxy.add_transform(
        RouterSearchTransform(
            max_results=args.max_results,
            always_visible=args.always_visible,
            match=args.match,
            embed=_embedder(args),
            min_similarity=args.min_similarity,
        )
    )
    if args.transport == "http":
        proxy.run(transport="http", host=args.host, port=args.port)
    else:
        proxy.run(transport="stdio")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _run_eval(args) if args.command == "eval" else _run_serve(args)
    except (OSError, ValueError, TypeError) as exc:  # bad files, data or options
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
