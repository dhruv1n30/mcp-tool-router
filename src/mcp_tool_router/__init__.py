"""Route LLM requests to the MCP tools that matter."""

from importlib.metadata import version

from mcp_tool_router.evaluate import Case, Report, evaluate, load_cases, load_tools
from mcp_tool_router.router import Embedder, ToolRouter, describe_tool, match_strength, tokenize

__version__ = version("mcp-tool-router")

__all__ = [
    "Case",
    "Embedder",
    "Report",
    "RouterSearchTransform",
    "ToolRouter",
    "__version__",
    "describe_tool",
    "evaluate",
    "load_cases",
    "load_tools",
    "match_strength",
    "tokenize",
]


def __getattr__(name: str):
    # Lazy so the core package works without the optional fastmcp dependency.
    if name == "RouterSearchTransform":
        from mcp_tool_router.transform import RouterSearchTransform

        return RouterSearchTransform
    raise AttributeError(f"module 'mcp_tool_router' has no attribute {name!r}")
