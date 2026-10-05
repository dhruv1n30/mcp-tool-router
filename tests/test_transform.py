import pytest

fastmcp = pytest.importorskip("fastmcp")

from fastmcp import Client, FastMCP  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402
from fastmcp.server.transforms.search import BM25SearchTransform  # noqa: E402

import mcp_tool_router  # noqa: E402
from mcp_tool_router import RouterSearchTransform  # noqa: E402


def build_server(transform) -> FastMCP:
    mcp = FastMCP("test")

    @mcp.tool
    def get_inventory_value(warehouse: str) -> str:
        """Total value of stock on hand in a warehouse."""
        return f"{warehouse}: 1200"

    @mcp.tool
    def get_sales_by_category(start_date: str) -> str:
        """Net sales grouped by product category."""
        return f"sales since {start_date}"

    @mcp.tool
    def send_email(to: str) -> str:
        """Send an email message to a recipient."""
        return f"sent to {to}"

    @mcp.tool
    def ping() -> str:
        """Health check."""
        return "pong"

    mcp.add_transform(transform)
    return mcp


async def search(client: Client, query: str) -> list[str]:
    result = await client.call_tool("search_tools", {"query": query})
    return [tool["name"] for tool in result.data]


async def test_list_tools_shows_only_pinned_and_synthetic_tools():
    async with Client(build_server(RouterSearchTransform(always_visible=["ping"]))) as client:
        assert [t.name for t in await client.list_tools()] == ["ping", "search_tools", "call_tool"]


async def test_search_returns_full_definitions_best_first():
    async with Client(build_server(RouterSearchTransform())) as client:
        result = await client.call_tool("search_tools", {"query": "stock value in a warehouse"})
        first = result.data[0]
        assert first["name"] == "get_inventory_value"
        assert "warehouse" in first["inputSchema"]["properties"]


async def test_typo_query_found_where_fastmcp_bm25_finds_nothing():
    # The reason this package exists: FastMCP's built-in BM25 matches exact tokens only.
    async with Client(build_server(BM25SearchTransform())) as client:
        assert await search(client, "inventroy worth") == []
    async with Client(build_server(RouterSearchTransform())) as client:
        assert await search(client, "inventroy worth") == ["get_inventory_value"]


async def test_prefix_query():
    async with Client(build_server(RouterSearchTransform())) as client:
        assert (await search(client, "sale per categ"))[0] == "get_sales_by_category"


async def test_max_results_caps_the_search():
    async with Client(build_server(RouterSearchTransform(max_results=1))) as client:
        assert len(await search(client, "sales stock email")) == 1


async def test_no_match_returns_no_tools():
    async with Client(build_server(RouterSearchTransform())) as client:
        assert await search(client, "zzqqx") == []


async def test_pinned_tools_are_not_search_results():
    async with Client(build_server(RouterSearchTransform(always_visible=["ping"]))) as client:
        assert "ping" not in await search(client, "health check ping")


async def test_exact_mode_behaves_like_plain_bm25():
    async with Client(build_server(RouterSearchTransform(match="exact"))) as client:
        assert await search(client, "inventroy worth") == []


async def test_call_tool_runs_a_discovered_tool():
    async with Client(build_server(RouterSearchTransform())) as client:
        args = {"name": "get_inventory_value", "arguments": {"warehouse": "A"}}
        assert (await client.call_tool("call_tool", args)).data == "A: 1200"


async def test_hidden_tools_stay_directly_callable():
    async with Client(build_server(RouterSearchTransform())) as client:
        assert (await client.call_tool("send_email", {"to": "x@y.z"})).data == "sent to x@y.z"


async def test_call_tool_rejects_unknown_tools():
    async with Client(build_server(RouterSearchTransform())) as client:
        with pytest.raises(ToolError):
            await client.call_tool("call_tool", {"name": "drop_database", "arguments": {}})


async def test_custom_synthetic_tool_names():
    transform = RouterSearchTransform(search_tool_name="find_tools", call_tool_name="run_tool")
    async with Client(build_server(transform)) as client:
        assert {t.name for t in await client.list_tools()} == {"find_tools", "run_tool"}


async def test_router_is_reused_until_the_catalog_changes():
    transform = RouterSearchTransform()
    server = build_server(transform)
    async with Client(server) as client:
        await search(client, "stock")
        first = transform._router
        await search(client, "email")
        assert transform._router is first

        @server.tool
        def get_weather_forecast(city: str) -> str:
            """Weather forecast for a city."""
            return f"Sunny in {city}"

        assert await search(client, "weather") == ["get_weather_forecast"]
        assert transform._router is not first


def test_max_results_must_be_positive():
    with pytest.raises(ValueError, match="max_results"):
        RouterSearchTransform(max_results=0)


def test_lazy_export_from_package_root():
    assert mcp_tool_router.RouterSearchTransform is RouterSearchTransform
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = mcp_tool_router.nope
