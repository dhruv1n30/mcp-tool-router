"""End to end: a real FastMCP server in a stdio subprocess, behind the router."""

import json
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("fastmcp")

from fastmcp import Client
from fastmcp.server import create_proxy

from mcp_tool_router import RouterSearchTransform
from mcp_tool_router.cli import main

pytestmark = pytest.mark.e2e


async def test_proxy_routes_to_a_subprocess_server(demo_config):
    proxy = create_proxy(demo_config, name="router")
    proxy.add_transform(RouterSearchTransform(max_results=2))
    async with Client(proxy) as client:
        assert [t.name for t in await client.list_tools()] == ["search_tools", "call_tool"]

        found = await client.call_tool("search_tools", {"query": "wether forcast"})
        assert [tool["name"] for tool in found.data] == ["get_weather_forecast"]

        result = await client.call_tool(
            "call_tool", {"name": "get_weather_forecast", "arguments": {"city": "Pune"}}
        )
        assert result.data == "Sunny in Pune"


def test_cli_eval_fetches_tools_live(tmp_path, demo_config, capfd):
    # capfd, not capsys: the stdio launcher needs a real stderr file descriptor.
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps(demo_config), encoding="utf-8")
    cases = tmp_path / "cases.csv"
    cases.write_text(
        "question,expected\nweather in Pune,get_weather_forecast\n"
        "value of stock,get_inventory_value\nemail the team,send_email\n",
        encoding="utf-8",
    )
    code = main(["eval", "--config", str(config), "--cases", str(cases), "--max-tools", "1"])
    out = capfd.readouterr().out
    assert code == 0, out
    assert "recall           100.0%  (3/3 kept every tool)" in out
    assert "tools per query  1.0 of 3" in out


@pytest.mark.parametrize(
    "command", [["mcp-tool-router"], [sys.executable, "-m", "mcp_tool_router.cli"]]
)
def test_installed_entry_points(command):
    if command[0] == "mcp-tool-router" and not shutil.which("mcp-tool-router"):
        pytest.skip("console script not installed in this environment")
    result = subprocess.run([*command, "--help"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    assert "eval" in result.stdout and "serve" in result.stdout
