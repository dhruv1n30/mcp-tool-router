import sys
import textwrap
from pathlib import Path

import pytest

# A small generic catalog in MCP `tools/list` shape.
CATALOG = [
    {
        "name": "get_sales_by_category",
        "description": "Net sales grouped by product category for a date range.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "First day, YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "Last day, YYYY-MM-DD"},
            },
        },
    },
    {
        "name": "get_inventory_value",
        "description": "Total value of stock on hand.",
        "inputSchema": {
            "type": "object",
            "properties": {"warehouse": {"type": "string", "description": "Warehouse code"}},
        },
    },
    {
        "name": "get_employee_hours",
        "description": "Hours worked per employee, from the timesheet.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_metric",
        "description": "One headline number for a period.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "description": "Which number",
                    "enum": ["revenue", "refunds", "tax"],
                }
            },
        },
    },
    {
        "name": "send_email",
        "description": "Send an email message to a recipient.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient address"},
                "subject": {"type": "string"},
            },
        },
    },
    {
        "name": "get_weather_forecast",
        "description": "Weather forecast for a city.",
        "inputSchema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
        },
    },
    {
        "name": "create_calendar_event",
        "description": "Create an event on the calendar.",
        "inputSchema": {
            "type": "object",
            "properties": {"title": {"type": "string"}, "when": {"type": "string"}},
        },
    },
    {
        "name": "get_user_profile",
        "description": "The signed-in user's name and email.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

DEMO_SERVER = textwrap.dedent(
    '''
    from fastmcp import FastMCP

    mcp = FastMCP("demo")

    @mcp.tool
    def get_weather_forecast(city: str) -> str:
        """Weather forecast for a city."""
        return f"Sunny in {city}"

    @mcp.tool
    def get_inventory_value(warehouse: str) -> str:
        """Total value of stock on hand in a warehouse."""
        return f"{warehouse}: 1200"

    @mcp.tool
    def send_email(to: str, subject: str) -> str:
        """Send an email message to a recipient."""
        return f"sent to {to}"

    if __name__ == "__main__":
        mcp.run(show_banner=False)
    '''
)


@pytest.fixture
def catalog():
    return [dict(tool) for tool in CATALOG]


@pytest.fixture
def demo_config(tmp_path: Path) -> dict:
    """MCP config that launches a real FastMCP server as a stdio subprocess."""
    script = tmp_path / "demo_server.py"
    script.write_text(DEMO_SERVER, encoding="utf-8")
    return {"mcpServers": {"demo": {"command": sys.executable, "args": [str(script)]}}}
