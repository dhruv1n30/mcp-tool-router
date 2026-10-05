from types import SimpleNamespace

import pytest

from mcp_tool_router.router import describe_tool

SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "description": "City name"},
        "units": {"type": "string", "enum": ["metric", "imperial"]},
    },
}


def words(tool):
    return set(describe_tool(tool)[1].split())


@pytest.mark.parametrize(
    "tool",
    [
        {"name": "forecast", "description": "Weather outlook", "inputSchema": SCHEMA},  # MCP
        {"name": "forecast", "description": "Weather outlook", "input_schema": SCHEMA},  # Anthropic
        {  # OpenAI chat completions
            "type": "function",
            "function": {
                "name": "forecast",
                "description": "Weather outlook",
                "parameters": SCHEMA,
            },
        },
        {  # OpenAI responses API / Gemini dict
            "type": "function",
            "name": "forecast",
            "description": "Weather outlook",
            "parameters": SCHEMA,
        },
        SimpleNamespace(name="forecast", description="Weather outlook", inputSchema=SCHEMA),
        SimpleNamespace(name="forecast", description="Weather outlook", parameters=SCHEMA),
    ],
    ids=["mcp", "anthropic", "openai-chat", "openai-responses", "object-mcp", "object-gemini"],
)
def test_every_supported_shape(tool):
    name, _ = describe_tool(tool)
    assert name == "forecast"
    assert {
        "forecast",
        "Weather",
        "outlook",
        "city",
        "City",
        "units",
        "metric",
        "imperial",
    } <= words(tool)


def test_object_schema_with_object_properties():
    # e.g. google.genai types.Schema: properties map to objects, not dicts.
    param = SimpleNamespace(description="Warehouse code", enum=None)
    schema = SimpleNamespace(properties={"warehouse": param})
    tool = SimpleNamespace(name="stock", description=None, parameters=schema)
    assert describe_tool(tool) == ("stock", "stock  warehouse Warehouse code")


def test_mcp_types_tool():
    types = pytest.importorskip("mcp.types")
    tool = types.Tool(name="forecast", description="Weather outlook", inputSchema=SCHEMA)
    assert describe_tool(tool)[0] == "forecast"
    assert "imperial" in words(tool)


def test_missing_description_and_schema():
    assert describe_tool({"name": "ping"}) == ("ping", "ping ")


def test_non_dict_function_field_is_ignored():
    assert describe_tool({"name": "ping", "function": "not-a-spec"})[0] == "ping"


@pytest.mark.parametrize("tool", [{"description": "no name"}, {"name": ""}, {"name": 42}])
def test_missing_or_bad_name_raises(tool):
    with pytest.raises(TypeError, match="no name"):
        describe_tool(tool)


@pytest.mark.parametrize("tool", [42, "send_email", None, ["name"]])
def test_unsupported_type_raises(tool):
    with pytest.raises(TypeError, match="Unsupported tool type"):
        describe_tool(tool)
