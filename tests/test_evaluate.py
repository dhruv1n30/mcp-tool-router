import json

import pytest

from mcp_tool_router import Case, Report, ToolRouter, evaluate, load_cases, load_tools
from mcp_tool_router.evaluate import payload_size

CASES = [
    Case("weather forecast in Paris", ("get_weather_forecast",)),
    Case("send an email", ("send_email",)),
    Case("inventory worth and sales", ("get_inventory_value", "get_sales_by_category")),
    Case("zzqqx", ("get_metric",)),  # no match -> fails open -> still a hit
    Case("weather", ("send_email",)),  # wrong tool kept -> miss
]


class TestEvaluate:
    def test_metrics(self, catalog):
        report = evaluate(ToolRouter(catalog, max_tools=2), CASES)
        assert (report.cases, report.hits, report.total_tools) == (5, 4, 8)
        assert report.recall == pytest.approx(0.8)
        assert report.mean_selected == pytest.approx((1 + 2 + 2 + 8 + 1) / 5)
        assert [(case.question, missing) for case, missing in report.misses] == [
            ("weather", ("send_email",))
        ]
        assert report.no_match == 1  # "zzqqx"

    def test_no_match_exposes_fail_open_hits(self, catalog):
        # Exact matching misses the typo, fails open, and still "hits": only
        # the no_match count shows the full catalog was sent.
        typo = [Case("inventroy", ("get_inventory_value",))]
        exact = evaluate(ToolRouter(catalog, match="exact"), typo)
        fuzzy = evaluate(ToolRouter(catalog, match="fuzzy"), typo)
        assert (exact.recall, exact.no_match, exact.mean_selected) == (1.0, 1, 8)
        assert (fuzzy.recall, fuzzy.no_match, fuzzy.mean_selected) == (1.0, 0, 1)

    def test_payload_numbers(self, catalog):
        report = evaluate(ToolRouter(catalog, max_tools=2), CASES[:1])
        sizes = {tool["name"]: payload_size(tool) for tool in catalog}
        assert report.payload_all == sum(sizes.values())
        assert report.mean_payload_selected == sizes["get_weather_forecast"]
        assert report.payload_reduction == pytest.approx(
            1 - sizes["get_weather_forecast"] / sum(sizes.values())
        )

    def test_multi_tool_case_needs_every_tool(self, catalog):
        report = evaluate(ToolRouter(catalog, max_tools=1), [CASES[2]])
        assert report.hits == 0
        assert len(report.misses[0][1]) == 1

    def test_unknown_expected_tool(self, catalog):
        with pytest.raises(ValueError, match="not in the catalog"):
            evaluate(ToolRouter(catalog), [Case("x", ("no_such_tool",))])

    def test_empty_cases(self, catalog):
        report = evaluate(ToolRouter(catalog), [])
        assert (report.recall, report.mean_selected, report.mean_payload_selected) == (0, 0, 0)


class TestReport:
    def test_format(self, catalog):
        text = evaluate(ToolRouter(catalog, max_tools=2), CASES).format()
        assert "recall           80.0%  (4/5 kept every tool)" in text
        assert "tools per query  2.8 of 8 on average" in text
        assert "miss: 'weather' dropped send_email" in text
        assert "no match         1 questions matched no tool" in text

    def test_format_truncates_misses(self):
        misses = [(Case(f"q{i}", ("t",)), ("t",)) for i in range(5)]
        report = Report(5, 0, 1, 1.0, 100, 100.0, misses)
        text = report.format(show_misses=2)
        assert text.count("miss:") == 2
        assert "... and 3 more misses" in text

    def test_zero_safe_properties(self):
        report = Report(0, 0, 0, 0.0, 0, 0.0)
        assert report.recall == 0.0
        assert report.payload_reduction == 0.0


class TestPayloadSize:
    def test_dict(self):
        assert payload_size({"name": "a"}) == len('{"name": "a"}')

    def test_pydantic_model_is_dumped(self):
        types = pytest.importorskip("mcp.types")
        tool = types.Tool(name="a", inputSchema={"type": "object"})
        assert payload_size(tool) == len(
            json.dumps(tool.model_dump(mode="json", exclude_none=True), sort_keys=True)
        )


class TestLoadTools:
    def test_list(self, tmp_path, catalog):
        path = tmp_path / "tools.json"
        path.write_text(json.dumps(catalog), encoding="utf-8")
        assert load_tools(path) == catalog

    def test_tools_list_result_shape(self, tmp_path, catalog):
        path = tmp_path / "tools.json"
        path.write_text(json.dumps({"tools": catalog, "nextCursor": None}), encoding="utf-8")
        assert load_tools(path) == catalog

    @pytest.mark.parametrize("content", ['{"name": "x"}', '"tools"', "42"])
    def test_wrong_shape(self, tmp_path, content):
        path = tmp_path / "tools.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="expected a JSON list"):
            load_tools(path)

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "tools.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError):
            load_tools(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(OSError):
            load_tools(tmp_path / "missing.json")


class TestLoadCases:
    def test_csv(self, tmp_path):
        path = tmp_path / "cases.csv"
        path.write_text(
            "question,expected\nweather in Paris,get_weather_forecast\n"
            '"sales, and stock",get_sales_by_category | get_inventory_value\n',
            encoding="utf-8",
        )
        assert load_cases(path) == [
            Case("weather in Paris", ("get_weather_forecast",)),
            Case("sales, and stock", ("get_sales_by_category", "get_inventory_value")),
        ]

    def test_csv_with_excel_bom_and_extra_columns(self, tmp_path):
        path = tmp_path / "cases.csv"
        path.write_bytes("﻿id,question,expected\n1,hi,ping\n".encode())
        assert load_cases(path) == [Case("hi", ("ping",))]

    def test_csv_missing_columns(self, tmp_path):
        path = tmp_path / "cases.csv"
        path.write_text("prompt,tool\nhi,ping\n", encoding="utf-8")
        with pytest.raises(ValueError, match="'question' and 'expected' columns"):
            load_cases(path)

    def test_csv_empty_expected(self, tmp_path):
        path = tmp_path / "cases.csv"
        path.write_text("question,expected\nhi, | \n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"cases.csv:2: no expected tool"):
            load_cases(path)

    def test_csv_no_rows(self, tmp_path):
        path = tmp_path / "cases.csv"
        path.write_text("question,expected\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no cases found"):
            load_cases(path)

    def test_jsonl(self, tmp_path):
        path = tmp_path / "cases.jsonl"
        path.write_text(
            '{"question": "weather", "expected": "get_weather_forecast"}\n\n'
            '{"question": "both", "expected": ["a", "b"]}\n',
            encoding="utf-8",
        )
        assert load_cases(path) == [
            Case("weather", ("get_weather_forecast",)),
            Case("both", ("a", "b")),
        ]

    @pytest.mark.parametrize("line", ['{"question": "x"}', '["x", "y"]'])
    def test_jsonl_missing_keys(self, tmp_path, line):
        path = tmp_path / "cases.jsonl"
        path.write_text(line + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"cases.jsonl:1: need 'question' and 'expected'"):
            load_cases(path)

    def test_jsonl_invalid_json(self, tmp_path):
        path = tmp_path / "cases.jsonl"
        path.write_text("{oops\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_cases(path)
