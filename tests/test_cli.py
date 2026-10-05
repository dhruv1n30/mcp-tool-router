import json
import sys

import pytest

from mcp_tool_router.cli import main


@pytest.fixture
def files(tmp_path, catalog):
    tools = tmp_path / "tools.json"
    tools.write_text(json.dumps(catalog), encoding="utf-8")
    cases = tmp_path / "cases.csv"
    cases.write_text(
        "question,expected\nweather forecast,get_weather_forecast\nsend an email,send_email\n",
        encoding="utf-8",
    )
    return tools, cases


class TestEval:
    def test_happy_path(self, files, capsys):
        tools, cases = files
        assert main(["eval", "--tools", str(tools), "--cases", str(cases)]) == 0
        out = capsys.readouterr().out
        assert "cases            2" in out
        assert "recall           100.0%" in out

    def test_min_recall_passes(self, files):
        tools, cases = files
        assert (
            main(["eval", "--tools", str(tools), "--cases", str(cases), "--min-recall", "1"]) == 0
        )

    def test_min_recall_fails_with_exit_1(self, tmp_path, files, capsys):
        tools, _ = files
        cases = tmp_path / "bad.csv"
        cases.write_text("question,expected\nweather,send_email\n", encoding="utf-8")
        code = main(["eval", "--tools", str(tools), "--cases", str(cases), "--min-recall", "0.9"])
        assert code == 1
        assert "FAIL: recall 0.0% is below 90.0%" in capsys.readouterr().err

    def test_options_reach_the_router(self, files, capsys):
        tools, cases = files
        args = ["eval", "--tools", str(tools), "--cases", str(cases), "--max-tools", "1"]
        args += ["--always-include", "get_user_profile, get_metric", "--match", "exact"]
        assert main(args) == 0
        # 1 routed tool + 2 pinned on every question
        assert "tools per query  3.0 of 8" in capsys.readouterr().out

    def test_show_misses_zero(self, tmp_path, files, capsys):
        tools, _ = files
        cases = tmp_path / "bad.csv"
        cases.write_text("question,expected\nweather,send_email\n", encoding="utf-8")
        main(["eval", "--tools", str(tools), "--cases", str(cases), "--show-misses", "0"])
        assert "miss:" not in capsys.readouterr().out

    def test_missing_file_exits_2(self, tmp_path, files, capsys):
        _, cases = files
        assert main(["eval", "--tools", str(tmp_path / "nope.json"), "--cases", str(cases)]) == 2
        assert capsys.readouterr().err.startswith("error:")

    def test_bad_data_exits_2(self, tmp_path, files, capsys):
        tools, _ = files
        cases = tmp_path / "unknown.csv"
        cases.write_text("question,expected\nhi,no_such_tool\n", encoding="utf-8")
        assert main(["eval", "--tools", str(tools), "--cases", str(cases)]) == 2
        assert "not in the catalog" in capsys.readouterr().err

    def test_bad_router_option_exits_2(self, files, capsys):
        tools, cases = files
        args = ["eval", "--tools", str(tools), "--cases", str(cases), "--max-tools", "0"]
        assert main(args) == 2
        assert "max_tools" in capsys.readouterr().err

    def test_tools_and_config_are_mutually_exclusive(self, files):
        tools, cases = files
        with pytest.raises(SystemExit) as exc:
            main(["eval", "--tools", str(tools), "--config", "x.json", "--cases", str(cases)])
        assert exc.value.code == 2

    def test_needs_a_tool_source(self, files):
        _, cases = files
        with pytest.raises(SystemExit) as exc:
            main(["eval", "--cases", str(cases)])
        assert exc.value.code == 2

    def test_bad_config_shape_exits_2(self, tmp_path, files, capsys):
        _, cases = files
        config = tmp_path / "mcp.json"
        config.write_text('{"servers": {}}', encoding="utf-8")
        assert main(["eval", "--config", str(config), "--cases", str(cases)]) == 2
        assert "'mcpServers'" in capsys.readouterr().err

    def test_config_without_fastmcp_explains_the_extra(self, tmp_path, files, monkeypatch, capsys):
        _, cases = files
        config = tmp_path / "mcp.json"
        config.write_text('{"mcpServers": {}}', encoding="utf-8")
        monkeypatch.setitem(sys.modules, "fastmcp", None)  # simulate it not being installed
        assert main(["eval", "--config", str(config), "--cases", str(cases)]) == 2
        assert "mcp-tool-router[fastmcp]" in capsys.readouterr().err


class TestUsage:
    def test_no_command(self):
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2

    def test_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
        assert "eval" in capsys.readouterr().out

    def test_bad_choice(self, files):
        tools, cases = files
        with pytest.raises(SystemExit):
            main(["eval", "--tools", str(tools), "--cases", str(cases), "--match", "semantic"])


class TestServe:
    def test_without_fastmcp_explains_the_extra(self, tmp_path, monkeypatch, capsys):
        config = tmp_path / "mcp.json"
        config.write_text('{"mcpServers": {}}', encoding="utf-8")
        monkeypatch.setitem(sys.modules, "fastmcp.server", None)
        assert main(["serve", "--config", str(config)]) == 2
        assert "mcp-tool-router[fastmcp]" in capsys.readouterr().err

    def test_bad_config_exits_2(self, tmp_path, capsys):
        pytest.importorskip("fastmcp")  # without it, the install hint wins
        config = tmp_path / "mcp.json"
        config.write_text("[]", encoding="utf-8")
        assert main(["serve", "--config", str(config)]) == 2
        assert "'mcpServers'" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("extra", "expected"),
        [
            ([], {"transport": "stdio"}),
            (
                ["--transport", "http", "--host", "0.0.0.0", "--port", "9000"],
                {"transport": "http", "host": "0.0.0.0", "port": 9000},
            ),
        ],
    )
    def test_runs_the_proxy_with_the_transform(self, tmp_path, monkeypatch, extra, expected):
        pytest.importorskip("fastmcp")
        from fastmcp.server.server import FastMCP

        from mcp_tool_router.transform import RouterSearchTransform

        calls = []
        monkeypatch.setattr(FastMCP, "run", lambda self, **kw: calls.append((self, kw)))
        config = tmp_path / "mcp.json"
        config.write_text('{"mcpServers": {"x": {"command": "x"}}}', encoding="utf-8")
        args = ["serve", "--config", str(config), "--max-results", "3"]
        args += ["--always-visible", "ping", "--match", "prefix", *extra]

        assert main(args) == 0
        ((proxy, kwargs),) = calls
        assert kwargs == expected
        (transform,) = [t for t in proxy.transforms if isinstance(t, RouterSearchTransform)]
        assert (transform._max_results, transform._always_visible, transform._match) == (
            3,
            {"ping"},
            "prefix",
        )
