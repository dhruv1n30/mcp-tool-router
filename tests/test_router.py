import pytest

from mcp_tool_router import ToolRouter


def names(tools):
    return [tool["name"] for tool in tools]


def ranked(router, query, **kwargs):
    return [name for name, _ in router.rank(query, **kwargs)]


class TestRank:
    def test_finds_the_obvious_tool_first(self, catalog):
        assert ranked(ToolRouter(catalog), "weather forecast in Paris")[0] == "get_weather_forecast"

    def test_scores_are_positive_and_descending(self, catalog):
        scores = [
            score for _, score in ToolRouter(catalog).rank("sales value of stock by category")
        ]
        assert scores and all(s > 0 for s in scores)
        assert scores == sorted(scores, reverse=True)

    def test_matches_parameter_descriptions(self, catalog):
        assert ranked(ToolRouter(catalog), "which warehouse code")[0] == "get_inventory_value"

    def test_matches_enum_values(self, catalog):
        assert ranked(ToolRouter(catalog), "how much tax did we pay")[0] == "get_metric"

    def test_tolerates_typos(self, catalog):
        assert ranked(ToolRouter(catalog), "inventroy")[0] == "get_inventory_value"

    def test_matches_word_stems(self, catalog):
        assert ranked(ToolRouter(catalog), "sale per categ")[0] == "get_sales_by_category"

    def test_exact_mode_misses_typos(self, catalog):
        assert ranked(ToolRouter(catalog, match="exact"), "inventroy") == []

    def test_rare_word_outweighs_common_word(self):
        # "report" is in every tool, "payroll" in one: IDF must make payroll decide.
        tools = [{"name": f"tool_{i}", "description": f"report number {i}"} for i in range(5)] + [
            {"name": "payroll_tool", "description": "payroll summary"}
        ]
        tools[0]["description"] += " report report"
        assert ranked(ToolRouter(tools), "payroll report")[0] == "payroll_tool"

    def test_shorter_description_wins_on_equal_match(self):
        tools = [
            {
                "name": "long_one",
                "description": "refunds " + " ".join(f"word{i}" for i in range(30)),
            },
            {"name": "short_one", "description": "refunds"},
        ]
        assert ranked(ToolRouter(tools), "refunds") == ["short_one", "long_one"]

    def test_exact_match_beats_fuzzy_match(self):
        tools = [
            {"name": "alpha", "description": "inventry"},  # fuzzy match only
            {"name": "beta", "description": "inventory"},  # exact
        ]
        assert ranked(ToolRouter(tools), "inventory") == ["beta", "alpha"]

    def test_ties_keep_catalog_order(self):
        tools = [{"name": n, "description": "refunds"} for n in ("zeta", "alpha", "mid")]
        assert ranked(ToolRouter(tools), "refunds") == ["zeta", "alpha", "mid"]

    def test_history_folds_in_earlier_topic(self, catalog):
        router = ToolRouter(catalog)
        assert ranked(router, "same again please") == []
        assert "get_sales_by_category" in ranked(
            router, "same again please", history=["sales by category"]
        )

    @pytest.mark.parametrize("query", ["", "   ", "what is the", "zzqqx"])
    def test_no_content_or_no_match_ranks_nothing(self, catalog, query):
        assert ToolRouter(catalog).rank(query) == []

    def test_empty_catalog(self):
        assert ToolRouter([]).rank("sales") == []


class TestSelect:
    def test_returns_the_callers_own_objects(self, catalog):
        selected = ToolRouter(catalog).select("weather")
        assert selected and all(any(s is c for c in catalog) for s in selected)

    def test_keeps_catalog_order_not_rank_order(self, catalog):
        selected = names(ToolRouter(catalog, max_tools=3).select("email calendar weather"))
        catalog_names = names(catalog)
        assert selected == sorted(selected, key=catalog_names.index)

    def test_caps_at_max_tools(self, catalog):
        assert len(ToolRouter(catalog, max_tools=2).select("sales stock hours email weather")) == 2

    def test_pinned_tools_always_present_and_outside_the_cap(self, catalog):
        router = ToolRouter(catalog, max_tools=1, always_include=["get_user_profile"])
        assert names(router.select("weather")) == ["get_weather_forecast", "get_user_profile"]

    def test_pinned_tool_that_also_matches_is_not_counted_twice(self, catalog):
        router = ToolRouter(catalog, max_tools=1, always_include=["get_weather_forecast"])
        selected = names(router.select("weather send"))
        assert selected == ["send_email", "get_weather_forecast"]

    def test_no_match_fails_open_by_default(self, catalog):
        assert ToolRouter(catalog).select("hello there") == catalog

    def test_no_match_with_pinned_policy(self, catalog):
        router = ToolRouter(catalog, always_include=["get_user_profile"], on_no_match="pinned")
        assert names(router.select("hello there")) == ["get_user_profile"]

    def test_pinned_policy_without_pins_returns_nothing(self, catalog):
        assert ToolRouter(catalog, on_no_match="pinned").select("zzqqx") == []

    def test_match_on_a_pinned_tool_only_does_not_fail_open(self, catalog):
        # Something matched (the pinned tool), so this is not a "no match" case.
        router = ToolRouter(catalog, always_include=["get_weather_forecast"])
        assert names(router.select("weather")) == ["get_weather_forecast"]

    def test_empty_catalog(self):
        assert ToolRouter([]).select("sales") == []

    def test_accepts_any_iterable(self, catalog):
        assert names(ToolRouter(iter(catalog)).select("weather"))[0] == "get_weather_forecast"

    def test_repeated_queries_use_the_cache_consistently(self, catalog):
        router = ToolRouter(catalog)
        first = router.select("inventroy value")
        assert router.select("inventroy value") == first


class TestValidation:
    @pytest.mark.parametrize("max_tools", [0, -3])
    def test_max_tools_must_be_positive(self, catalog, max_tools):
        with pytest.raises(ValueError, match="max_tools"):
            ToolRouter(catalog, max_tools=max_tools)

    def test_bad_match_mode(self, catalog):
        with pytest.raises(ValueError, match="match must be"):
            ToolRouter(catalog, match="semantic")

    def test_bad_no_match_policy(self, catalog):
        with pytest.raises(ValueError, match="on_no_match"):
            ToolRouter(catalog, on_no_match="nothing")

    def test_unknown_pinned_name(self, catalog):
        with pytest.raises(ValueError, match="not in the catalog"):
            ToolRouter(catalog, always_include=["get_user_profil"])

    def test_duplicate_tool_names(self, catalog):
        with pytest.raises(ValueError, match="Duplicate tool names"):
            ToolRouter([*catalog, catalog[0]])

    def test_bad_tool_in_catalog(self, catalog):
        with pytest.raises(TypeError):
            ToolRouter([*catalog, 42])
