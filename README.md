# mcp-tool-router

Send an LLM only the tools that matter.

Agents with many MCP servers pay for every tool definition on every request:
dozens of schemas fill the context window, cost tokens, and make the model
more likely to pick the wrong tool. `mcp-tool-router` ranks your catalog
against each question, by keywords (BM25) and optionally by meaning
(embeddings), and returns the few tools that are relevant.

- **`ToolRouter`**: per-question tool selection for your own agent loop
  (OpenAI, Anthropic, Gemini or MCP tool formats). Zero dependencies.
- **Semantic matching**: "what is revenue in the last 5 days" finds
  `get_sales_data` even though its description never says "revenue".
- **`RouterSearchTransform`**: a drop-in replacement for FastMCP's
  `BM25SearchTransform` that also matches word stems, typos and meaning.
- **`mcp-tool-router eval`**: measures whether routing keeps the right tool,
  and how much schema payload it saves. Use `--min-recall` as a CI gate.
- **`mcp-tool-router serve`**: puts any `mcpServers` config behind a
  `search_tools` + `call_tool` proxy.

## Install

```bash
pip install mcp-tool-router              # ToolRouter + eval, no dependencies
pip install "mcp-tool-router[semantic]"  # + a local embedding model (fastembed, CPU, no API key)
pip install "mcp-tool-router[fastmcp]"   # + RouterSearchTransform, serve, eval --config
```

## Route tools in your own agent loop

```python
from mcp_tool_router import ToolRouter

router = ToolRouter(tools, max_tools=8, always_include=["get_user_profile"])

relevant = router.select("how much stock is in warehouse B?")
response = client.chat.completions.create(model=..., messages=..., tools=relevant)
```

`tools` can be MCP `tools/list` results (dicts or `mcp.types.Tool`), OpenAI
function tools, Anthropic tools, or Gemini function declarations. `select`
returns your own objects, so they go straight back to the same SDK.

- **Pinned tools** (`always_include`) are sent on every question, outside the cap.
- **Follow-ups**: `router.select("and last month?", history=["sales by category"])`
  keeps the earlier topic's tools in play.
- **No match** fails open to the full catalog by default (`on_no_match="all"`);
  use `on_no_match="pinned"` to send only pinned tools instead.
- **Prompt caching**: results come back in catalog order, so the same tool set
  always serializes identically.

## Match by meaning, not just keywords

Keywords can't connect "revenue", "money we made" or "earnings" to a tool
described as "Sales totals and order counts". Pass an embedder and the router
also ranks tools by meaning:

```python
from mcp_tool_router import ToolRouter
from mcp_tool_router.embedders import fastembed_embedder

router = ToolRouter(tools, max_tools=5, embed=fastembed_embedder())
router.select("what is revenue in last 5 days")  # -> [..., get_sales_data, ...]
```

`fastembed_embedder()` runs `snowflake/snowflake-arctic-embed-s` locally on
the CPU (about 130 MB, downloaded on first use, roughly 20 ms per question).
Any function with the signature `embed(texts, *, query) -> vectors` works, so
you can use your provider's embeddings instead:

```python
def embed(texts, *, query):
    response = openai_client.embeddings.create(model="text-embedding-3-small", input=list(texts))
    return [item.embedding for item in response.data]


router = ToolRouter(tools, embed=embed)
```

How the two signals combine: the keyword ranking and the meaning ranking are
merged with Reciprocal Rank Fusion. A tool found both ways ranks highest, and
a tool found only one way still makes the list. Tool descriptions are embedded
once, on the first question; each question is embedded once and cached.

There is always a "nearest" tool by meaning, so with an embedder the router
fills its `max_tools` budget on most questions. Set `min_similarity` (tune it
with `eval`) if you want unrelated questions to fall through to `on_no_match`.

## Use it inside a FastMCP server

```python
from fastmcp import FastMCP
from mcp_tool_router import RouterSearchTransform

mcp = FastMCP("My server")
mcp.add_transform(RouterSearchTransform(max_results=8, always_visible=["ping"]))
# list_tools now returns ping, search_tools and call_tool
```

Same interface as FastMCP's built-in search transforms, with better matching.
For the query `inventroy worth`, FastMCP's `BM25SearchTransform` returns no
tools and `RouterSearchTransform` returns `get_inventory_value`. This is
checked in `tests/test_transform.py`. Pass `embed=` for semantic search too.

## Proxy existing MCP servers

```bash
mcp-tool-router serve --config claude_desktop_config.json --max-results 8 --semantic
```

Point your client at the proxy instead of the individual servers. It sees two
tools, `search_tools` and `call_tool`, in place of the whole catalog.

## Measure before you trust it

Write labelled questions as CSV (`question,expected`, with `|` between several
expected tools) or JSONL, then:

```bash
mcp-tool-router eval --tools examples/tools.json --cases examples/cases.csv --max-tools 3
mcp-tool-router eval --tools examples/tools.json --cases examples/cases.csv --max-tools 3 --semantic
```

On the bundled toy example (8 tools, 17 questions, 5 of them phrased with
words the descriptions don't contain):

| | recall | no match (sent everything) | wrong tool | tools per question |
|---|---|---|---|---|
| keywords only | 94.1% | 4 | 1 | 2.8 of 8 |
| keywords + meaning (`--semantic`) | 100% | 0 | 0 | 3.0 of 8 |

The example is a toy; run `eval` on your own tools and real user questions.
Use `--config mcp.json` to fetch the tools from live servers instead of a file,
and `--min-recall 0.95` to fail a CI job when routing starts dropping tools.

Watch the **no match** line. A question that matches nothing fails open to the
whole catalog, which counts as a hit, so recall alone can look perfect while
the token savings disappear.

## How it works

1. **Tokenize**: names, descriptions, parameter names and descriptions, and enum
   values are split into words (snake_case and camelCase aware, Unicode
   normalized, stopwords removed). Indexing enum values makes tools like
   `get_metric(metric="tax")` findable by "tax".
2. **Match**: each question word matches catalog words exactly (1.0), by
   prefix ("sale"/"sales", 0.9), or by typo similarity ("inventroy", 0.7).
   `match="exact"` or `"prefix"` turns the looser kinds off.
3. **Rank**: Okapi BM25. Words found in few tools count for more than words
   found in many, and shorter descriptions win ties.
4. **Meaning** (optional): cosine similarity between the question's embedding
   and each tool's, fused with the BM25 ranking by Reciprocal Rank Fusion.

## Limits

- Without an embedder, matching is lexical: "revenue" won't find a sales tool.
  Pass `embed=` (or `--semantic`), or put your users' words in the descriptions.
- The default embedding model is English. Pass a multilingual model name to
  `fastembed_embedder()` for other languages. The stopword list is English too.
- Small embedding models separate close topics by thin margins. Route to a
  handful of tools (`max_tools` 3 to 8), not one, and check with `eval`.
- Payload sizes are characters / 4, a rough token estimate. Exact counts
  need your model's tokenizer.
- `RouterSearchTransform` subclasses FastMCP's search-transform hooks, so the
  extra is pinned to `fastmcp>=4,<5`.

## Development

```bash
uv sync --all-extras
uv run pytest --cov      # unit, property-based (Hypothesis) and end-to-end tests
uv run pytest -m model   # also run the real embedding model (downloads ~130 MB)
uv run ruff check . && uv run ruff format --check .
```

The end-to-end tests start a real FastMCP server in a subprocess and route to
it through the proxy.

## License

MIT
