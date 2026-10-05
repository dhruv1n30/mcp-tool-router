"""Ready-made embedders for semantic routing.

Any function ``(texts, *, query) -> vectors`` works as ``ToolRouter(embed=...)``.
For example, with OpenAI::

    def embed(texts, *, query):
        response = client.embeddings.create(model="text-embedding-3-small", input=list(texts))
        return [item.embedding for item in response.data]
"""

from __future__ import annotations

from collections.abc import Sequence

from mcp_tool_router.router import Embedder

# Small (~130 MB), CPU-only, and the best of the small models we compared on
# meaning-only routing questions; see the README.
DEFAULT_MODEL = "snowflake/snowflake-arctic-embed-s"


def fastembed_embedder(model: str = DEFAULT_MODEL, **options: object) -> Embedder:
    """A local embedder using fastembed (ONNX, no API key, no GPU).

    The model downloads on first use. ``options`` go to
    ``fastembed.TextEmbedding`` (e.g. ``cache_dir``, ``threads``).
    Requires the ``semantic`` extra: ``pip install 'mcp-tool-router[semantic]'``.
    """
    from fastembed import TextEmbedding

    encoder = TextEmbedding(model, **options)

    def embed(texts: Sequence[str], *, query: bool) -> list[list[float]]:
        # Retrieval models embed questions and passages with different prefixes.
        vectors = encoder.query_embed(list(texts)) if query else encoder.passage_embed(list(texts))
        return [vector.tolist() for vector in vectors]

    return embed
