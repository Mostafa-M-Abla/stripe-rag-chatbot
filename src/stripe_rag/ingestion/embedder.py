"""Dense (OpenAI text-embedding-3-small) and sparse (BM42 via fastembed) embedders."""
from __future__ import annotations

import logging

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


class OpenAIEmbedder:
    """Dense embedder backed by the OpenAI embeddings API (``text-embedding-3-large`` by default).

    Requests are batched in groups of ``BATCH_SIZE`` and retried on transient errors
    with tenacity exponential back-off.
    """

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        """Initialise the async OpenAI client.

        Args:
            api_key: OpenAI API key.
            model: Embedding model name (defaults to ``text-embedding-3-small``; production
                uses ``text-embedding-3-large`` as set in ``Settings``).
        """
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch with up to 3 retries; returns one float vector per text."""
        response = await self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

    async def embed_all(self, texts: list[str]) -> list[list[float]]:
        """Embed all texts in BATCH_SIZE chunks, logging progress."""
        all_embeddings: list[list[float]] = []
        batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
        for i, batch in enumerate(batches):
            logger.info(f"Embedding batch {i + 1}/{len(batches)} ({len(batch)} texts)")
            embeddings = await self._embed_batch(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    async def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper: embed a single string (used for query embedding at retrieval time)."""
        results = await self._embed_batch([text])
        return results[0]


class SparseEmbedder:
    """BM42 sparse encoder via fastembed; runs locally with no API key required.

    The fastembed model is downloaded on first instantiation and cached on disk.
    Output is a ``{token_id: weight}`` sparse dict compatible with Qdrant's
    ``SparseVector`` format.
    """

    _MODEL_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"

    def __init__(self) -> None:
        """Load the BM42 fastembed model (download on first use, cached thereafter)."""
        # Lazy import so the model is only downloaded when actually used
        from fastembed import SparseTextEmbedding  # type: ignore[import-untyped]

        logger.info(f"Loading sparse model: {self._MODEL_NAME}")
        self._model = SparseTextEmbedding(self._MODEL_NAME)

    def embed_one(self, text: str) -> dict[int, float]:
        """Return the sparse BM42 vector for a single string (used for query embedding)."""
        results = list(self._model.embed([text]))
        emb = results[0]
        return dict(zip(emb.indices.tolist(), emb.values.tolist(), strict=False))

    def embed_all(self, texts: list[str]) -> list[dict[int, float]]:
        """Batch-encode a list of texts; returns one ``{token_id: weight}`` dict per text."""
        output: list[dict[int, float]] = []
        for emb in tqdm(self._model.embed(texts), total=len(texts), desc="Sparse embed"):
            output.append(dict(zip(emb.indices.tolist(), emb.values.tolist(), strict=False)))
        return output
