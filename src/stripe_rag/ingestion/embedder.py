"""Dense (OpenAI text-embedding-3-small) and sparse (BM42 via fastembed) embedders."""
from __future__ import annotations

import logging

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
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
        results = await self._embed_batch([text])
        return results[0]


class SparseEmbedder:
    _MODEL_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"

    def __init__(self) -> None:
        # Lazy import so the model is only downloaded when actually used
        from fastembed import SparseTextEmbedding  # type: ignore[import-untyped]

        logger.info(f"Loading sparse model: {self._MODEL_NAME}")
        self._model = SparseTextEmbedding(self._MODEL_NAME)

    def embed_one(self, text: str) -> dict[int, float]:
        results = list(self._model.embed([text]))
        emb = results[0]
        return dict(zip(emb.indices.tolist(), emb.values.tolist(), strict=False))

    def embed_all(self, texts: list[str]) -> list[dict[int, float]]:
        output: list[dict[int, float]] = []
        for emb in tqdm(self._model.embed(texts), total=len(texts), desc="Sparse embed"):
            output.append(dict(zip(emb.indices.tolist(), emb.values.tolist(), strict=False)))
        return output
