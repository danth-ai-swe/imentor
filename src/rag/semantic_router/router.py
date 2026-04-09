import asyncio
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.utils.logger_utils import alog_method_call

_EMBEDDINGS_PATH = Path(__file__).with_name("route_embeddings.npz")

_cached_embeddings: Optional[Dict[str, np.ndarray]] = None


def load_precomputed_embeddings() -> Optional[Dict[str, np.ndarray]]:
    global _cached_embeddings
    if _cached_embeddings is not None:
        return _cached_embeddings
    if not _EMBEDDINGS_PATH.exists():
        return None
    data = np.load(_EMBEDDINGS_PATH)
    _cached_embeddings = {key: data[key] for key in data.files}
    return _cached_embeddings


class Route:
    def __init__(self, name: str = None, samples: List = None):
        if samples is None:
            samples = []
        self.name = name
        self.samples = samples


class SemanticRouter:
    def __init__(
            self,
            embedding,
            routes: List[Route],
            precomputed_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.routes = routes
        self.embedding = embedding
        self.routes_embedding: Dict[str, np.ndarray] = {}

        for route in self.routes:
            if precomputed_embeddings and route.name in precomputed_embeddings:
                self.routes_embedding[route.name] = precomputed_embeddings[route.name]
            else:
                self.routes_embedding[route.name] = self._encode(route.samples)

    def _encode(self, texts):
        if hasattr(self.embedding, "encode"):
            return np.array(self.embedding.encode(texts), dtype=float)
        if hasattr(self.embedding, "embed_documents"):
            return np.array(self.embedding.embed_documents(list(texts)), dtype=float)
        raise ValueError("Unsupported embedding adapter for SemanticRouter")

    async def _aencode(self, texts):
        if hasattr(self.embedding, "aembed_documents"):
            vecs = await self.embedding.aembed_documents(list(texts))
            return np.array(vecs, dtype=float)
        if hasattr(self.embedding, "encode"):
            return np.array(
                await asyncio.to_thread(self.embedding.encode, texts), dtype=float
            )
        if hasattr(self.embedding, "embed_documents"):
            return np.array(
                await asyncio.to_thread(self.embedding.embed_documents, list(texts)),
                dtype=float,
            )
        raise ValueError("Unsupported embedding adapter for SemanticRouter (async)")

    @alog_method_call
    async def aguide(self, query, top_k_samples: int = 15):
        query_embedding = self._normalize(await self._aencode([query]))
        scores = []

        for route in self.routes:
            routes_embedding = self._normalize(self.routes_embedding[route.name])
            sims = np.dot(routes_embedding, query_embedding.T).flatten()
            k = min(top_k_samples, len(sims))
            top_k_sims = np.sort(sims)[-k:]
            score = np.mean(top_k_sims)
            scores.append((score, route.name))

        scores.sort(reverse=True)
        return scores[0]

    @staticmethod
    def _normalize(matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
