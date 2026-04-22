from typing import Dict, List, Optional

import numpy as np

from src.utils.logger_utils import alog_method_call


class Route:
    def __init__(self, name: str = None, samples: List[str] = None):
        self.name = name
        self.samples = samples or []


class SemanticRouter:
    def __init__(
            self,
            routes: List[Route],
            routes_embedding: Dict[str, np.ndarray],  # đã normalized sẵn
    ):
        self.routes = routes
        self.routes_embedding = routes_embedding  # shape (N, dim), L2-normalized

    @classmethod
    async def abuild(
            cls,
            routes: List[Route],
            embedder,
            precomputed_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> "SemanticRouter":
        routes_embedding: Dict[str, np.ndarray] = {}

        for route in routes:
            if precomputed_embeddings and route.name in precomputed_embeddings:
                raw = precomputed_embeddings[route.name]
            else:
                raw = await cls._aencode(embedder, route.samples)
            # Pre-normalize một lần duy nhất lúc build
            routes_embedding[route.name] = cls._normalize(raw)

        return cls(routes=routes, routes_embedding=routes_embedding)

    @staticmethod
    async def _aencode(embedder, texts: List[str]) -> np.ndarray:
        vecs = await embedder.aembed_documents(texts)
        return np.array(vecs)

    @alog_method_call
    async def aguide(
            self,
            query_vec: np.ndarray,  # shape (1, dim), chưa normalize
            top_k_samples: int = 15,
    ) -> tuple[float, str]:
        # Normalize query một lần
        query_norm = self._normalize(query_vec)  # (1, dim)

        best_score = -1.0
        best_name = self.routes[0].name

        for route in self.routes:
            # routes_embedding đã normalized sẵn từ lúc build
            sims = np.dot(self.routes_embedding[route.name], query_norm.T).flatten()
            k = min(top_k_samples, len(sims))
            # np.partition nhanh hơn np.sort cho top-k
            top_k_sims = np.partition(sims, -k)[-k:]
            score = float(top_k_sims.mean())
            if score > best_score:
                best_score = score
                best_name = route.name

        return best_score, best_name

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return matrix / norms
