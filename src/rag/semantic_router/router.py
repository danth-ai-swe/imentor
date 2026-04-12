from typing import Dict, List, Optional, Tuple

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
            routes_embedding: Dict[str, np.ndarray],
    ):
        """
        Không nhận embedder hay samples ở đây.
        Dùng SemanticRouter.abuild(...) để tạo instance đã encode xong.
        """
        self.routes = routes
        self.routes_embedding: Dict[str, np.ndarray] = routes_embedding

    @classmethod
    async def abuild(
            cls,
            routes: List[Route],
            embedder,  # AzureEmbeddingClient
            precomputed_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> "SemanticRouter":
        """
        Factory async — encode samples cho từng route chưa có precomputed.
        Gọi một lần trong lifespan, cache lại instance.
        """
        routes_embedding: Dict[str, np.ndarray] = {}

        for route in routes:
            if precomputed_embeddings and route.name in precomputed_embeddings:
                routes_embedding[route.name] = precomputed_embeddings[route.name]
            else:
                routes_embedding[route.name] = await cls._aencode(embedder, route.samples)

        return cls(routes=routes, routes_embedding=routes_embedding)

    @staticmethod
    async def _aencode(embedder, texts: List[str]) -> np.ndarray:
        vecs = await embedder.aembed_documents(texts)  # List[List[float]]
        return np.array(vecs)  # → shape: (N, dim)

    @alog_method_call
    async def aguide(
            self,
            query_vec: np.ndarray,
            top_k_samples: int = 15,
    ) -> Tuple[float, str]:
        query_embedding = self._normalize(query_vec)
        scores = []

        for route in self.routes:
            routes_embedding = self._normalize(self.routes_embedding[route.name])
            sims = np.dot(routes_embedding, query_embedding.T).flatten()
            k = min(top_k_samples, len(sims))
            top_k_sims = np.sort(sims)[-k:]
            score = float(np.mean(top_k_sims))
            scores.append((score, route.name))

        scores.sort(reverse=True)
        return scores[0]

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
