from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence, Optional, List

import numpy as np
from fastembed.rerank.cross_encoder import TextCrossEncoder


@dataclass
class RerankResult:
    document: str
    score: float
    index: int


class Reranker:
    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-12-v2"):
        self._reranker = TextCrossEncoder(model_name=model_name)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1 / (1 + np.exp(-x))

    def rerank(
            self,
            query: str,
            documents: Sequence[str],
            top_k: Optional[int] = None,
            min_score: Optional[float] = None,
    ) -> List[RerankResult]:

        # lấy raw score từ cross encoder
        raw_scores = list(self._reranker.rerank(query, list(documents), batch_size=32))

        # convert score
        scores = [float(self._sigmoid(s)) for s in raw_scores]

        # tạo result object
        results = [
            RerankResult(document=doc, score=score, index=i)
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]

        # sort theo score giảm dần
        results.sort(key=lambda x: x.score, reverse=True)

        # filter theo min_score
        if min_score is not None:
            results = [r for r in results if r.score >= min_score]

        # lấy top_k
        if top_k is not None:
            results = results[:top_k]

        return results


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()


if __name__ == '__main__':
    reranker = get_reranker()
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "Madrid is the capital of Spain."
    ]
    results = reranker.rerank(query, documents, top_k=2)
    for r in results:
        print(f"Document: {r.document}, Score: {r.score:.4f}, Index: {r.index}")
