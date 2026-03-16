import time

import numpy as np
from fastembed.rerank.cross_encoder import TextCrossEncoder


class Reranker():
    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-12-v2"):
        self._reranker = TextCrossEncoder(model_name=model_name)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1 / (1 + np.exp(-x))

    def __call__(self, query: str, passages: list[str]) -> tuple[list[float], list[str]]:
        # Get scores from the reranker model
        raw_scores = list(self._reranker.rerank(query, passages, batch_size=32))

        # Sort passages based on scores (descending)
        ranked_pairs = sorted(zip(raw_scores, passages), key=lambda x: x[0], reverse=True)

        ranked_scores = [float(self._sigmoid(score)) for score, _ in ranked_pairs]
        ranked_passages = [passage for _, passage in ranked_pairs]

        return ranked_scores, ranked_passages


def main():
    # Vietnamese query
    query = "The story of a female figure with a powerful historical influence."

    # English documents
    documents = [
        "A tired political journalist investigates the disappearance of a woman's missing son.",
        "A princess banished into the forest by her evil stepmother finds refuge with seven miners.",
        "A biographical film about the revolutionary Udham Singh planning his revenge.",
        "In 1431, Joan of Arc is put on trial for heresy by an ecclesiastical court.",
        "A programmer experiences a strange night after meeting a mysterious girl in a coffee shop.",
    ]

    # ── Load model ──────────────────────────────────────────
    print("Loading model...")
    t_load_start = time.perf_counter()

    reranker = Reranker()

    t_load_end = time.perf_counter()
    print(f"✓ Model loaded in {t_load_end - t_load_start:.3f}s\n")

    # ── Rerank ───────────────────────────────────────────────
    print("Reranking documents...")
    t_infer_start = time.perf_counter()

    ranked_scores, ranked_passages = reranker(query, documents)

    t_infer_end = time.perf_counter()

    infer_ms = (t_infer_end - t_infer_start) * 1000
    per_doc_ms = infer_ms / len(documents)

    print(f"✓ Inference done in {infer_ms:.1f} ms ({per_doc_ms:.1f} ms/doc)\n")

    # ── Results ──────────────────────────────────────────────
    print(f"Query (Vietnamese): {query}\n")
    print("─" * 60)
    print("Reranked results:\n")

    for i, (passage, score) in enumerate(zip(ranked_passages, ranked_scores), 1):
        print(f"{i}. Score = {score:.4f}")
        print(f"   {passage}\n")

    # ── Timing summary ───────────────────────────────────────
    total = t_infer_end - t_load_start

    print("─" * 60)
    print(f"Total time       : {total:.3f}s")
    print(f"Model load time  : {t_load_end - t_load_start:.3f}s")
    print(f"Inference time   : {infer_ms:.1f}ms")
    print(f"Documents tested : {len(documents)}")


if __name__ == "__main__":
    main()
