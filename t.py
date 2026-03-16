
from fastembed.rerank.cross_encoder import TextCrossEncoder
import time
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def main():
    # Query tiếng Việt
    query = "Câu chuyện về một nhân vật nữ có tầm ảnh hưởng lịch sử mạnh mẽ"

    # Documents tiếng Việt
    documents = [
        "Một nhà báo chính trị mệt mỏi điều tra hành trình tìm kiếm đứa con trai mất tích của một người phụ nữ.",
        "Bị người mẹ kế độc ác đày vào rừng, một công chúa tìm được nơi nương náu bên bảy người thợ mỏ.",
        "Bộ phim tiểu sử về nhà cách mạng Udham Singh lên kế hoạch trả thù.",
        "Năm 1431, Jeanne d'Arc bị đưa ra xét xử về tội dị giáo bởi các thẩm phán giáo hội.",
        "Một lập trình viên trải qua một đêm kỳ lạ sau khi gặp một cô gái trong quán cà phê.",
    ]

    # ── Load model ──────────────────────────────────────────
    print("Loading model...")
    t_load_start = time.perf_counter()

    reranker = TextCrossEncoder(
        model_name="jinaai/jina-reranker-v2-base-multilingual"
    )

    t_load_end = time.perf_counter()
    print(f"  ✓ Model loaded in {t_load_end - t_load_start:.3f}s\n")

    # ── Rerank ───────────────────────────────────────────────
    print("Reranking...")
    t_infer_start = time.perf_counter()

    scores = list(reranker.rerank(query, documents, batch_size=32))

    t_infer_end = time.perf_counter()
    infer_ms = (t_infer_end - t_infer_start) * 1000
    per_doc_ms = infer_ms / len(documents)
    print(f"  ✓ Inference done in {infer_ms:.1f}ms  ({per_doc_ms:.1f}ms/doc)\n")

    # ── Sort + print ─────────────────────────────────────────
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    print(f"Query: {query}\n")
    print("─" * 60)
    print("Reranked results:\n")

    for i, (doc, score) in enumerate(ranked, 1):
        prob = sigmoid(score)
        print(f"{i}. Score = {prob:.4f}")
        print(f"   {doc}\n")

    # ── Tổng kết thời gian ───────────────────────────────────
    total = t_infer_end - t_load_start
    print("─" * 60)
    print(f"⏱  Tổng thời gian  : {total:.3f}s")
    print(f"   Load model      : {t_load_end - t_load_start:.3f}s")
    print(f"   Inference       : {infer_ms:.1f}ms")
    print(f"   Docs processed  : {len(documents)}")


if __name__ == "__main__":
    main()
