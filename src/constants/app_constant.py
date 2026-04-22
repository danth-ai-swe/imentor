from pathlib import Path

COLLECTION_NAME = "imt_kb_v1"
DENSE_EMBEDDING_DIM = 1536
BM25_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
BM25_OPTIONS = {"language": "none", "ascii_folding": True, "tokenizer": "multilingual"}

MAX_INPUT_CHARS = 2_000

CHAT_HISTORY_TIMEOUT = 10

MAX_RECENT_HISTORY_ENTRIES = 10

# ── Config ───────────────────────────────────────────────────────────────────
_HTTP_TIMEOUT = 5  # fail-fast
TOP_URLS = 2
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
EMBED_DIM = 1536
FAISS_TTL_SECONDS = 86_400  # 24 h

DEFAULT_MAX_RESULTS = 3
DEFAULT_ENGINES = ["google", "bing", "duckduckgo"]
DEFAULT_CATEGORIES = ["general"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INGEST_DIR = DATA_DIR / "ingest"
T_ZIP = DATA_DIR / "t.7z"
TT_ZIP = DATA_DIR / "tt.7z"
STORAGE_FAQ_TEMPLATE = DATA_DIR / "storage" / "iMentor_FAQ_Template.pdf"
PREPARES_DIR = DATA_DIR / "prepares"
QUIZ_DIR = DATA_DIR / "quiz"
PDFS_DIR = DATA_DIR / "output" / "pdfs"
METADATA_NODE_XLSX = DATA_DIR / "metadata_node.xlsx"
METADATA_NODE_JSON = DATA_DIR / "metadata_node.json"

QUIZ_KEYWORDS: set[str] = {
    "quiz"
}
VECTOR_SEARCH_TOP_K: int = 2
NEIGHBOR_PREV_INDEX: int = -1  # phần tử cuối của list "previous" = chunk liền trước
NEIGHBOR_NEXT_INDEX: int = 0  # phần tử đầu của list "next"     = chunk liền sau
INTENT_CORE_KNOWLEDGE: str = "core_knowledge"
INTENT_OFF_TOPIC: str = "off_topic"
INTENT_QUIZ: str = "quiz"
OVERALL_CORE_KNOWLEDGE: str = "overall_course_knowledge"

OVERALL_COLLECTION_NAME = "overall_course"
OVERALL_INGEST_DIR = DATA_DIR / "ingest_overall"

OFF_TOPIC_RESPONSE_MAP = {
    "Vietnamese": ("""
Insuripedia là một trợ lý AI chuyên về kiến thức bảo hiểm, 
vì vậy Insuripedia không thể trả lời câu hỏi của bạn về chủ đề đó. 
Bạn có câu hỏi nào liên quan đến bảo hiểm không?
""".strip()),
    "English": ("""
Insuripedia is an AI assistant specializing in insurance 
knowledge so Insuripedia cannot answer your question about 
that topic. Do you have a question related to insurance?
""".strip()),
    "Japanese": ("""
Insuripediaは保険知識に特化したAIアシスタントです。
そのため、そのトピックに関するご質問にはお答えできません。
保険に関するご質問はありますか？
""".strip())
}
INPUT_TOO_LONG_RESPONSE = """
Your question is a bit long. Please try to shorten it to 
under {max_chars:,} characters to get the best results.
"""
UNSUPPORTED_LANGUAGE_MSG = (
    "Sorry, Insuripedia can't understand your language."
)
NO_RESULT_RESPONSE_MAP = {
    "Vietnamese": ("""
Xin lỗi. Insuripedia không thể tìm thấy thông tin nào về 
chủ đề bảo hiểm này trong kho dữ liệu.
""".strip()),
    "English": ("""
Sorry. Insuripedia cannot find any information about this 
insurance-related topic in the knowledge hub.
""".strip()),
    "Japanese": ("""
申し訳ありませんが、この保険関連のトピックについて
ナレッジベース内に情報が見つかりませんでした。
""".strip())
}
