from pathlib import Path

COLLECTION_NAME = "imt_kb_v3"
DENSE_EMBEDDING_DIM = 1536
BM25_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
BM25_OPTIONS = {"language": "none", "ascii_folding": True, "tokenizer": "multilingual"}

MAX_INPUT_CHARS = 15_000
RELEVANCE_SCORE_THRESHOLD = 0.70

CHARS_PER_TOKEN = 4
CHAT_HISTORY_TOKEN_BUDGET = 2_000
CHAT_HISTORY_TIMEOUT = 10
MAX_ASSISTANT_RESPONSE_CHARS = 500
TRUNCATION_SUFFIX = "…"

MAX_RECENT_HISTORY_ENTRIES = 6

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
INGEST_DIR = DATA_DIR / "ingest"
INGEST_ZIP = DATA_DIR / "ingest.zip"
PREPARES_DIR = DATA_DIR / "prepares"
PDFS_DIR = DATA_DIR / "output" / "pdfs"
METADATA_NODE_XLSX = DATA_DIR / "metadata_node.xlsx"

DEFAULT_CLEANED_DIR = (PROJECT_ROOT / "data" / "imt-data-process-v2" / "cleaned")
DOCKER_CLEANED_DIR = Path("/data/imt-data-process-v2/cleaned")

CLASSIFY_WORKERS = 4

QUIZ_KEYWORDS: set[str] = {
    "quiz", "quiz me", "test me", "mock exam", "practice questions",
    "practice test", "flashcard", "flashcards", "test my knowledge",
    "kiểm tra", "làm quiz", "làm bài quiz", "trắc nghiệm",
    "câu hỏi ôn tập", "thi thử", "ôn tập", "làm bài kiểm tra",
    "cho tôi làm quiz", "bài tập", "câu hỏi trắc nghiệm"
}
