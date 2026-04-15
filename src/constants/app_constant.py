from pathlib import Path

COLLECTION_NAME = "imt_kb_v10"
DENSE_EMBEDDING_DIM = 1536
BM25_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
BM25_OPTIONS = {"language": "none", "ascii_folding": True, "tokenizer": "multilingual"}

MAX_INPUT_CHARS = 2_000

CHAT_HISTORY_TIMEOUT = 10

MAX_RECENT_HISTORY_ENTRIES = 6

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INGEST_DIR = DATA_DIR / "ingest"
T_ZIP = DATA_DIR / "t.7z"
TT_ZIP = DATA_DIR / "tt.7z"
PREPARES_DIR = DATA_DIR / "prepares"
QUIZ_DIR = DATA_DIR / "quiz"
PDFS_DIR = DATA_DIR / "output" / "pdfs"
METADATA_NODE_XLSX = DATA_DIR / "metadata_node.xlsx"

QUIZ_KEYWORDS: set[str] = {
    "quiz"
}
