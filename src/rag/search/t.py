"""
web_rag_pipeline.py
────────────────────────────────────────────────────────────────────────────
Full pipeline:
  question
    → SearxNG search (5 results)
    → semantic rank snippets vs query  (embedding cosine)
    → top-3 URLs
    → FAISS cache check per URL
      ├─ hit  → search chunks in FAISS directly
      └─ miss → crawl (trafilatura, parallel) → chunk → embed → insert FAISS
    → top-5 chunks
    → build prompt
    → LLM answer
────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import logging
import pickle
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import faiss
import numpy as np
import trafilatura
from langchain_community.utilities import SearxSearchWrapper

from src.rag.llm.chat_llm import get_openai_chat_client
from src.rag.llm.embedding_llm import get_openai_embedding_client

logger = logging.getLogger(__name__)

SEARX_HOST = "http://localhost:8080"
DEFAULT_MAX_RESULTS = 5
DEFAULT_ENGINES = ["google", "bing", "duckduckgo"]
DEFAULT_CATEGORIES = ["general"]

TOP_URLS = 3
TOP_CHUNKS = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
EMBED_DIM = 1536

import os
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])

FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "faiss", "faiss_index.bin")
FAISS_META_PATH = os.path.join(PROJECT_ROOT, "faiss", "faiss_meta.pkl")

os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
os.makedirs(os.path.dirname(FAISS_META_PATH), exist_ok=True)

# Type aliases (Python 3.12+)
type RawResult = dict[str, Any]
type ExtractedResult = dict[str, Any]
type SourceItem = dict[str, Any]

_WEB_ANSWER_PROMPT = """\
You are a helpful assistant answering a user question using web search results.

Answer the question thoroughly based on the provided web content.
- Use only the information from the retrieved web content below.
- Respond in {detected_language}.
- Be concise and structured. Use bullet points or numbered lists if appropriate.
- Do NOT fabricate information not present in the sources.
- Do NOT list or mention sources in your answer — they are provided separately.

## Web Content
{web_context}

## User Question
{user_input}
"""

# ══════════════════════════════════════════════════════════════════════════════
# 1. SEARXNG CLIENT
# ══════════════════════════════════════════════════════════════════════════════

_searx_client: SearxSearchWrapper | None = None


def get_searx_client() -> SearxSearchWrapper:
    global _searx_client
    if _searx_client is None:
        _searx_client = SearxSearchWrapper(
            searx_host=SEARX_HOST,
            engines=DEFAULT_ENGINES,
            categories=DEFAULT_CATEGORIES,
            k=DEFAULT_MAX_RESULTS,
            params={"language": "en", "format": "json"},
        )
    return _searx_client


async def asearch_and_extract(
        search_query: str,
        *,
        max_results: int = DEFAULT_MAX_RESULTS,
        engines: list[str] | None = None,
        categories: list[str] | None = None,
        query_suffix: str = "",
) -> list[ExtractedResult]:
    """
    Query SearxNG and return a deduplicated list of result dicts.
    Each dict: {url, title, snippet, engines, category, hostname}
    """
    client = get_searx_client()

    try:
        raw: list[RawResult] = await client.aresults(
            query=search_query,
            num_results=max_results,
            engines=engines or DEFAULT_ENGINES,
            query_suffix=query_suffix,
            categories=",".join(categories or DEFAULT_CATEGORIES),
        )
    except Exception:
        logger.exception("SearxNG aresults() failed for query: %r", search_query)
        return []

    if not raw or "Result" in raw[0]:
        logger.warning("SearxNG: no results for query: %r", search_query)
        return []

    extracted: list[ExtractedResult] = []
    seen_urls: set[str] = set()

    for r in raw:
        url: str = r.get("link", "").strip()
        title: str = r.get("title", "").strip()
        snippet: str = r.get("snippet", "").strip()
        eng: list = r.get("engines", [])
        category: str = r.get("category", "")

        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        extracted.append({
            "url": url,
            "title": title,
            "snippet": snippet,
            "engines": eng,
            "category": category,
            "hostname": urlparse(url).netloc,
        })

    logger.info("SearxNG: query=%r → %d results", search_query, len(extracted))
    return extracted


# ══════════════════════════════════════════════════════════════════════════════
# 2. FAISS STORE
# ══════════════════════════════════════════════════════════════════════════════

class FaissStore:
    """
    Cosine-similarity store backed by IndexFlatIP + L2-normalised vectors.

    Persistence:
      faiss_index.bin  — raw FAISS index
      faiss_meta.pkl   — {metadata: List[dict], url_index: dict[url → list[int]]}
    """

    def __init__(self):
        if Path(FAISS_INDEX_PATH).exists() and Path(FAISS_META_PATH).exists():
            self.index: faiss.IndexFlatIP = faiss.read_index(FAISS_INDEX_PATH)
            with open(FAISS_META_PATH, "rb") as f:
                saved = pickle.load(f)
            self.metadata: list[dict] = saved["metadata"]
            self.url_index: dict[str, list[int]] = saved["url_index"]
            logger.info("FAISS loaded — %d vectors", self.index.ntotal)
        else:
            self.index = faiss.IndexFlatIP(EMBED_DIM)
            self.metadata = []
            self.url_index = {}

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self):
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(FAISS_META_PATH, "wb") as f:
            pickle.dump({"metadata": self.metadata, "url_index": self.url_index}, f)

    # ── write ────────────────────────────────────────────────────────────────

    def add_chunks(self, url: str, chunks: list[str], embeddings: list[list[float]]):
        vectors = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)

        start_id = len(self.metadata)
        self.index.add(vectors)
        now = time.time()
        ids: list[int] = []
        for i, chunk in enumerate(chunks):
            chunk_id = start_id + i
            self.metadata.append({"id": chunk_id, "url": url, "text": chunk, "timestamp": now})
            ids.append(chunk_id)

        self.url_index[url] = ids
        self.save()
        logger.info("FAISS: +%d chunks for %s  (total=%d)", len(ids), url, self.index.ntotal)

    # ── read ─────────────────────────────────────────────────────────────────

    def has_url(self, url: str) -> bool:
        return bool(self.url_index.get(url))

    def search(
            self,
            query_emb: list[float],
            top_k: int = TOP_CHUNKS,
            url_filter: str | None = None,
    ) -> list[dict]:
        """
        Search globally, then filter by url_filter if given.
        Over-fetches (×10) to compensate for post-filter drop.
        """
        if self.index.ntotal == 0:
            return []

        k = min(top_k * (10 if url_filter else 1), self.index.ntotal)
        q = np.array([query_emb], dtype="float32")
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx]
            if url_filter and meta["url"] != url_filter:
                continue
            results.append({**meta, "score": float(score)})
            if len(results) >= top_k:
                break

        return results


# Module-level singleton
_faiss_store: FaissStore | None = None


def get_faiss_store() -> FaissStore:
    global _faiss_store
    if _faiss_store is None:
        _faiss_store = FaissStore()
    return _faiss_store


# ══════════════════════════════════════════════════════════════════════════════
# 3. SEMANTIC URL RANKER
# ══════════════════════════════════════════════════════════════════════════════

async def rank_urls_by_snippet(
        query: str,
        results: list[ExtractedResult],
        top_k: int = TOP_URLS,
) -> list[ExtractedResult]:
    """
    Embed query + snippets in one batch call; rank by cosine similarity.
    Returns top_k results with an added 'semantic_score' key.
    """
    embedder = get_openai_embedding_client()

    texts = [
        f"{r.get('title', '')}.\n{r.get('snippet', '')}"
        for r in results
    ]
    embeddings = await embedder.aembed_documents([query, *texts])

    q_emb = np.array(embeddings[0], dtype="float32")
    q_emb /= np.linalg.norm(q_emb) + 1e-9

    scored: list[ExtractedResult] = []
    for i, result in enumerate(results):
        s_emb = np.array(embeddings[i + 1], dtype="float32")
        s_emb /= np.linalg.norm(s_emb) + 1e-9

        score = float(np.dot(q_emb, s_emb))

        scored.append({
            **result,
            "semantic_score": score,
        })

    scored.sort(key=lambda x: x["semantic_score"], reverse=True)
    top = scored[:top_k]

    logger.info(
        "URL ranking: %s",
        [(r["url"], round(r["semantic_score"], 3)) for r in top],
    )
    return top


# ══════════════════════════════════════════════════════════════════════════════
# 4. CRAWLER  (trafilatura + httpx fallback + snippet fallback)
# ══════════════════════════════════════════════════════════════════════════════

import httpx

# Domains known to block bots or require login — skip crawl, use snippet only
_BLOCKED_DOMAINS = {
    "facebook.com", "www.facebook.com",
    "instagram.com", "twitter.com", "x.com",
    "linkedin.com", "tiktok.com",
    "accounts.google.com",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_HTTP_TIMEOUT = 15  # seconds


def _is_blocked_domain(url: str) -> bool:
    hostname = urlparse(url).netloc.lower().lstrip("www.")
    return any(hostname == d or hostname.endswith("." + d) for d in _BLOCKED_DOMAINS)


async def _fetch_html_httpx(url: str) -> str | None:
    """Fallback fetcher với browser-like headers."""
    try:
        async with httpx.AsyncClient(
                headers=_HEADERS,
                follow_redirects=True,
                timeout=_HTTP_TIMEOUT,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text
    except Exception as e:
        logger.warning("httpx fetch failed %s: %s", url, e)
        return None


async def _crawl_single(url: str) -> tuple[str, str | None]:
    """
    Returns (url, markdown_text | None).

    Strategy:
      1. Skip blocked domains ngay lập tức
      2. Thử trafilatura.fetch_url (có user-agent setting)
      3. Nếu empty → fallback httpx với browser headers
      4. Extract markdown từ raw HTML
    """
    if _is_blocked_domain(url):
        logger.info("Blocked domain, skipping crawl: %s", url)
        return url, None

    loop = asyncio.get_event_loop()

    # ── attempt 1: trafilatura built-in fetcher ──────────────────────────────
    try:
        downloaded = await loop.run_in_executor(
            None,
            lambda: trafilatura.fetch_url(
                url,
                config=trafilatura.settings.use_config(),  # default config
            ),
        )
    except Exception:
        downloaded = None

    # ── attempt 2: httpx fallback nếu trafilatura trả None ───────────────────
    if not downloaded:
        logger.debug("trafilatura empty, trying httpx: %s", url)
        downloaded = await _fetch_html_httpx(url)

    if not downloaded:
        logger.warning("Skipped (empty after fallback): %s", url)
        return url, None

    # ── extract markdown ──────────────────────────────────────────────────────
    try:
        text = await loop.run_in_executor(
            None,
            lambda: trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                output_format="markdown",
                favor_recall=True,  # bắt nhiều content hơn
            ),
        )
    except Exception:
        text = None

    if not text:
        logger.warning("Skipped (extract failed): %s", url)
        return url, None

    logger.info("Crawled OK: %s (%d chars)", url, len(text))
    return url, text


async def crawl_urls_parallel(urls: list[str]) -> dict[str, str]:
    pairs = await asyncio.gather(*[_crawl_single(url) for url in urls])
    return {url: text for url, text in pairs if text}


# ══════════════════════════════════════════════════════════════════════════════
# 5. CHUNKER
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text(
        text: str,
        size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
        min_len: int = 40,
) -> list[str]:
    """Sliding-window character chunker with overlap."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start: start + size].strip()
        if len(chunk) >= min_len:
            chunks.append(chunk)
        start += size - overlap
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 6. INGEST  (crawl → chunk → embed → FAISS)
# ══════════════════════════════════════════════════════════════════════════════
async def ingest_urls_batch(url_text_map: dict[str, str]):
    embedder = get_openai_embedding_client()
    store = get_faiss_store()

    all_chunks: list[str] = []
    chunk_url_map: list[str] = []

    for url, text in url_text_map.items():
        chunks = chunk_text(text)
        for c in chunks:
            all_chunks.append(c)
            chunk_url_map.append(url)

    if not all_chunks:
        logger.warning("No chunks to ingest (batch)")
        return

    embeddings = await embedder.aembed_documents(all_chunks)

    # group lại theo URL
    url_to_chunks: dict[str, list[str]] = {}
    url_to_embs: dict[str, list[list[float]]] = {}

    for chunk, emb, url in zip(all_chunks, embeddings, chunk_url_map):
        url_to_chunks.setdefault(url, []).append(chunk)
        url_to_embs.setdefault(url, []).append(emb)

    for url in url_to_chunks:
        store.add_chunks(url, url_to_chunks[url], url_to_embs[url])


async def _ingest_url(url: str, text: str):
    embedder = get_openai_embedding_client()
    chunks = chunk_text(text)
    if not chunks:
        logger.warning("No chunks produced for %s", url)
        return
    embeddings = await embedder.aembed_documents(chunks)
    get_faiss_store().add_chunks(url, chunks, embeddings)


# ══════════════════════════════════════════════════════════════════════════════
# 7. RETRIEVER  (cache-aware: FAISS hit → skip crawl)
# ══════════════════════════════════════════════════════════════════════════════
async def retrieve_chunks(
        urls: list[str],
        query_emb: list[float],
        top_k: int = TOP_CHUNKS,
        # truyền thêm search results để fallback
        url_snippets: dict[str, str] | None = None,
) -> list[dict]:
    store = get_faiss_store()

    to_crawl = [url for url in urls if not store.has_url(url)]
    if to_crawl:
        crawled = await crawl_urls_parallel(to_crawl)

        # snippet fallback cho URL crawl thất bại
        if url_snippets:
            for url in to_crawl:
                if url not in crawled and url in url_snippets:
                    snippet = url_snippets[url]
                    if snippet:
                        crawled[url] = snippet
                        logger.info("Using snippet fallback for: %s", url)

        await ingest_urls_batch(crawled)

    # global search trước
    all_chunks = store.search(query_emb, top_k=top_k * 5)

    # filter theo top_urls
    all_chunks = [c for c in all_chunks if c["url"] in urls]

    seen: set[str] = set()
    unique: list[dict] = []
    for chunk in sorted(all_chunks, key=lambda x: x["score"], reverse=True):
        sig = chunk["text"][:80]
        if sig not in seen:
            seen.add(sig)
            unique.append(chunk)

    return unique[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# 9. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

async def web_rag_answer(user_input: str) -> dict[str, Any]:
    """
    End-to-end RAG pipeline. Returns:
      {
        "answer":      str,
        "sources":     [{"url", "title", "score"}, ...],
        "chunks_used": int,
      }
    """
    embedder = get_openai_embedding_client()
    chat = get_openai_chat_client()

    # 1. Search ────────────────────────────────────────────────────────────────
    raw_results = await asearch_and_extract(user_input)
    if not raw_results:
        return {"answer": "Không tìm thấy kết quả web cho câu hỏi này.", "sources": [], "chunks_used": 0}

    # 2. Rank URLs semantically ────────────────────────────────────────────────
    top_results = await rank_urls_by_snippet(user_input, raw_results, top_k=TOP_URLS)
    top_urls = [r["url"] for r in top_results]

    # 3. Embed query once (reused for retrieval) ───────────────────────────────
    query_emb = await embedder.aembed_query(user_input)

    # 4. Retrieve chunks (cache-aware) ─────────────────────────────────────────
    url_snippets = {r["url"]: r.get("snippet", "") for r in top_results}

    chunks = await retrieve_chunks(
        top_urls,
        query_emb,
        top_k=TOP_CHUNKS,
        url_snippets=url_snippets,  # ← thêm dòng này
    )
    if not chunks:
        return {"answer": "Không truy xuất được nội dung từ các trang web.", "sources": top_urls, "chunks_used": 0}

    # 5. Build prompt ──────────────────────────────────────────────────────────
    web_context = "\n\n---\n\n".join(
        f"[{i + 1}] score={c['score']:.3f} | {c['url']}\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    prompt = _WEB_ANSWER_PROMPT.format(
        detected_language="ENGLISH",
        web_context=web_context,
        user_input=user_input,
    )

    # 6. LLM answer ────────────────────────────────────────────────────────────
    answer = await chat.ainvoke(prompt)

    return {
        "answer": answer,
        "sources": [
            {"url": r["url"], "title": r.get("title", ""), "score": r["semantic_score"]}
            for r in top_results
        ],
        "chunks_used": len(chunks),
    }


async def main():
    result = await web_rag_answer("What are the latest advancements in AI research?")
    print(result["answer"])
    print(result["sources"])


if __name__ == '__main__':
    asyncio.run(main())
