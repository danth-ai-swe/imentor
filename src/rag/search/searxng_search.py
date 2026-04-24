"""
web_rag_pipeline.py — Optimized v3
────────────────────────────────────────────────────────────────────────────
Changes vs v2:
 • Removed: rank_urls_by_keyword, _is_blocked_domain, TOP_CHUNKS, timeout waits
 • Removed: trafilatura.fetch_url fallback (skip immediately on httpx fail)
 • Removed: crawl semaphore (TOP_URLS=2, no need to throttle)
 • HTTP timeout 8 → 5 s, fail-fast no retry
 • trafilatura.extract runs in a shared ThreadPoolExecutor (max_workers=4)
 • FAISS save runs in executor (non-blocking pickle I/O)
 • Per-step timing logs for bottleneck tracing
 • Embedding batch is the single biggest cost — logged explicitly
────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import aiohttp
import faiss
import httpx
import numpy as np
import trafilatura
from langchain_community.utilities import SearxSearchWrapper
from langchain_community.utilities.searx_search import SearxResults

from src.config.app_config import get_app_config
from src.constants.app_constant import NO_RESULT_RESPONSE_MAP, _HTTP_TIMEOUT, DEFAULT_ENGINES, DEFAULT_CATEGORIES, \
    DEFAULT_MAX_RESULTS, EMBED_DIM, FAISS_TTL_SECONDS, CHUNK_SIZE, CHUNK_OVERLAP, TOP_URLS
from src.rag.llm.embedding_llm import get_openai_embedding_client
from src.utils.logger_utils import logger

config = get_app_config()
SEARX_HOST = config.SEARX_HOST
CF_ACCESS_CLIENT_ID = config.CF_ACCESS_CLIENT_ID
CF_ACCESS_CLIENT_SECRET = config.CF_ACCESS_CLIENT_SECRET
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "faiss", "faiss_index.bin")
FAISS_META_PATH = os.path.join(PROJECT_ROOT, "faiss", "faiss_meta.pkl")
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "CF-Access-Client-Id": CF_ACCESS_CLIENT_ID,
    "CF-Access-Client-Secret": CF_ACCESS_CLIENT_SECRET,
}

_WEB_ANSWER_PROMPT = """
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

# ── Shared thread pool for CPU-bound work (trafilatura.extract, FAISS save) ──
_cpu_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag-cpu")

# ── Shared httpx client ───────────────────────────────────────────────────────
_http_client: httpx.AsyncClient | None = None
_http_client_lock = asyncio.Lock()


async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        async with _http_client_lock:
            if _http_client is None or _http_client.is_closed:
                _http_client = httpx.AsyncClient(
                    headers=_HEADERS,
                    follow_redirects=True,
                    timeout=_HTTP_TIMEOUT,
                    limits=httpx.Limits(
                        max_connections=20,
                        max_keepalive_connections=10,
                        keepalive_expiry=30,
                    ),
                    http2=True,
                )
    return _http_client


# ── Searx client ──────────────────────────────────────────────────────────────
class _CFSearxWrapper(SearxSearchWrapper):
    async def _asearx_api_query(self, params: dict) -> SearxResults:
        cf_headers = {
            "CF-Access-Client-Id": CF_ACCESS_CLIENT_ID,
            "CF-Access-Client-Secret": CF_ACCESS_CLIENT_SECRET,
        }
        async with aiohttp.ClientSession(headers=cf_headers) as session:
            async with session.get(self.searx_host, params=params) as resp:
                resp.raise_for_status()
                return SearxResults(await resp.text())


_searx_client: _CFSearxWrapper | None = None
_searx_lock = asyncio.Lock()


async def get_searx_client() -> _CFSearxWrapper:
    global _searx_client
    async with _searx_lock:
        if _searx_client is None:
            _searx_client = _CFSearxWrapper(
                searx_host=SEARX_HOST,
                engines=DEFAULT_ENGINES,
                categories=DEFAULT_CATEGORIES,
                k=DEFAULT_MAX_RESULTS,
                params={"language": "auto", "format": "json"},
            )
    return _searx_client


async def asearch_and_extract(search_query: str) -> list[dict]:
    t0 = time.perf_counter()
    client = await get_searx_client()
    try:
        raw: list[dict] = await client.aresults(
            query=search_query,
            num_results=DEFAULT_MAX_RESULTS,
            engines=DEFAULT_ENGINES,
            categories=",".join(DEFAULT_CATEGORIES),
        )
    except Exception:
        logger.exception("[search] SearxNG failed for query: %r", search_query)
        return []

    if not raw or not isinstance(raw[0], dict) or "link" not in raw[0]:
        logger.warning("[search] No usable results for: %r", search_query)
        return []

    seen: set[str] = set()
    results = []
    for r in raw:
        url = r.get("link", "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        results.append({
            "url": url,
            "title": r.get("title", "").strip(),
            "snippet": r.get("snippet", "").strip(),
        })

    logger.info("[search] %.2fs → %d results for %r", time.perf_counter() - t0, len(results), search_query)
    return results


# ── FAISS Store ───────────────────────────────────────────────────────────────
class FaissStore:
    def __init__(self):
        self._lock = asyncio.Lock()
        if Path(FAISS_INDEX_PATH).exists() and Path(FAISS_META_PATH).exists():
            self.index: faiss.IndexFlatIP = faiss.read_index(FAISS_INDEX_PATH)
            with open(FAISS_META_PATH, "rb") as f:
                saved = pickle.load(f)
            self.metadata: list[dict] = saved["metadata"]
            self.url_index: dict[str, list[int]] = saved["url_index"]
            logger.info("[faiss] Loaded — %d vectors", self.index.ntotal)
        else:
            self.index = faiss.IndexFlatIP(EMBED_DIM)
            self.metadata = []
            self.url_index = {}

    def _save_sync(self):
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(FAISS_META_PATH, "wb") as f:
            pickle.dump({"metadata": self.metadata, "url_index": self.url_index}, f)

    async def save_async(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_cpu_executor, self._save_sync)

    def is_url_fresh(self, url: str) -> bool:
        ids = self.url_index.get(url)
        if not ids:
            return False
        return (time.time() - self.metadata[ids[0]].get("timestamp", 0)) < FAISS_TTL_SECONDS

    async def add_chunks_batch(
            self,
            url_to_chunks: dict[str, list[str]],
            url_to_embs: dict[str, list[list[float]]],
    ):
        t0 = time.perf_counter()
        async with self._lock:
            now = time.time()
            for url, chunks in url_to_chunks.items():
                embs = url_to_embs[url]
                vectors = np.array(embs, dtype="float32")
                faiss.normalize_L2(vectors)
                start_id = len(self.metadata)
                self.index.add(vectors)
                ids = []
                for i, chunk in enumerate(chunks):
                    cid = start_id + i
                    self.metadata.append({"id": cid, "url": url, "text": chunk, "timestamp": now})
                    ids.append(cid)
                self.url_index[url] = ids
                logger.info("[faiss] +%d chunks ← %s", len(ids), url)

            await self.save_async()
        logger.info("[faiss] add_chunks_batch done in %.2fs, total=%d", time.perf_counter() - t0, self.index.ntotal)

    def search(self, query_emb: list[float], top_k: int, url_filter: set[str] | None = None) -> list[dict]:
        if self.index.ntotal == 0:
            return []

        k = min(top_k * (4 if url_filter else 1), self.index.ntotal)
        q = np.array([query_emb], dtype="float32")
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx]
            if url_filter and meta["url"] not in url_filter:
                continue
            results.append({**meta, "score": float(score)})
            if len(results) >= top_k:
                break
        return results


_faiss_store: FaissStore | None = None
_faiss_init_lock = asyncio.Lock()


async def get_faiss_store() -> FaissStore:
    global _faiss_store
    async with _faiss_init_lock:
        if _faiss_store is None:
            _faiss_store = FaissStore()
    return _faiss_store


# ── Crawler — httpx only, no fallback, no semaphore ──────────────────────────
async def _crawl_single(url: str) -> tuple[str, str | None]:
    t0 = time.perf_counter()
    client = await get_http_client()

    try:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        logger.warning("[crawl] fetch failed %s — skip (%s)", url, e)
        return url, None

    logger.info("[crawl] fetch %.2fs — %s", time.perf_counter() - t0, url)

    loop = asyncio.get_running_loop()
    t1 = time.perf_counter()
    try:
        text: str | None = await loop.run_in_executor(
            _cpu_executor,
            lambda: trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                output_format="markdown",
                favor_recall=True,
            ),
        )
    except Exception:
        text = None

    if not text:
        logger.warning("[crawl] extract failed — skip %s", url)
        return url, None

    logger.info("[crawl] extract %.2fs (%d chars) — %s", time.perf_counter() - t1, len(text), url)
    return url, text


async def crawl_urls_parallel(urls: list[str]) -> dict[str, str]:
    t0 = time.perf_counter()
    pairs = await asyncio.gather(*[_crawl_single(u) for u in urls])
    result = {url: text for url, text in pairs if text}
    logger.info("[crawl] parallel done %.2fs — %d/%d ok", time.perf_counter() - t0, len(result), len(urls))
    return result


# ── Chunker ───────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP, min_len: int = 40) -> list[str]:
    import re
    sentences = re.split(r'(?<=[.!?\n])\s+', text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) <= size:
            current += (" " if current else "") + sentence
        else:
            if len(current) >= min_len:
                chunks.append(current.strip())
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + " " + sentence
    if len(current) >= min_len:
        chunks.append(current.strip())
    return chunks


# ── Ingest ────────────────────────────────────────────────────────────────────
async def ingest_urls_batch(url_text_map: dict[str, str]):
    if not url_text_map:
        return

    t0 = time.perf_counter()
    embedder = get_openai_embedding_client()
    store = await get_faiss_store()

    all_chunks: list[str] = []
    chunk_url_map: list[str] = []

    for url, text in url_text_map.items():
        for chunk in chunk_text(text):
            all_chunks.append(chunk)
            chunk_url_map.append(url)

    if not all_chunks:
        logger.warning("[ingest] No chunks to embed")
        return

    logger.info("[ingest] Embedding %d chunks...", len(all_chunks))
    t_emb = time.perf_counter()
    embeddings = await embedder.aembed_documents(all_chunks)
    logger.info("[ingest] Embed done %.2fs for %d chunks", time.perf_counter() - t_emb, len(all_chunks))

    url_to_chunks: dict[str, list[str]] = {}
    url_to_embs: dict[str, list[list[float]]] = {}
    for chunk, emb, url in zip(all_chunks, embeddings, chunk_url_map):
        url_to_chunks.setdefault(url, []).append(chunk)
        url_to_embs.setdefault(url, []).append(emb)

    await store.add_chunks_batch(url_to_chunks, url_to_embs)
    logger.info("[ingest] Total ingest %.2fs", time.perf_counter() - t0)


# ── Retriever ─────────────────────────────────────────────────────────────────
async def retrieve_chunks(
        urls: list[str],
        query_emb: list[float],
        top_k: int,
        url_snippets: dict[str, str] | None = None,
) -> list[dict]:
    store = await get_faiss_store()
    to_crawl = [url for url in urls if not store.is_url_fresh(url)]

    logger.info("[retrieve] cache_hit=%d miss=%d", len(urls) - len(to_crawl), len(to_crawl))

    if to_crawl:
        crawled = await crawl_urls_parallel(to_crawl)
        # snippet fallback for completely failed crawls
        if url_snippets:
            for url in to_crawl:
                if url not in crawled and url_snippets.get(url):
                    crawled[url] = url_snippets[url]
                    logger.info("[retrieve] snippet fallback: %s", url)
        await ingest_urls_batch(crawled)

    t_search = time.perf_counter()
    all_chunks = store.search(query_emb, top_k=top_k, url_filter=set(urls))
    logger.info("[retrieve] faiss.search %.3fs → %d raw chunks", time.perf_counter() - t_search, len(all_chunks))

    return all_chunks


# ── Main pipeline ─────────────────────────────────────────────────────────────
async def web_rag_answer(llm, embedder, user_input: str, detected_lang: str, top_k: int = 2) -> dict[str, Any]:
    t_total = time.perf_counter()

    # Step 1: search + embed query in parallel
    logger.info("[pipeline] START — query=%r", user_input[:80])
    raw_results, query_emb = await asyncio.gather(
        asearch_and_extract(user_input),
        embedder.aembed_query(user_input),
    )
    if not raw_results:
        return {"answer": NO_RESULT_RESPONSE_MAP.get(detected_lang), "sources": []}

    # Step 2: take top URLs directly (search engine already ranks them)
    top_results = raw_results[:TOP_URLS]
    top_urls = [r["url"] for r in top_results]
    logger.info("[pipeline] top_urls=%s", top_urls)

    # Step 3: retrieve chunks
    url_snippets = {r["url"]: r.get("snippet", "") for r in top_results}
    chunks = await retrieve_chunks(top_urls, query_emb, top_k=top_k, url_snippets=url_snippets)

    if not chunks:
        return {"answer": NO_RESULT_RESPONSE_MAP.get(detected_lang), "sources": []}

    # Step 4: build prompt
    web_context = "\n\n---\n\n".join(
        f"[{i + 1}] {c['url']}\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    prompt = _WEB_ANSWER_PROMPT.format(
        detected_language=detected_lang,
        web_context=web_context,
        user_input=user_input,
    )

    # Step 5: LLM
    t_llm = time.perf_counter()
    answer = await llm.ainvoke(prompt)
    logger.info("[pipeline] llm %.2fs", time.perf_counter() - t_llm)
    logger.info("[pipeline] TOTAL %.2fs", time.perf_counter() - t_total)

    return {
        "answer": answer,
        "sources": [
            {
                "url": r["url"],
                "name": r.get("title", ""),
                "page_number": 1,
                "total_pages": 1,
            }
            for r in top_results
        ],
    }
