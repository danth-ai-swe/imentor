from typing import Any, Dict, List, Optional, Tuple

from tavily import AsyncTavilyClient

from src.rag.llm.chat_llm import AzureChatClient
from src.utils.logger_utils import logger

TAVILY_API_KEY = "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"
_tavily_client: Optional[AsyncTavilyClient] = None


def get_tavily_client() -> AsyncTavilyClient:
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
    return _tavily_client


_WEB_ANSWER_PROMPT = """You are a helpful assistant answering a user question using web search results.

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


# ── Step 2: Search + extract via Tavily ───────────────────────────────────────

async def asearch_and_extract(
        search_query: str,
        *,
        max_results: int = 2,
        relevance_threshold: float = 0.8,
        chunks_per_source: int = 2
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run Tavily search then extract full content from high-relevance URLs.

    Returns:
        extracted_results : list of dicts from tavily.extract (per source)
        search_meta       : list of {url, title} preserved from the search results,
                            used later to build structured source items
    """
    client = get_tavily_client()

    # --- Search ---
    try:
        search_response = await client.search(
            query=search_query,
            search_depth="basic",
            max_results=max_results,
        )
    except Exception:
        logger.exception("Tavily search failed")
        return [], []

    results: List[Dict[str, Any]] = search_response.get("results", [])

    # Filter by relevance threshold
    relevant = [r for r in results if r.get("score", 0) > relevance_threshold and r.get("url")]

    # Fallback: if nothing passes threshold, take top-3 by score
    if not relevant:
        relevant = sorted(results, key=lambda r: r.get("score", 0), reverse=True)[:2]
        relevant = [r for r in relevant if r.get("url")]

    if not relevant:
        logger.warning("Tavily: no URLs to extract from")
        return [], []

    relevant_urls = [r["url"] for r in relevant]

    # Preserve title + url for source item construction
    search_meta: List[Dict[str, Any]] = [
        {"url": r["url"], "title": r.get("title", "")}
        for r in relevant
    ]

    # --- Extract ---
    try:
        extracted = await client.extract(
            urls=relevant_urls,
            query=search_query,
            chunks_per_source=chunks_per_source,
            extract_depth="basic",
        )
        extracted_results: List[Dict[str, Any]] = extracted.get("results", [])
    except Exception:
        logger.exception("Tavily extract failed — falling back to raw search snippets")
        extracted_results = [
            {"url": r["url"], "raw_content": r.get("content", "")}
            for r in relevant
        ]

    return extracted_results, search_meta


# ── Step 3: Build context string ──────────────────────────────────────────────

def _build_web_context(extracted_results: List[Dict[str, Any]]) -> str:
    """Flatten extracted chunks into a single context string with source labels."""
    blocks: List[str] = []
    for item in extracted_results:
        url = item.get("url", "unknown")
        content = item.get("content") or item.get("raw_content", "")

        if isinstance(content, list):
            text = "\n".join(
                chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                for chunk in content
            )
        else:
            text = str(content)

        if text.strip():
            blocks.append(f"[Source: {url}]\n{text.strip()}")

    return "\n\n---\n\n".join(blocks)


# ── Step 4: Build structured source items ────────────────────────────────────

def build_source_items(search_meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert Tavily search metadata into structured source items.

    Output shape per item:
    {
        "name"       : str,   # site hostname, e.g. "wikipedia.org"
        "title"      : str,   # article/page title from Tavily
        "url"        : str,   # full URL
        "page"       : 1,
        "page_total" : 1,
    }
    """
    items: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    for meta in search_meta:
        url = meta.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        title = meta.get("title", "") or ""
        items.append({
            "name": title,
            "url": url,
            "page_number": 1,
            "total_pages": 1,
        })

    return items


# ── Step 5: Generate answer ───────────────────────────────────────────────────

async def agenerate_web_answer(
        llm: AzureChatClient,
        user_input: str,
        detected_language: str,
        extracted_results: List[Dict[str, Any]],
        search_meta: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generate a final answer from web-extracted content.

    Returns:
        answer       : LLM-generated answer string (no inline source mentions)
        source_items : structured source list for the API output field
    """
    web_context = _build_web_context(extracted_results)

    if not web_context.strip():
        return "", []

    prompt = _WEB_ANSWER_PROMPT.format(
        detected_language=detected_language,
        web_context=web_context,
        user_input=user_input,
    )

    try:
        answer = (await llm.ainvoke(prompt)).strip()
    except Exception:
        logger.exception("agenerate_web_answer: LLM call failed")
        answer = ""

    source_items = build_source_items(search_meta)
    return answer, source_items
