"""T2 CLI driver for the LangGraph hybrid agent pipeline.

Modes:
    python scripts/run_agent.py                       # interactive REPL
    python scripts/run_agent.py --compare             # REPL, runs both old + new pipelines side-by-side
    python scripts/run_agent.py --questions FILE      # batch over one-question-per-line file → CSV stdout

Run with PYTHONPATH set to the project root, e.g.:
    PYTHONPATH=. python scripts/run_agent.py
"""

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants.app_constant import COLLECTION_NAME, OVERALL_COLLECTION_NAME
from src.rag.db_vector import get_qdrant_client
from src.rag.llm.chat_llm import get_openai_chat_client
from src.rag.llm.embedding_llm import get_async_client, get_openai_embedding_client, get_sync_client
from src.rag.search.agent import run_agent_pipeline
from src.rag.search.pipeline import async_pipeline_dispatch
from src.rag.search.reranker import _get_reranker


async def warm_up():
    """Mirror main.py lifespan warm-up (minus FastAPI router pieces)."""
    # Warm both Qdrant clients so the first overall-collection search doesn't
    # pay TCP-connection cost. The ColBERT model is module-level singleton
    # so loading it once is enough for both clients.
    core_qdrant = get_qdrant_client(COLLECTION_NAME)
    try:
        core_qdrant.client.get_collections()
    except Exception:
        pass
    get_qdrant_client(OVERALL_COLLECTION_NAME)
    get_sync_client()
    get_async_client()
    get_openai_chat_client()
    get_openai_embedding_client()
    # Warm reranker in a thread so first agent call doesn't pay model-load cost.
    await asyncio.get_running_loop().run_in_executor(None, _get_reranker)


def print_result(label: str, result: dict, latency: float):
    print(f"\n--- {label} ({latency:.2f}s) ---")
    print(f"  intent           : {result.get('intent')}")
    print(f"  detected_language: {result.get('detected_language')}")
    print(f"  answer_satisfied : {result.get('answer_satisfied')}")
    print(f"  web_search_used  : {result.get('web_search_used')}")
    print(f"  sources          : {len(result.get('sources') or [])}")
    response = result.get("response") or ""
    print(f"  response ({len(response)} chars):")
    print("  " + (response.replace("\n", "\n  ") if response else "(none)"))


async def run_one_agent(question: str) -> tuple[dict, float]:
    t0 = time.perf_counter()
    result = await run_agent_pipeline(question)
    return result, time.perf_counter() - t0


async def run_one_old(question: str) -> tuple[dict, float]:
    t0 = time.perf_counter()
    result = await async_pipeline_dispatch(question)
    return dict(result), time.perf_counter() - t0


async def repl_loop(compare: bool):
    print("Insuripedia agent REPL. Type a question (Ctrl-C / Ctrl-D to exit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return
        if not question:
            continue

        if compare:
            (agent_res, agent_t), (old_res, old_t) = await asyncio.gather(
                run_one_agent(question), run_one_old(question),
            )
            print_result("AGENT", agent_res, agent_t)
            print_result("OLD  ", old_res, old_t)
        else:
            agent_res, agent_t = await run_one_agent(question)
            print_result("AGENT", agent_res, agent_t)


async def run_batch(path: str):
    """Read one question per line, run both pipelines, emit CSV to stdout."""
    questions = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not questions:
        print("No questions found.", file=sys.stderr)
        return

    writer = csv.writer(sys.stdout)
    writer.writerow([
        "question",
        "agent_intent", "agent_satisfied", "agent_web", "agent_sources", "agent_resp_len", "agent_latency_s",
        "old_intent",   "old_satisfied",   "old_web",   "old_sources",   "old_resp_len",   "old_latency_s",
    ])

    for q in questions:
        (agent_res, agent_t), (old_res, old_t) = await asyncio.gather(
            run_one_agent(q), run_one_old(q),
        )
        writer.writerow([
            q,
            agent_res.get("intent"),
            agent_res.get("answer_satisfied"),
            agent_res.get("web_search_used"),
            len(agent_res.get("sources") or []),
            len(agent_res.get("response") or ""),
            f"{agent_t:.2f}",
            old_res.get("intent"),
            old_res.get("answer_satisfied"),
            old_res.get("web_search_used"),
            len(old_res.get("sources") or []),
            len(old_res.get("response") or ""),
            f"{old_t:.2f}",
        ])
        sys.stdout.flush()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Also run pipeline.py for side-by-side comparison.")
    parser.add_argument("--questions", type=str, default=None, help="Path to a file with one question per line; emit CSV summary.")
    args = parser.parse_args()

    await warm_up()

    if args.questions:
        await run_batch(args.questions)
        return

    await repl_loop(compare=args.compare)


if __name__ == "__main__":
    asyncio.run(main())
