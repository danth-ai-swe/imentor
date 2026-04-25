"""LangGraph hybrid agent pipeline — parallel to src.rag.search.pipeline.

The agent replaces the semantic-router + if/else dispatch in pipeline.py
with an LLM tool-calling node, while keeping pre- and post-processing
deterministic. See docs/superpowers/specs/2026-04-25-langgraph-agent-search-design.md.
"""

from src.rag.search.agent.graph import build_agent_graph, get_compiled_graph, run_agent_pipeline

__all__ = ["build_agent_graph", "get_compiled_graph", "run_agent_pipeline"]
