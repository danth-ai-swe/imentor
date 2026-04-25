"""Pure-logic smoke checks for the agent finalize() mapping.

Run: python tests/manual_smoke_agent.py
Exits 0 on success, raises AssertionError on failure.
"""

from src.constants.app_constant import (
    INTENT_CORE_KNOWLEDGE,
    INTENT_OFF_TOPIC,
    INTENT_OVERALL_COURSE_KNOWLEDGE,
    INTENT_QUIZ,
)
from src.rag.search.agent.nodes import finalize_node
from src.rag.search.agent.state import make_initial_state


def case(name, **state_overrides):
    s = make_initial_state("q")
    s.update(state_overrides)
    out = finalize_node(s)
    print(f"  [{name}] intent={out['intent']} satisfied={out['answer_satisfied']} web={out['web_search_used']} resp_len={len(out.get('response') or '')}")
    return out


def main():
    print("finalize_node smoke checks:")

    # 1. Input too long
    out = case("input_too_long",
               early_exit_reason="input_too_long",
               response="too long",
               intent=INTENT_OFF_TOPIC,
               detected_language="English")
    assert out["intent"] == INTENT_OFF_TOPIC
    assert out["answer_satisfied"] is False

    # 2. Unsupported language
    out = case("unsupported_language",
               early_exit_reason="unsupported_language",
               response="lang msg",
               intent=INTENT_OFF_TOPIC,
               detected_language="English")
    assert out["intent"] == INTENT_OFF_TOPIC

    # 3. Quiz
    out = case("quiz",
               early_exit_reason="quiz",
               response=None,
               intent=INTENT_QUIZ,
               detected_language="Vietnamese")
    assert out["intent"] == INTENT_QUIZ
    assert out["response"] is None

    # 4. Pre-processing clarity exit
    out = case("pre_clarity",
               early_exit_reason="clarification",
               response="please rephrase",
               intent=INTENT_CORE_KNOWLEDGE,
               detected_language="Vietnamese")
    assert out["intent"] == INTENT_CORE_KNOWLEDGE
    assert out["response"] == "please rephrase"

    # 5. Tool-driven clarification (off_topic)
    out = case("tool_off_topic",
               clarification={"type": "off_topic", "response": "we only do insurance"},
               detected_language="English")
    assert out["intent"] == INTENT_OFF_TOPIC
    assert "insurance" in out["response"]

    # 6. Tool-driven clarification (vague)
    out = case("tool_vague",
               clarification={"type": "vague", "response": "could you clarify?"},
               detected_language="English")
    assert out["intent"] == INTENT_CORE_KNOWLEDGE

    # 7. Web answer
    out = case("web_done",
               web_answer="web answer text",
               sources=[{"url": "x"}],
               detected_language="English")
    assert out["intent"] == INTENT_CORE_KNOWLEDGE
    assert out["web_search_used"] is True
    assert out["response"] == "web answer text"

    # 8. Core RAG with answer
    out = case("core_ok",
               selected_collection="core",
               response="core answer",
               chunks=[{"id": "c1", "text": "x", "metadata": {}, "score": 0.5}],
               sources=[],
               detected_language="English")
    assert out["intent"] == INTENT_CORE_KNOWLEDGE
    assert out["answer_satisfied"] is True

    # 9. Overall RAG with answer
    out = case("overall_ok",
               selected_collection="overall",
               response="overall answer",
               detected_language="English")
    assert out["intent"] == INTENT_OVERALL_COURSE_KNOWLEDGE
    assert out["answer_satisfied"] is True

    # 10. Overall with no chunks
    out = case("overall_empty",
               selected_collection="overall",
               response=None,
               detected_language="English")
    assert out["intent"] == INTENT_OVERALL_COURSE_KNOWLEDGE
    assert out["answer_satisfied"] is False

    # 11. No collection selected at all (agent gave up)
    out = case("agent_gave_up",
               selected_collection=None,
               detected_language="English")
    assert out["answer_satisfied"] is False

    print("\nAll 11 cases passed.")


if __name__ == "__main__":
    main()
