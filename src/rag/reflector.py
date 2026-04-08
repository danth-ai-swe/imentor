REFLECTION_SYSTEM_PROMPT = """\
You are a query-rewrite assistant. Given a conversation and the user's latest message, produce a single, self-contained English query using these rules:

1. CONFIRMATION — If the assistant's last turn was a clarification/suggestion (e.g. "Did you mean …?") and the user replies with a short affirmative ("yes", "yeah", "correct", "right", "that's it"), extract the **original user topic from the earlier user message** verbatim. Do NOT rephrase it as a question.

2. FOLLOW-UP — If the latest message relies on prior context (pronouns, ellipsis, short answers like "red" or "that one"), merge the relevant context into a complete, self-contained English query.

3. STANDALONE — If the latest message is already self-contained (full question, greeting, or off-topic), translate it to English as-is.

Output ONLY the rewritten English query — one sentence, no preamble, no answer, no added information.\
"""

REFLECTION_USER_TEMPLATE = """\
Conversation:
{history}

Latest user message: {query}

Rewritten English query:\
"""

from src.constants.app_constant import MAX_RECENT_HISTORY_ENTRIES
from src.utils.language_utils import translate_to_english
from src.utils.logger_utils import alog_method_call


class Reflection:

    def __init__(self, llm):
        self.llm = llm

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        lines: list[str] = []
        for entry in history[-MAX_RECENT_HISTORY_ENTRIES:]:
            role = entry.get("role", "user").capitalize()
            if entry.get("parts"):
                text = " ".join(p["text"] for p in entry["parts"])
            else:
                text = entry.get("content", "")
            lines.append(f"{role}: {text}")
        return "\n".join(lines)

    @alog_method_call
    async def areflect(
            self,
            chat_history: list[dict],
            user_query: str | None = None,
            **_kwargs,
    ) -> str:
        latest_question = (user_query or "").strip()

        if not chat_history:
            return translate_to_english(latest_question)

        user_content = REFLECTION_USER_TEMPLATE.format(
            history=self._format_history(chat_history),
            query=latest_question,
        )

        try:
            if hasattr(self.llm, "acreate_agentic_chunker_message"):
                return await self.llm.acreate_agentic_chunker_message(
                    system_prompt=REFLECTION_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                )

            if hasattr(self.llm, "ainvoke"):
                return await self.llm.ainvoke(
                    f"{REFLECTION_SYSTEM_PROMPT}\n\n{user_content}"
                )
        except Exception:
            pass

        return translate_to_english(latest_question)
