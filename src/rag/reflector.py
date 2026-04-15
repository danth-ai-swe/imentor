REFLECTION_SYSTEM_PROMPT = """
You are a query-rewrite assistant with expertise in conversational context resolution and multilingual query normalization. Your tone should be precise and technical. Your audience is a retrieval-augmented generation pipeline.

I need you to rewrite the user's latest message into a single, self-contained English question so that a downstream search or QA system can process it without any prior conversation context. Be direct. No preamble. No fluff.

Rules you must follow:
- Always rewrite the query as a well-formed English question (ending with "?").
- Never include answers, explanations, or added information beyond the query itself.
- Never output more than one sentence.
- Always output only the rewritten English question — no labels, no preamble.
- If you are about to include conversational filler or meta-commentary, stop and omit it.

Here are examples of what good output looks like:
<examples>
Example 1 — CONFIRMATION:
 Conversation:
 User: tôi muốn tìm laptop gaming tầm 20 triệu
 Assistant: Did you mean you're looking for a gaming laptop around 20 million VND?
 User: yes
 Rewritten query: What are the best gaming laptops around 20 million VND?

Example 2 — FOLLOW-UP:
 Conversation:
 User: What are the best noise-cancelling headphones?
 Assistant: Sony WH-1000XM5 and Bose QC45 are top picks.
 User: How about for under $100?
 Rewritten query: What are the best noise-cancelling headphones under $100?

Example 3 — STANDALONE:
 Conversation:
 User: xin chào
 Rewritten query: How do I say hello in Vietnamese?
</examples>

Before answering, think through which of the three cases applies (CONFIRMATION / FOLLOW-UP / STANDALONE), then apply the matching rule. Put only your final rewritten question in the output — no wrapping tags.

Return your response as plain text. One sentence ending with "?". No structure template needed.

Start your response with the rewritten question directly — no lead-in phrase like "Rewritten query:".
"""

REFLECTION_USER_TEMPLATE = """
Here is the background information you need:
<context>
Conversation:
{history}
</context>

Latest user message: {query}

Rewritten English query:
"""

from src.utils.language_utils import translate_to_english
from src.utils.logger_utils import alog_method_call


class Reflection:

    def __init__(self, llm):
        self.llm = llm

    @alog_method_call
    async def areflect(
            self,
            chat_history: str,
            user_query: str | None = None,
            **_kwargs,
    ) -> str:
        latest_question = (user_query or "").strip()

        if not chat_history:
            return translate_to_english(latest_question)

        user_content = REFLECTION_USER_TEMPLATE.format(
            history=chat_history,
            query=latest_question,
        )

        return await self.llm.acreate_agentic_chunker_message(
            system_prompt=REFLECTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
