"""
fill_missing_explanation.py
───────────────────────────
Scans quiz_LOMA281.json for objects missing the "explanation" field,
calls Azure OpenAI to generate it, and writes the result back to the file.
"""

import json
import re
import time
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from src.rag.llm.chat_llm import get_openai_chat_client

# ─── CONFIG ───────────────────────────────────────────────────────────────────
FILE_PATH = r"/data/quiz_LOMA281_backup.json"

EXPLANATION_SYSTEM_PROMPT = """\
You are a senior LOMA certification exam designer and mentor with 15+ years of experience.
Your tone is: authoritative yet supportive — precise in technical content, encouraging in explanations.
Your audience is adult learners preparing for LOMA 280/281/290 exams.

─────────────────────────────────────────────────────────
TASK
─────────────────────────────────────────────────────────
You will receive a multiple-choice question object (JSON) that is missing its "explanation" field.
Generate ONLY the "explanation" value — a single JSON-encoded string.

─────────────────────────────────────────────────────────
EXPLANATION WRITING RULES
─────────────────────────────────────────────────────────
Write as a supportive mentor speaking directly to the learner.
Tone: friendly, empathetic, encouraging — never condescending.
Use Markdown inside the string (\\n for newlines, **bold** for emphasis).

Structure — exactly THREE blocks in this order:

  ── BLOCK 1 ── ✅ Confirm & explain the correct answer  (2–3 sentences)
     State which option is correct and WHY, linking to the definition or rule from correct_reason.
     For calculations: show every arithmetic step explicitly.

  ── BLOCK 2 ── ❌ Analyse each wrong option  (1 sentence per wrong option)
     For EACH incorrect option, name the other concept it actually describes
     and explain in one sentence why it is wrong for this question.

  ── BLOCK 3 ── 💡 Key takeaway  (1–2 sentences)
     One memorable rule or pattern the learner can carry forward.

Emoji: use ONLY block markers (✅ ❌ 💡) plus ONE optional extra (💪 📌 🔑) — max 4 total.
Length: 80–150 words total.

JSON ENCODING RULES:
  • Use \\n for newlines — never literal line breaks inside string values.
  • **bold** is valid — do NOT escape asterisks.
  • No markdown code fences in output.

─────────────────────────────────────────────────────────
OUTPUT FORMAT
─────────────────────────────────────────────────────────
Return ONLY a valid JSON object with a single key:

{"explanation": "<your explanation string here>"}

Nothing else. No preamble. No trailing text.
"""


# ─── PROMPT BUILDER ───────────────────────────────────────────────────────────
def build_user_prompt(question_obj: dict) -> str:
    obj_copy = {k: v for k, v in question_obj.items() if k != "explanation"}
    return (
        "Generate the explanation field for the following quiz question.\n\n"
        "QUESTION OBJECT:\n"
        f"{json.dumps(obj_copy, ensure_ascii=False, indent=2)}\n\n"
        "Return only: {\"explanation\": \"<your explanation>\"}"
    )


# ─── RESPONSE PARSER ──────────────────────────────────────────────────────────
def parse_explanation(raw: str) -> str:
    raw = raw.strip()

    # Remove markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
        return parsed["explanation"]
    except (json.JSONDecodeError, KeyError):
        match = re.search(r'"explanation"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        if match:
            return match.group(1)
        raise ValueError(f"Cannot parse explanation from LLM response:\n{raw}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main() -> None:
    path = Path(FILE_PATH)

    if not path.exists():
        print(f"❌ File not found: {FILE_PATH}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data: list = json.load(f)

    if not isinstance(data, list):
        print("❌ JSON root must be a list.")
        return

    missing_indices = [
        i for i, obj in enumerate(data)
        if isinstance(obj, dict) and not obj.get("explanation", "").strip()
    ]

    total = len(data)
    n_missing = len(missing_indices)

    print(f"📂 File : {FILE_PATH}")
    print(f"📊 Total objects   : {total}")
    print(f"🔍 Missing explanation: {n_missing}")

    if n_missing == 0:
        print("✅ Nothing to do.")
        return

    client = get_openai_chat_client()

    success_count = 0
    fail_count = 0

    for seq, idx in enumerate(missing_indices, start=1):
        obj = data[idx]
        q_preview = obj.get("question", "(no question)")[:70]
        print(f"\n[{seq}/{n_missing}] Index {idx}: \"{q_preview}...\"")

        try:
            response = client.chat(
                messages=[
                    SystemMessage(content=EXPLANATION_SYSTEM_PROMPT),
                    HumanMessage(content=build_user_prompt(obj)),
                ]
            )

            explanation = parse_explanation(response)

            # Insert explanation after correct_reason
            new_obj = {}
            inserted = False

            for key in obj.keys():
                new_obj[key] = obj[key]
                if key == "correct_reason" and not inserted:
                    new_obj["explanation"] = explanation
                    inserted = True

            if not inserted:
                new_obj["explanation"] = explanation

            data[idx] = new_obj

            success_count += 1
            print(f"   ✅ Done ({len(explanation)} chars)")

        except Exception as exc:
            fail_count += 1
            print(f"   ❌ Failed: {exc}")

        # Save after each item (safe mode)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        time.sleep(0.5)  # avoid rate limit

    print("\n" + "─" * 50)
    print(f"✅ Generated : {success_count}")
    print(f"❌ Failed    : {fail_count}")
    print(f"💾 Saved to  : {FILE_PATH}")


if __name__ == "__main__":
    main()
