"""
Script: update_quiz_names.py
Purpose: Update course, module, and lesson fields in quiz_LOMA281.json
         from numeric values to full descriptive names.
"""

import json
import os

# ── Mapping tables ──────────────────────────────────────────────────────────

COURSE_NAMES = {
    "LOMA281": "LOMA 281 - Meeting Customer Needs with Insurance and Annuities - 3rd Edition",
    "LOMA291": "LOMA 291 - Improving the Bottom Line - Insurance Company Operations - 2nd Edition",
}

# Format: (course, module_number) -> full module name
MODULE_NAMES = {
    ("LOMA281", "1"): "Module 1 - Risk and Insurance",
    ("LOMA281", "2"): "Module 2 - Individual Insurance Products",
    ("LOMA281", "3"): "Module 3 - Benefits, Provisions, and Ownership Rights",
    ("LOMA281", "4"): "Module 4 - Group Products",

    ("LOMA291", "1"): "Module 1 - Managing the Company to Meet Stakeholder Needs",
    ("LOMA291", "2"): "Module 2 - Serving the Customer throughout the Policy Lifecycle",
    ("LOMA291", "3"): "Module 3 - Key Support Functions for Insurer Success",
    ("LOMA291", "4"): "Module 4 - Functions and Goals of Financial Management",
}

# Format: (course, module_number, lesson_number) -> full lesson name
LESSON_NAMES = {
    # LOMA 281 - Module 1
    ("LOMA281", "1", "1"): "Lesson 1 - Risky Business",
    ("LOMA281", "1", "2"): "Lesson 2 - Organization and Regulation of Insurance Companies",
    ("LOMA281", "1", "3"): "Lesson 3 - Life Insurance Policies as Contracts",
    ("LOMA281", "1", "4"): "Lesson 4 - The Value Exchange in the Insurance Transaction",

    # LOMA 281 - Module 2
    ("LOMA281", "2", "1"): "Lesson 1 - Term Life Insurance",
    ("LOMA281", "2", "2"): "Lesson 2 - Cash Value Life Insurance",
    ("LOMA281", "2", "3"): "Lesson 3 - Annuities",
    ("LOMA281", "2", "4"): "Lesson 4 - Health Insurance",

    # LOMA 281 - Module 3
    ("LOMA281", "3", "1"): "Lesson 1 - Supplemental Benefits",
    ("LOMA281", "3", "2"): "Lesson 2 - Life Insurance Policy Provisions",
    ("LOMA281", "3", "3"): "Lesson 3 - Life Insurance Policy Ownership Rights",

    # LOMA 281 - Module 4
    ("LOMA281", "4", "1"): "Lesson 1 - Group Insurance",
    ("LOMA281", "4", "2"): "Lesson 2 - Group Life Insurance",
    ("LOMA281", "4", "3"): "Lesson 3 - Group Retirement Plans",

    # LOMA 291 - Module 1
    ("LOMA291", "1", "1"): "Lesson 1 - Many Stakeholders, Many Demands",
    ("LOMA291", "1", "2"): "Lesson 2 - The Great Organizational Pyramid",
    ("LOMA291", "1", "3"): "Lesson 3 - Risk, Return, and Risk Management",

    # LOMA 291 - Module 2
    ("LOMA291", "2", "1"): "Lesson 1 - Distribution",
    ("LOMA291", "2", "2"): "Lesson 2 - New Business and Underwriting",
    ("LOMA291", "2", "3"): "Lesson 3 - Customer Service",
    ("LOMA291", "2", "4"): "Lesson 4 - Claims Administration",

    # LOMA 291 - Module 3
    ("LOMA291", "3", "1"): "Lesson 1 - Marketing",
    ("LOMA291", "3", "2"): "Lesson 2 - Product Development",
    ("LOMA291", "3", "3"): "Lesson 3 - Legal and Compliance Functions",

    # LOMA 291 - Module 4
    ("LOMA291", "4", "1"): "Lesson 1 - Financial Functions in an Insurance Company",
    ("LOMA291", "4", "2"): "Lesson 2 - Goals for Financial Management",
}

# ── Main logic ───────────────────────────────────────────────────────────────

def normalize_course_key(course_raw: str) -> str:
    """Normalize course string to lookup key, e.g. 'LOMA281' or 'LOMA 281' -> 'LOMA281'."""
    return course_raw.replace(" ", "").upper()


def update_source(source: dict) -> dict:
    """Update a single source dict in-place and return it."""
    course_raw = str(source.get("course", "")).strip()
    module_raw = str(source.get("module", "")).strip()
    lesson_raw = str(source.get("lesson", "")).strip()

    course_key = normalize_course_key(course_raw)

    # --- course ---
    if course_key in COURSE_NAMES:
        source["course"] = COURSE_NAMES[course_key]
    else:
        print(f"  [WARN] Unknown course: '{course_raw}'")

    # --- module ---
    module_key = (course_key, module_raw)
    if module_key in MODULE_NAMES:
        source["module"] = MODULE_NAMES[module_key]
    else:
        print(f"  [WARN] Unknown module: course='{course_raw}' module='{module_raw}'")

    # --- lesson ---
    lesson_key = (course_key, module_raw, lesson_raw)
    if lesson_key in LESSON_NAMES:
        source["lesson"] = LESSON_NAMES[lesson_key]
    else:
        print(f"  [WARN] Unknown lesson: course='{course_raw}' module='{module_raw}' lesson='{lesson_raw}'")

    return source


def process_file(input_path: str, output_path: str | None = None) -> None:
    """Load the JSON file, update all sources, and save."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected the JSON root to be a list of question objects.")

    updated_count = 0
    for i, question in enumerate(data):
        sources = question.get("sources", [])
        for source in sources:
            update_source(source)
            updated_count += 1

    # Write output
    save_path = output_path or input_path
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done! Updated {updated_count} source(s) across {len(data)} question(s).")
    print(f"   Saved to: {save_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INPUT_FILE = r"C:\imt-ai-brain\data\quiz_LOMA291_backup.json"

    # To overwrite the original file, leave OUTPUT_FILE as None.
    # To save a separate copy, set OUTPUT_FILE to a new path, e.g.:
    OUTPUT_FILE = r"C:\imt-ai-brain\data\quiz_LOMA291.json"

    process_file(INPUT_FILE, OUTPUT_FILE)