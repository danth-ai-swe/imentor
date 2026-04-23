"""
Parse LOMA Syllabus Excel files into a normalized JSON blob.

Input:  data/LOMA*_Syllabus_*.xlsx (discovered via SYLLABUS_FILES)
Output: data/ingest_overall/syllabus.json (keyed by "LOMA NNN")

Chạy: python -m src.rag.ingest.load_syllabus
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.constants.app_constant import SYLLABUS_FILES, SYLLABUS_JSON

_MODULE_RE = re.compile(r"Module\s+(\d+)[:\-]\s*(.*)", re.DOTALL)
_LESSON_RE = re.compile(r"Lesson\s+(\d+)[:\-]\s*(.*)", re.DOTALL)


def _course_code_from_filename(path: Path) -> str:
    """'LOMA281_Syllabus_1.0.xlsx' -> 'LOMA 281'."""
    m = re.match(r"(LOMA)(\d+)_", path.name)
    if not m:
        raise ValueError(f"Cannot parse course code from filename: {path.name}")
    return f"{m.group(1)} {m.group(2)}"


def _clean_cell(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _to_float(val: Any) -> float:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _split_multiline(val: Any) -> list[str]:
    """Split a multi-line cell on \n, strip, drop empties."""
    text = _clean_cell(val)
    return [line.strip() for line in text.split("\n") if line.strip()]


def _parse_cover(xlsx_path: Path) -> dict[str, str]:
    """Extract Document Code / Version / Effective Date from the Cover sheet."""
    df = pd.read_excel(xlsx_path, sheet_name="Cover", header=None)
    out = {"document_code": "", "version": "", "effective_date": ""}
    for _, row in df.iterrows():
        label = _clean_cell(row.iloc[5]) if len(row) > 5 else ""
        value = row.iloc[8] if len(row) > 8 else None
        if label == "Document Code":
            out["document_code"] = _clean_cell(value)
        elif label == "Version":
            out["version"] = _clean_cell(value)
        elif label == "Effective Date":
            if isinstance(value, (pd.Timestamp, datetime)):
                out["effective_date"] = value.strftime("%Y-%m-%d")
            else:
                out["effective_date"] = _clean_cell(value)
    return out


def _parse_syllabus_sheet(xlsx_path: Path) -> dict[str, Any]:
    """
    Extract Topic Name/Code, Training Method, Type of Learners, Prerequisites,
    Course Objectives, Course Outcomes, Training Materials, Assessment Scheme,
    Passing Criteria from the Syllabus sheet.

    Sheet layout (header-less): col B=item num, col C=label, col D=main value,
    col E=sub label, col F=sub value (only used for Expected Achievement).
    """
    df = pd.read_excel(xlsx_path, sheet_name="Syllabus", header=None)
    out: dict[str, Any] = {
        "topic_name": "",
        "topic_code": "",
        "training_method": "",
        "type_of_learners": "",
        "prerequisites": "",
        "objectives": [],
        "outcomes": [],
        "training_materials": [],
        "assessment_scheme": [],
        "passing_criteria": "",
    }
    label_to_key_simple = {
        "Topic Name *": "topic_name",
        "Topic Code": "topic_code",
        "Training Method(s) *": "training_method",
        "Type of Learners *": "type_of_learners",
        "Prerequisites of knowledge *": "prerequisites",
    }
    label_to_key_list = {
        "Course Objectives *": "objectives",
        "Course Outcomes *": "outcomes",
    }
    in_assessment_block = False
    for _, row in df.iterrows():
        label_c = _clean_cell(row.iloc[2]) if len(row) > 2 else ""
        value_d = row.iloc[3] if len(row) > 3 else None
        value_e = row.iloc[4] if len(row) > 4 else None

        if label_c in label_to_key_simple:
            out[label_to_key_simple[label_c]] = _clean_cell(value_d)
            in_assessment_block = False
            continue
        if label_c in label_to_key_list:
            out[label_to_key_list[label_c]] = _split_multiline(value_d)
            in_assessment_block = False
            continue
        if label_c == "Assessment Scheme *":
            in_assessment_block = True
            scheme = _clean_cell(value_d)
            if scheme:
                out["assessment_scheme"].append(scheme)
            continue
        if in_assessment_block:
            sub = _clean_cell(value_d)
            if sub == "":
                if _clean_cell(label_c) != "":
                    in_assessment_block = False
                continue
            if sub == "Passing criteria":
                out["passing_criteria"] = _clean_cell(value_e)
            elif sub in ("Quiz", "Assignments"):
                if sub not in out["assessment_scheme"]:
                    out["assessment_scheme"].append(sub)
    return out


def _parse_module_header(raw: str) -> tuple[int, str, str]:
    """
    'Module 1: Risk and Insurance\\nIn this module...' ->
        (1, 'Risk and Insurance', 'In this module...')
    Title is taken from the first line; description is the rest (joined).
    Returns (0, "", "") if input is not a module header.
    """
    m = _MODULE_RE.match(raw)
    if not m:
        return 0, "", ""
    mod_num = int(m.group(1))
    tail = m.group(2)
    parts = tail.split("\n", 1)
    title = parts[0].strip()
    description = parts[1].strip() if len(parts) == 2 else ""
    return mod_num, title, description


def _parse_lesson_header(raw: str) -> tuple[int, str]:
    """'Lesson 1: Risky Business?' -> (1, 'Risky Business?'). (0, '') on no match."""
    m = _LESSON_RE.match(raw)
    if not m:
        return 0, ""
    return int(m.group(1)), m.group(2).strip()


def _parse_schedule(xlsx_path: Path) -> tuple[dict[str, Any], dict[str, float]]:
    """
    Return (modules, totals).

    modules: dict keyed by module number as str ("1", "2", ...) ->
        {module_num, title, description, lessons: {lesson_num_str -> lesson_dict}}
    totals: {self_learning_hours, quiz_hours, review_hours, total_hours}

    Sheet layout (header at row 1): col B=Module, col C=Lesson,
    col D=Section/Objectives, col E=Delivery Mode, col F=Duration,
    col I=Directory, col J=Remark.
    """
    df = pd.read_excel(xlsx_path, sheet_name="Schedule", header=None)

    modules: dict[str, Any] = {}
    totals = {
        "self_learning_hours": 0.0,
        "quiz_hours": 0.0,
        "review_hours": 0.0,
        "total_hours": 0.0,
    }
    totals_labels = {
        "Document self-learning": "self_learning_hours",
        "Quiz": "quiz_hours",
        "Review": "review_hours",
        "Total hours": "total_hours",
    }

    current_mod_num: int = 0
    current_lesson: dict[str, Any] | None = None

    for idx in range(2, len(df)):
        row = df.iloc[idx]
        module_cell = _clean_cell(row.iloc[1]) if len(row) > 1 else ""
        lesson_cell = _clean_cell(row.iloc[2]) if len(row) > 2 else ""
        section_cell = _clean_cell(row.iloc[3]) if len(row) > 3 else ""
        delivery_cell = _clean_cell(row.iloc[4]) if len(row) > 4 else ""
        duration_val = row.iloc[5] if len(row) > 5 else None
        directory_cell = _clean_cell(row.iloc[8]) if len(row) > 8 else ""
        remark_cell = _clean_cell(row.iloc[9]) if len(row) > 9 else ""

        if module_cell in totals_labels and lesson_cell:
            totals[totals_labels[module_cell]] = _to_float(lesson_cell)
            continue

        mod_num, mod_title, mod_desc = _parse_module_header(module_cell)
        if mod_num:
            current_mod_num = mod_num
            modules[str(mod_num)] = {
                "module_num": mod_num,
                "title": mod_title,
                "description": mod_desc,
                "lessons": {},
            }

        lesson_num, lesson_title = _parse_lesson_header(lesson_cell)
        if lesson_num and current_mod_num:
            current_lesson = {
                "lesson_num": lesson_num,
                "title": lesson_title,
                "objectives": section_cell,
                "self_learning_hours": _to_float(duration_val),
                "quiz_hours": 0.0,
                "review_hours": 0.0,
                "delivery_mode": delivery_cell,
                "directory": directory_cell,
                "remark": remark_cell,
            }
            modules[str(current_mod_num)]["lessons"][str(lesson_num)] = current_lesson
            continue

        if current_lesson is not None and delivery_cell in ("Quiz", "Review"):
            key = "quiz_hours" if delivery_cell == "Quiz" else "review_hours"
            current_lesson[key] = _to_float(duration_val)

    return modules, totals


def parse_one(xlsx_path: Path) -> tuple[str, dict[str, Any]]:
    """Parse a single Syllabus xlsx into (course_code, course_blob)."""
    course_code = _course_code_from_filename(xlsx_path)
    cover = _parse_cover(xlsx_path)
    syllabus = _parse_syllabus_sheet(xlsx_path)
    modules, totals = _parse_schedule(xlsx_path)
    return course_code, {
        "course_code": course_code,
        "topic_name": syllabus["topic_name"],
        "topic_code": syllabus["topic_code"],
        "version": cover["version"],
        "effective_date": cover["effective_date"],
        "document_code": cover["document_code"],
        "training_method": syllabus["training_method"],
        "type_of_learners": syllabus["type_of_learners"],
        "prerequisites": syllabus["prerequisites"],
        "objectives": syllabus["objectives"],
        "outcomes": syllabus["outcomes"],
        "assessment": {
            "scheme": syllabus["assessment_scheme"],
            "passing_criteria": syllabus["passing_criteria"],
        },
        "totals": totals,
        "modules": modules,
    }


def build_syllabus_blob() -> dict[str, Any]:
    """Parse every file in SYLLABUS_FILES into a {course_code -> blob} dict."""
    if not SYLLABUS_FILES:
        raise FileNotFoundError(
            "No Syllabus xlsx files found. Expected data/LOMA*_Syllabus_*.xlsx"
        )
    blob: dict[str, Any] = {}
    for path in SYLLABUS_FILES:
        code, data = parse_one(path)
        blob[code] = data
    return blob


def main() -> dict[str, Any]:
    blob = build_syllabus_blob()
    SYLLABUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SYLLABUS_JSON, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=2)
    print(f"✅ Wrote {len(blob)} courses → {SYLLABUS_JSON}")
    for code, data in blob.items():
        n_modules = len(data["modules"])
        n_lessons = sum(len(m["lessons"]) for m in data["modules"].values())
        print(f"   {code}: {n_modules} modules, {n_lessons} lessons")
    return blob


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    main()
