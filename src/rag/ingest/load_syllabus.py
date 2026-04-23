"""
Parse LOMA Syllabus Excel files into a normalized JSON blob.

Input:  data/LOMA*_Syllabus_*.xlsx (discovered via SYLLABUS_FILES)
Output: data/ingest_overall/syllabus.json (keyed by "LOMA NNN")

Chạy: python -m src.rag.ingest.load_syllabus
"""
import json
import re
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
