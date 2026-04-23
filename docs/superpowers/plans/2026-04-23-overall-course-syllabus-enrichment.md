# Overall Course — Syllabus Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrich the existing `course`, `module`, and `lesson` chunks in `data/ingest_overall/overall_course.json` with data parsed from `data/LOMA281_Syllabus_1.0.xlsx` and `data/LOMA291_Syllabus_1.0.xlsx`, keeping `chunk_type` names and deterministic UUID v5 IDs stable.

**Architecture:** Two-step pipeline per spec A2 — new module `src/rag/ingest/load_syllabus.py` parses both Excel files into `data/ingest_overall/syllabus.json`, then `src/rag/ingest/build_overall_chunks.py` reads that intermediate file and enriches existing `course` / `module` / `lesson` chunks (in text body and in payload). Course-to-syllabus matching is done via the `LOMA NNN` substring extracted from the filename and matched against the `course` field in existing chunks (spec B1).

**Tech Stack:** Python 3.10+, pandas + openpyxl (already in `requirements.txt`), uuid, json, pathlib, re. No test framework installed — verification is manual (run the build script and inspect the generated JSON + run the ad-hoc assertion snippets listed in each task).

**Test strategy:** There is no pytest in this repo (spec section 2 confirms "no unit test framework yet"). Every task has a **Verify** step with a one-shot Python snippet that asserts the expected shape/content of the generated artifact. Snippets can be run via `python -c "..."` or saved temporarily as `tmp_verify.py` and deleted before commit. Treat snippet AssertionErrors as test failures — fix and re-run.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/constants/app_constant.py` | Modify (append 3 constants) | Paths for Syllabus glob + intermediate JSON |
| `src/rag/ingest/load_syllabus.py` | **Create** | Parse Excel → normalized `syllabus.json`; 4 helper parsers (Cover, Syllabus, Schedule, one public `main()`) |
| `src/rag/ingest/build_overall_chunks.py` | Modify (edit 3 existing functions + call `load_syllabus.main()` from `main()`) | Enrich existing chunks with syllabus data |
| `data/ingest_overall/syllabus.json` | Generated artifact | Normalized intermediate — input to the enricher |
| `data/ingest_overall/overall_course.json` | Regenerated | Final enriched chunks |

No other files touched. `_build_node_chunks` and `_build_overview_chunk` in `build_overall_chunks.py` are not modified (spec G1 + out-of-scope).

---

## Task 1: Add Syllabus path constants

**Files:**
- Modify: `src/constants/app_constant.py` (append after line 36 where `METADATA_NODE_JSON` is defined)

- [ ] **Step 1: Add constants**

Open `src/constants/app_constant.py`. Find the existing line:
```python
METADATA_NODE_JSON = DATA_DIR / "metadata_node.json"
```

Add immediately after it:
```python
SYLLABUS_FILES = sorted(DATA_DIR.glob("LOMA*_Syllabus_*.xlsx"))
SYLLABUS_JSON = DATA_DIR / "ingest_overall" / "syllabus.json"
```

- [ ] **Step 2: Verify constants resolve**

Run:
```bash
python -c "from src.constants.app_constant import SYLLABUS_FILES, SYLLABUS_JSON; print(SYLLABUS_FILES); print(SYLLABUS_JSON)"
```
Expected output:
```
[WindowsPath('D:/Deverlopment/huudan.com/PythonProject/data/LOMA281_Syllabus_1.0.xlsx'), WindowsPath('D:/Deverlopment/huudan.com/PythonProject/data/LOMA291_Syllabus_1.0.xlsx')]
D:\Deverlopment\huudan.com\PythonProject\data\ingest_overall\syllabus.json
```
If `SYLLABUS_FILES` is empty, the Excel files are missing from `data/` — stop and fix before continuing.

- [ ] **Step 3: Commit**

```bash
git add src/constants/app_constant.py
git commit -m "feat: add SYLLABUS_FILES and SYLLABUS_JSON constants"
```

---

## Task 2: Create `load_syllabus.py` skeleton + Cover sheet parser

**Files:**
- Create: `src/rag/ingest/load_syllabus.py`

- [ ] **Step 1: Create file with imports, helpers, and `_parse_cover`**

Create `src/rag/ingest/load_syllabus.py`:

```python
"""
Parse LOMA Syllabus Excel files into a normalized JSON blob.

Input:  data/LOMA*_Syllabus_*.xlsx (discovered via SYLLABUS_FILES)
Output: data/ingest_overall/syllabus.json (keyed by "LOMA NNN")

Chạy: python -m src.rag.ingest.load_syllabus
"""
import json
import re
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
            if isinstance(value, pd.Timestamp):
                out["effective_date"] = value.strftime("%Y-%m-%d")
            else:
                out["effective_date"] = _clean_cell(value)
    return out
```

- [ ] **Step 2: Verify Cover parser**

Run:
```bash
python -c "
from pathlib import Path
from src.rag.ingest.load_syllabus import _parse_cover, _course_code_from_filename
p = Path('data/LOMA281_Syllabus_1.0.xlsx')
assert _course_code_from_filename(p) == 'LOMA 281'
cover = _parse_cover(p)
print(cover)
assert cover['version'] == '1.0', cover
assert cover['effective_date'] == '2025-06-26', cover
print('OK')
"
```
Expected last line: `OK`. If any assert fails, inspect the printed dict and adjust the column indices in `_parse_cover`.

- [ ] **Step 3: Commit**

```bash
git add src/rag/ingest/load_syllabus.py
git commit -m "feat(load_syllabus): add module skeleton + Cover sheet parser"
```

---

## Task 3: `load_syllabus.py` — Syllabus sheet parser

**Files:**
- Modify: `src/rag/ingest/load_syllabus.py`

- [ ] **Step 1: Append `_parse_syllabus_sheet`**

Append at the end of `src/rag/ingest/load_syllabus.py`:

```python
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
    # Assessment Scheme block: label "Assessment Scheme *" in col C, then
    # sub-labels in col D across following rows; "Passing criteria" value is
    # in col E on the matching sub-row.
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
                # blank row ends assessment block
                if _clean_cell(label_c) != "":
                    in_assessment_block = False
                continue
            if sub == "Passing criteria":
                out["passing_criteria"] = _clean_cell(value_e)
            elif sub in ("Quiz", "Assignments"):
                if sub not in out["assessment_scheme"]:
                    out["assessment_scheme"].append(sub)
    return out
```

- [ ] **Step 2: Verify Syllabus parser**

Run:
```bash
python -c "
from pathlib import Path
from src.rag.ingest.load_syllabus import _parse_syllabus_sheet
s = _parse_syllabus_sheet(Path('data/LOMA281_Syllabus_1.0.xlsx'))
print('topic_name:', s['topic_name'])
print('training_method:', s['training_method'])
print('objectives:', s['objectives'])
print('outcomes:', s['outcomes'])
print('passing_criteria:', s['passing_criteria'])
print('assessment_scheme:', s['assessment_scheme'])
assert 'Meeting Customer Needs' in s['topic_name'], s
assert s['training_method'] == 'Selft-study', s
assert len(s['objectives']) >= 5, s
assert len(s['outcomes']) >= 5, s
assert s['passing_criteria'] == '80% quiz points', s
assert 'Quiz' in s['assessment_scheme'], s
print('OK')
"
```
Expected last line: `OK`. If parsing gives wrong counts or empty lists, print `df.iloc[:20, :].to_string()` inside the function to debug row positions.

- [ ] **Step 3: Commit**

```bash
git add src/rag/ingest/load_syllabus.py
git commit -m "feat(load_syllabus): add Syllabus sheet parser"
```

---

## Task 4: `load_syllabus.py` — Schedule sheet parser (module descriptions + lesson rows + totals)

**Files:**
- Modify: `src/rag/ingest/load_syllabus.py`

- [ ] **Step 1: Append `_parse_schedule`**

Append at the end of `src/rag/ingest/load_syllabus.py`:

```python
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

    # Skip rows 0 (title) and 1 (header). Start at row 2.
    for idx in range(2, len(df)):
        row = df.iloc[idx]
        module_cell = _clean_cell(row.iloc[1]) if len(row) > 1 else ""
        lesson_cell = _clean_cell(row.iloc[2]) if len(row) > 2 else ""
        section_cell = _clean_cell(row.iloc[3]) if len(row) > 3 else ""
        delivery_cell = _clean_cell(row.iloc[4]) if len(row) > 4 else ""
        duration_val = row.iloc[5] if len(row) > 5 else None
        directory_cell = _clean_cell(row.iloc[8]) if len(row) > 8 else ""
        remark_cell = _clean_cell(row.iloc[9]) if len(row) > 9 else ""

        # Summary rows at the bottom have the label in col B and number in col C.
        if module_cell in totals_labels and lesson_cell:
            totals[totals_labels[module_cell]] = _to_float(lesson_cell)
            continue

        # New module header row (col B non-empty, starts with "Module N:")
        mod_num, mod_title, mod_desc = _parse_module_header(module_cell)
        if mod_num:
            current_mod_num = mod_num
            modules[str(mod_num)] = {
                "module_num": mod_num,
                "title": mod_title,
                "description": mod_desc,
                "lessons": {},
            }

        # New lesson header row (col C non-empty, starts with "Lesson N:")
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

        # Quiz / Review continuation rows (no lesson header, but delivery cell set)
        if current_lesson is not None and delivery_cell in ("Quiz", "Review"):
            key = "quiz_hours" if delivery_cell == "Quiz" else "review_hours"
            current_lesson[key] = _to_float(duration_val)

    return modules, totals
```

- [ ] **Step 2: Verify Schedule parser**

Run:
```bash
python -c "
from pathlib import Path
from src.rag.ingest.load_syllabus import _parse_schedule
modules, totals = _parse_schedule(Path('data/LOMA281_Syllabus_1.0.xlsx'))
print('#modules:', len(modules))
print('M1 title:', modules['1']['title'])
print('M1 desc len:', len(modules['1']['description']))
print('M1 lessons:', list(modules['1']['lessons'].keys()))
m1l1 = modules['1']['lessons']['1']
print('M1L1 title:', m1l1['title'])
print('M1L1 hours:', m1l1['self_learning_hours'], m1l1['quiz_hours'], m1l1['review_hours'])
print('M1L1 dir:', m1l1['directory'])
print('totals:', totals)
assert len(modules) == 4, modules.keys()
assert modules['1']['title'] == 'Risk and Insurance', modules['1']
assert 'In this module' in modules['1']['description'], modules['1']['description'][:80]
assert list(modules['1']['lessons'].keys()) == ['1', '2', '3', '4'], list(modules['1']['lessons'].keys())
assert m1l1['self_learning_hours'] == 2.0
assert m1l1['quiz_hours'] == 0.5
assert m1l1['review_hours'] == 0.5
assert totals['total_hours'] == 42.0, totals
assert totals['self_learning_hours'] == 28.0, totals
print('OK')
"
```
Expected last line: `OK`. If module/lesson count is wrong, inspect `df.iloc[:30, 1:4]` to see how Module/Lesson text is distributed across rows.

- [ ] **Step 3: Also verify LOMA291 (different shape — 4 modules, some with only 2 lessons)**

Run:
```bash
python -c "
from pathlib import Path
from src.rag.ingest.load_syllabus import _parse_schedule
modules, totals = _parse_schedule(Path('data/LOMA291_Syllabus_1.0.xlsx'))
print('#modules:', len(modules))
for k, m in modules.items():
    print(f'  M{k}: {m[\"title\"]} ({len(m[\"lessons\"])} lessons)')
print('totals:', totals)
assert len(modules) == 4, list(modules.keys())
assert totals['total_hours'] == 36.0, totals
print('OK')
"
```
Expected last line: `OK`.

- [ ] **Step 4: Commit**

```bash
git add src/rag/ingest/load_syllabus.py
git commit -m "feat(load_syllabus): add Schedule sheet parser for modules + totals"
```

---

## Task 5: `load_syllabus.py` — combine + `main()` + write JSON

**Files:**
- Modify: `src/rag/ingest/load_syllabus.py`

- [ ] **Step 1: Append the top-level orchestrator and `main()`**

Append at the end of `src/rag/ingest/load_syllabus.py`:

```python
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
    main()
```

- [ ] **Step 2: Run end-to-end and verify JSON**

Run:
```bash
python -m src.rag.ingest.load_syllabus
```
Expected output (approximately):
```
✅ Wrote 2 courses → D:\...\data\ingest_overall\syllabus.json
   LOMA 281: 4 modules, 14 lessons
   LOMA 291: 4 modules, 12 lessons
```

Then verify the JSON:
```bash
python -c "
import json
from src.constants.app_constant import SYLLABUS_JSON
blob = json.loads(SYLLABUS_JSON.read_text(encoding='utf-8'))
assert set(blob.keys()) == {'LOMA 281', 'LOMA 291'}, blob.keys()
c = blob['LOMA 281']
assert c['totals']['total_hours'] == 42.0
assert len(c['objectives']) >= 5
assert c['modules']['1']['lessons']['1']['self_learning_hours'] == 2.0
assert 'Risk' in c['modules']['1']['lessons']['1']['objectives']
print('OK')
"
```
Expected last line: `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/rag/ingest/load_syllabus.py data/ingest_overall/syllabus.json
git commit -m "feat(load_syllabus): add main() and generate syllabus.json"
```

---

## Task 6: Enrich `course` chunks (F1)

**Files:**
- Modify: `src/rag/ingest/build_overall_chunks.py`

- [ ] **Step 1: Add imports + helpers for course matching at top of file**

Open `src/rag/ingest/build_overall_chunks.py`. Find:
```python
from src.constants.app_constant import METADATA_NODE_JSON, OVERALL_INGEST_DIR
```
Replace with:
```python
from src.constants.app_constant import (
    METADATA_NODE_JSON, OVERALL_INGEST_DIR, SYLLABUS_JSON,
)
```

Find the existing `_lesson_number` function. Add these helpers immediately below it:

```python
_COURSE_CODE_RE = re.compile(r"LOMA\s+\d+")


def _course_code(course: str) -> str:
    """'LOMA 281 - Meeting Customer Needs...' -> 'LOMA 281' (or '' if no match)."""
    m = _COURSE_CODE_RE.search(course)
    return m.group(0) if m else ""


def _fmt_hours(h: float) -> str:
    """2.0 -> '2h', 0.5 -> '0.5h'."""
    if h == int(h):
        return f"{int(h)}h"
    return f"{h}h"
```

- [ ] **Step 2: Modify `_build_course_chunks` to accept `syllabus` and append enrichment**

Replace the existing `_build_course_chunks` function with:

```python
def _build_course_chunks(rows: list[dict], syllabus: dict) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["Course"]].append(r)

    chunks: list[dict] = []
    for course, nodes in grouped.items():
        module_nodes: dict[str, list[dict]] = defaultdict(list)
        lesson_set: set[tuple[str, str]] = set()
        for n in nodes:
            module_nodes[n["Module"]].append(n)
            lesson_set.add((n["Module"], n["Lesson"]))

        sorted_modules = sorted(
            module_nodes.items(), key=lambda x: _module_number(x[0])
        )
        module_lines: list[str] = []
        for module, mnodes in sorted_modules:
            mlessons = {n["Lesson"] for n in mnodes}
            module_lines.append(f"- {module} ({len(mlessons)} lessons)")

        cat_counts = Counter(n["Category"] for n in nodes)
        top_cats = [c for c, _ in cat_counts.most_common(_TOP_CATEGORIES_COURSE)]

        text_lines = [
            course,
            f"Number of modules: {len(sorted_modules)}",
            f"Number of lessons: {len(lesson_set)}",
            f"Number of nodes: {len(nodes)}",
            "Modules:",
            *module_lines,
            "",
            f"Main topics: {', '.join(top_cats)}",
        ]

        payload = {
            "chunk_type": "course",
            "course": course,
            "module_count": len(sorted_modules),
            "lesson_count": len(lesson_set),
            "node_count": len(nodes),
        }

        # Enrich from syllabus if matched
        syl = syllabus.get(_course_code(course))
        if syl:
            if syl["objectives"]:
                text_lines.append("")
                text_lines.append("Course Objectives:")
                text_lines.extend(f"- {o}" for o in syl["objectives"])
            if syl["outcomes"]:
                text_lines.append("")
                text_lines.append("Course Outcomes:")
                text_lines.extend(f"- {o}" for o in syl["outcomes"])
            totals = syl["totals"]
            passing = syl["assessment"]["passing_criteria"]
            if passing or totals["total_hours"]:
                text_lines.append("")
                completion = []
                if passing:
                    completion.append(f"Completion: {passing}.")
                if totals["total_hours"]:
                    completion.append(
                        f"Total {_fmt_hours(totals['total_hours'])} "
                        f"({_fmt_hours(totals['self_learning_hours'])} self-study, "
                        f"{_fmt_hours(totals['quiz_hours'])} quiz, "
                        f"{_fmt_hours(totals['review_hours'])} review)."
                    )
                text_lines.append(" ".join(completion))
            payload.update({
                "version": syl["version"],
                "effective_date": syl["effective_date"],
                "topic_code": syl["topic_code"],
                "type_of_learners": syl["type_of_learners"],
                "training_method": syl["training_method"],
                "assessment_scheme": syl["assessment"]["scheme"],
                "total_hours": totals["total_hours"],
            })

        chunks.append({
            "id": _det_id("course", course),
            "text": "\n".join(text_lines),
            "payload": payload,
        })
    return chunks
```

- [ ] **Step 3: Update `build_all_chunks` signature to thread `syllabus` in**

Find `build_all_chunks` and replace with:

```python
def build_all_chunks(rows: list[dict], syllabus: dict) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    chunks.extend(_build_node_chunks(rows))
    chunks.extend(_build_lesson_chunks(rows, syllabus))
    chunks.extend(_build_module_chunks(rows, syllabus))
    chunks.extend(_build_course_chunks(rows, syllabus))
    chunks.append(_build_overview_chunk(rows))
    return chunks
```

*Note: this will break `_build_lesson_chunks` and `_build_module_chunks` call sites until Tasks 7 and 8 update their signatures. We fix that below.*

- [ ] **Step 4: Temporarily patch `_build_lesson_chunks` and `_build_module_chunks` signatures**

These will be enriched in Tasks 7 and 8. For now, just accept and ignore the new arg so the file still imports:

In `_build_lesson_chunks`:
```python
def _build_lesson_chunks(rows: list[dict], syllabus: dict) -> list[dict]:
```
(keep body unchanged)

In `_build_module_chunks`:
```python
def _build_module_chunks(rows: list[dict], syllabus: dict) -> list[dict]:
```
(keep body unchanged)

- [ ] **Step 5: Update `main()` to load syllabus and pass to `build_all_chunks`**

Replace the existing `main()`:

```python
def main() -> None:
    with open(METADATA_NODE_JSON, "r", encoding="utf-8") as f:
        rows = json.load(f)

    syllabus: dict = {}
    if SYLLABUS_JSON.exists():
        with open(SYLLABUS_JSON, "r", encoding="utf-8") as f:
            syllabus = json.load(f)
    else:
        print(f"⚠️  {SYLLABUS_JSON} not found — skipping enrichment")

    chunks = build_all_chunks(rows, syllabus)

    OVERALL_INGEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OVERALL_INGEST_DIR / "overall_course.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    counts = Counter(c["payload"]["chunk_type"] for c in chunks)
    print(f"✅ Wrote {len(chunks)} chunks → {out_path}")
    for ctype, cnt in sorted(counts.items()):
        print(f"   {ctype}: {cnt}")
```

- [ ] **Step 6: Run end-to-end and verify course chunk enrichment**

Run:
```bash
python -m src.rag.ingest.build_overall_chunks
```
Expected: finishes without error, prints chunk counts.

Then verify:
```bash
python -c "
import json
from src.constants.app_constant import OVERALL_INGEST_DIR
chunks = json.loads((OVERALL_INGEST_DIR / 'overall_course.json').read_text(encoding='utf-8'))
course_chunks = [c for c in chunks if c['payload']['chunk_type'] == 'course']
loma281 = next(c for c in course_chunks if 'LOMA 281' in c['payload']['course'])
print(loma281['text'])
print('---')
print(loma281['payload'])
assert 'Course Objectives:' in loma281['text']
assert 'Course Outcomes:' in loma281['text']
assert 'Completion:' in loma281['text']
assert loma281['payload']['total_hours'] == 42.0
assert loma281['payload']['version'] == '1.0'
assert loma281['payload']['effective_date'] == '2025-06-26'
print('OK')
"
```
Expected last line: `OK`.

- [ ] **Step 7: Commit**

```bash
git add src/rag/ingest/build_overall_chunks.py data/ingest_overall/overall_course.json
git commit -m "feat(build_overall_chunks): enrich course chunks with syllabus data"
```

---

## Task 7: Enrich `module` chunks (E1)

**Files:**
- Modify: `src/rag/ingest/build_overall_chunks.py`

- [ ] **Step 1: Replace `_build_module_chunks` with enriched version**

Replace the existing `_build_module_chunks` function (the temporary version from Task 6 Step 4) with:

```python
def _build_module_chunks(rows: list[dict], syllabus: dict) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["Course"], r["Module"])].append(r)

    chunks: list[dict] = []
    for (course, module), nodes in grouped.items():
        lesson_nodes: dict[str, list[dict]] = defaultdict(list)
        for n in nodes:
            lesson_nodes[n["Lesson"]].append(n)
        sorted_lessons = sorted(
            lesson_nodes.items(), key=lambda x: _lesson_number(x[0])
        )
        lesson_lines = [
            f"- {lesson} ({len(nlist)} nodes)"
            for lesson, nlist in sorted_lessons
        ]
        cat_counts = Counter(n["Category"] for n in nodes)
        cats_sorted = [c for c, _ in cat_counts.most_common()]

        text_lines = [f"{module} (Course: {course})"]

        # Module Overview from syllabus (E1): prepend after header line
        mod_num = _module_number(module)
        syl = syllabus.get(_course_code(course))
        if syl and str(mod_num) in syl["modules"]:
            desc = syl["modules"][str(mod_num)]["description"]
            if desc:
                text_lines.append("")
                text_lines.append("Module Overview (from syllabus):")
                text_lines.append(desc)
                text_lines.append("")

        text_lines.extend([
            f"Number of lessons: {len(sorted_lessons)}",
            f"Number of nodes: {len(nodes)}",
            "Lessons:",
            *lesson_lines,
            "",
            f"Categories covered: {', '.join(cats_sorted)}",
        ])

        chunks.append({
            "id": _det_id("module", course, module),
            "text": "\n".join(text_lines),
            "payload": {
                "chunk_type": "module",
                "course": course,
                "module": mod_num,
                "lesson_count": len(sorted_lessons),
                "node_count": len(nodes),
            },
        })
    return chunks
```

- [ ] **Step 2: Run end-to-end and verify module chunk enrichment**

Run:
```bash
python -m src.rag.ingest.build_overall_chunks
```
Expected: finishes without error.

Verify:
```bash
python -c "
import json
from src.constants.app_constant import OVERALL_INGEST_DIR
chunks = json.loads((OVERALL_INGEST_DIR / 'overall_course.json').read_text(encoding='utf-8'))
mods = [c for c in chunks if c['payload']['chunk_type'] == 'module']
m = next(c for c in mods if 'LOMA 281' in c['payload']['course'] and c['payload']['module'] == 1)
print(m['text'])
assert 'Module Overview (from syllabus):' in m['text']
assert 'In this module' in m['text']
# Overview must appear before 'Number of lessons:' line
idx_ov = m['text'].index('Module Overview')
idx_nl = m['text'].index('Number of lessons:')
assert idx_ov < idx_nl, 'Overview should appear before Number of lessons'
print('OK')
"
```
Expected last line: `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/rag/ingest/build_overall_chunks.py data/ingest_overall/overall_course.json
git commit -m "feat(build_overall_chunks): enrich module chunks with syllabus overview"
```

---

## Task 8: Enrich `lesson` chunks (D1)

**Files:**
- Modify: `src/rag/ingest/build_overall_chunks.py`

- [ ] **Step 1: Replace `_build_lesson_chunks` with enriched version**

Replace the existing `_build_lesson_chunks` function (the temporary version from Task 6 Step 4) with:

```python
def _build_lesson_chunks(rows: list[dict], syllabus: dict) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["Course"], r["Module"], r["Lesson"])
        grouped[key].append(r)

    chunks: list[dict] = []
    for (course, module, lesson), nodes in grouped.items():
        first = nodes[0]
        concept_lines = [f"- {n['Node Name']}: {n['Definition']}" for n in nodes]
        text_lines = [
            f"{lesson} (Course: {course}, {module})",
            f"Number of nodes: {len(nodes)}",
            "Concepts covered:",
            *concept_lines,
            "",
            f"Lesson Summary (from source): {first['Summary']}",
        ]

        payload = {
            "chunk_type": "lesson",
            "course": course,
            "module": _module_number(module),
            "lesson": _lesson_number(lesson),
            "node_count": len(nodes),
        }

        # Syllabus enrichment (D1)
        mod_num = _module_number(module)
        les_num = _lesson_number(lesson)
        syl = syllabus.get(_course_code(course))
        lesson_syl = None
        if syl and str(mod_num) in syl["modules"]:
            lesson_syl = syl["modules"][str(mod_num)]["lessons"].get(str(les_num))
        if lesson_syl:
            if lesson_syl["objectives"]:
                text_lines.append("")
                text_lines.append("Learning Objectives (from syllabus):")
                text_lines.append(lesson_syl["objectives"])
            text_lines.append("")
            text_lines.append(
                f"Duration: {_fmt_hours(lesson_syl['self_learning_hours'])} self-study, "
                f"{_fmt_hours(lesson_syl['quiz_hours'])} quiz, "
                f"{_fmt_hours(lesson_syl['review_hours'])} review"
            )
            payload.update({
                "delivery_mode": lesson_syl["delivery_mode"],
                "directory": lesson_syl["directory"],
                "remark": lesson_syl["remark"],
                "self_learning_hours": lesson_syl["self_learning_hours"],
                "quiz_hours": lesson_syl["quiz_hours"],
                "review_hours": lesson_syl["review_hours"],
            })

        chunks.append({
            "id": _det_id("lesson", course, module, lesson),
            "text": "\n".join(text_lines),
            "payload": payload,
        })
    return chunks
```

- [ ] **Step 2: Run end-to-end and verify lesson chunk enrichment**

Run:
```bash
python -m src.rag.ingest.build_overall_chunks
```

Verify:
```bash
python -c "
import json
from src.constants.app_constant import OVERALL_INGEST_DIR
chunks = json.loads((OVERALL_INGEST_DIR / 'overall_course.json').read_text(encoding='utf-8'))
lessons = [c for c in chunks if c['payload']['chunk_type'] == 'lesson']
l = next(c for c in lessons if 'LOMA 281' in c['payload']['course'] and c['payload']['module'] == 1 and c['payload']['lesson'] == 1)
print(l['text'])
print('---')
print(l['payload'])
assert 'Learning Objectives (from syllabus):' in l['text']
assert 'Duration: 2h self-study, 0.5h quiz, 0.5h review' in l['text']
assert l['payload']['self_learning_hours'] == 2.0
assert l['payload']['directory'] == 'Lesson 1'
assert 'LOMA281_M1L1' in l['payload']['remark']
print('OK')
"
```
Expected last line: `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/rag/ingest/build_overall_chunks.py data/ingest_overall/overall_course.json
git commit -m "feat(build_overall_chunks): enrich lesson chunks with syllabus objectives and duration"
```

---

## Task 9: Wire `load_syllabus.main()` into `build_overall_chunks.main()`

**Files:**
- Modify: `src/rag/ingest/build_overall_chunks.py`

- [ ] **Step 1: Import `load_syllabus.main` and call it unconditionally**

Find in `src/rag/ingest/build_overall_chunks.py`:
```python
from src.constants.app_constant import (
    METADATA_NODE_JSON, OVERALL_INGEST_DIR, SYLLABUS_JSON,
)
```

Add below it:
```python
from src.rag.ingest.load_syllabus import main as load_syllabus_main
```

Replace the `main()` function with:

```python
def main() -> None:
    syllabus = load_syllabus_main()

    with open(METADATA_NODE_JSON, "r", encoding="utf-8") as f:
        rows = json.load(f)

    chunks = build_all_chunks(rows, syllabus)

    OVERALL_INGEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OVERALL_INGEST_DIR / "overall_course.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    counts = Counter(c["payload"]["chunk_type"] for c in chunks)
    print(f"✅ Wrote {len(chunks)} chunks → {out_path}")
    for ctype, cnt in sorted(counts.items()):
        print(f"   {ctype}: {cnt}")
```

- [ ] **Step 2: Delete stale `syllabus.json` and run end-to-end to verify it's regenerated**

Run:
```bash
python -c "from src.constants.app_constant import SYLLABUS_JSON; SYLLABUS_JSON.unlink(missing_ok=True)"
python -m src.rag.ingest.build_overall_chunks
```

Expected output (approximately):
```
✅ Wrote 2 courses → ...\syllabus.json
   LOMA 281: 4 modules, 14 lessons
   LOMA 291: 4 modules, 12 lessons
✅ Wrote <N> chunks → ...\overall_course.json
   course: 2
   lesson: 26
   module: 8
   node: ...
   overview: 1
```

Verify `syllabus.json` was regenerated:
```bash
python -c "
from src.constants.app_constant import SYLLABUS_JSON
assert SYLLABUS_JSON.exists(), 'syllabus.json was not regenerated'
print('OK')
"
```
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/rag/ingest/build_overall_chunks.py
git commit -m "feat(build_overall_chunks): auto-regenerate syllabus.json on each run"
```

---

## Task 10: Final full-pipeline verification

**Files:** (no code changes — verification only)

- [ ] **Step 1: Run the full builder one more time, clean**

```bash
python -m src.rag.ingest.build_overall_chunks
```

Expected: both `✅ Wrote ... syllabus.json` and `✅ Wrote ... overall_course.json` lines, chunk counts unchanged from before this feature (same set of chunk IDs — enrichment does not add or remove chunks).

- [ ] **Step 2: Sanity-check both courses across all 3 enriched chunk types**

Run this verification script:
```bash
python -c "
import json
from src.constants.app_constant import OVERALL_INGEST_DIR
chunks = json.loads((OVERALL_INGEST_DIR / 'overall_course.json').read_text(encoding='utf-8'))

# Basic counts
by_type = {}
for c in chunks:
    by_type.setdefault(c['payload']['chunk_type'], []).append(c)
print({k: len(v) for k, v in by_type.items()})

assert len(by_type['course']) == 2
assert len(by_type['module']) == 8  # 4 modules x 2 courses
assert 'overview' in by_type

# Both courses enriched at every level
for code in ('LOMA 281', 'LOMA 291'):
    course = next(c for c in by_type['course'] if code in c['payload']['course'])
    assert 'Course Objectives:' in course['text'], code
    assert 'Course Outcomes:' in course['text'], code
    assert 'Completion:' in course['text'], code
    assert course['payload'].get('version') == '1.0', code
    assert course['payload'].get('total_hours', 0) > 0, code

    mods = [m for m in by_type['module'] if code in m['payload']['course']]
    assert len(mods) == 4, (code, len(mods))
    for m in mods:
        assert 'Module Overview (from syllabus):' in m['text'], (code, m['payload']['module'])

    lessons = [l for l in by_type['lesson'] if code in l['payload']['course']]
    assert all('Learning Objectives (from syllabus):' in l['text'] for l in lessons), code
    assert all('Duration:' in l['text'] for l in lessons), code

# Deterministic IDs unchanged: spot-check LOMA 281 course
import uuid
ns = uuid.UUID('12345678-1234-5678-1234-567812345678')
expected = str(uuid.uuid5(ns, 'course::LOMA 281 - Meeting Customer Needs with Insurance and Annuities - 3rd Edition'))
actual = next(c for c in by_type['course'] if 'LOMA 281' in c['payload']['course'])['id']
assert actual == expected, (actual, expected)
print('OK')
"
```

Expected last line: `OK`. If any assertion fails, identify which enrichment is missing and revisit the corresponding Task (6/7/8).

- [ ] **Step 3: Confirm no stray changes**

Run:
```bash
git status
```
Expected: clean tree (everything committed). Regenerated JSON may or may not differ from last-committed version — commit any updated JSON:
```bash
git add data/ingest_overall/
git commit -m "chore: regenerate overall_course and syllabus JSON" || echo "nothing to commit"
```

- [ ] **Step 4: Hand-off note to user (no git action)**

Print a short status: syllabus.json generated, 2 courses / 8 modules / 26 lessons enriched, `overall_course.json` rebuilt with identical chunk IDs. Remind the user they can now hit:
```
GET /documents/ingest?collection_name=overall_course&force_restart=true
```
to push the enriched chunks to Qdrant (spec H2 — user triggers this, plan does not).

---

## Self-Review Summary

**Spec coverage:** every decision locked in Q&A (A, A2, B1, C1, D1, E1, F1, G1, H2) has a task: A2→Tasks 2–9, B1→Task 6 Step 1 (`_course_code`), C1→Task 4 (objectives passed verbatim), D1→Task 8, E1→Task 7, F1→Task 6, G1→overview is untouched (nothing added), H2→Task 10 Step 4.

**Spec §4 edge cases:**
- Missing syllabus for a course → Task 6/7/8 use `.get()` with `if syl` guard; no enrichment applied, chunk produced normally.
- Mismatched module/lesson numbers → Task 7/8 use `.get()` on the inner dict.
- Empty objectives cell → Task 8 guards with `if lesson_syl["objectives"]`.
- `Selft-study` typo → echoed verbatim in payload.
- Effective date cell as Timestamp → handled in Task 2 `_parse_cover` (`strftime` if `pd.Timestamp`, else raw).
- Duration NaN → `_to_float` in Task 2 returns 0.0.
- Duration formatting (`Xh` dropping `.0`) → Task 6 `_fmt_hours` used consistently in Tasks 6 and 8.

**Type consistency:**
- `_course_code`, `_fmt_hours` defined in Task 6, used in Tasks 7 & 8 — all reference the same names.
- `build_all_chunks(rows, syllabus)` signature defined in Task 6 Step 3, both `_build_lesson_chunks` and `_build_module_chunks` accept matching `(rows, syllabus)` signatures from Task 6 Step 4 onward.
- `syllabus.json` schema established in Task 5 matches keys read in Tasks 6–8: `objectives`, `outcomes`, `totals.total_hours`, `assessment.passing_criteria`, `modules[str(n)].description`, `modules[str(n)].lessons[str(n)].objectives/self_learning_hours/etc`.

**Placeholder scan:** no TBD/TODO/"similar to". Every code step shows full code; every verify step shows the exact Python assertion snippet and the expected last line (`OK`).
