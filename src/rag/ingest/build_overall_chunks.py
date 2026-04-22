"""
Build text chunks tổng hợp cho intent overall_course_knowledge.
Đọc data/metadata_node.json → tạo 5 loại chunk (node/lesson/module/course/overview)
→ ghi ra data/ingest_overall/overall_course.json.

Chạy: python -m src.rag.ingest.build_overall_chunks
"""
import json
import re
import uuid
from collections import Counter, defaultdict
from typing import Any

from src.constants.app_constant import METADATA_NODE_JSON, OVERALL_INGEST_DIR

# Namespace cố định để generate deterministic UUID v5 — cho phép
# rebuild + re-ingest mà không tạo duplicate point.
_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

_TOP_CATEGORIES_COURSE = 8


def _det_id(*parts: str) -> str:
    name = "::".join(parts)
    return str(uuid.uuid5(_NAMESPACE, name))


def _module_number(module_str: str) -> int:
    m = re.match(r"Module\s+(\d+)", module_str)
    return int(m.group(1)) if m else 0


def _lesson_number(lesson_str: str) -> int:
    m = re.match(r"Lesson\s+(\d+)", lesson_str)
    return int(m.group(1)) if m else 0


def _build_node_chunks(rows: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for r in rows:
        text_parts = [
            f"Node {r['Node ID']} — {r['Node Name']}",
            f"Definition: {r['Definition']}",
            f"Category: {r['Category']}",
            f"Tags: {r['Domain Tags']}",
            f"Related Nodes: {r['Related Nodes']}",
            f"Course: {r['Course']} | Module: {r['Module']} | Lesson: {r['Lesson']}",
            f"Path: {r['path']}",
        ]
        chunks.append({
            "id": _det_id("node", str(r["Node ID"])),
            "text": "\n".join(text_parts),
            "payload": {
                "chunk_type": "node",
                "node_id": int(r["Node ID"]),
                "node_name": r["Node Name"],
                "category": r["Category"],
                "course": r["Course"],
                "module": _module_number(r["Module"]),
                "lesson": _lesson_number(r["Lesson"]),
                "tags": r["Domain Tags"],
                "file_name": r["Source"],
            },
        })
    return chunks


def _build_lesson_chunks(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["Course"], r["Module"], r["Lesson"])
        grouped[key].append(r)

    chunks: list[dict] = []
    for (course, module, lesson), nodes in grouped.items():
        first = nodes[0]
        concept_lines = [f"- {n['Node Name']}: {n['Definition']}" for n in nodes]
        text = "\n".join([
            f"{lesson} (Course: {course}, {module})",
            f"Number of nodes: {len(nodes)}",
            "Concepts covered:",
            *concept_lines,
            "",
            f"Lesson Summary (from source): {first['Summary']}",
        ])
        chunks.append({
            "id": _det_id("lesson", course, module, lesson),
            "text": text,
            "payload": {
                "chunk_type": "lesson",
                "course": course,
                "module": _module_number(module),
                "lesson": _lesson_number(lesson),
                "node_count": len(nodes),
            },
        })
    return chunks


def _build_module_chunks(rows: list[dict]) -> list[dict]:
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

        text = "\n".join([
            f"{module} (Course: {course})",
            f"Number of lessons: {len(sorted_lessons)}",
            f"Number of nodes: {len(nodes)}",
            "Lessons:",
            *lesson_lines,
            "",
            f"Categories covered: {', '.join(cats_sorted)}",
        ])
        chunks.append({
            "id": _det_id("module", course, module),
            "text": text,
            "payload": {
                "chunk_type": "module",
                "course": course,
                "module": _module_number(module),
                "lesson_count": len(sorted_lessons),
                "node_count": len(nodes),
            },
        })
    return chunks


def _build_course_chunks(rows: list[dict]) -> list[dict]:
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

        text = "\n".join([
            course,
            f"Number of modules: {len(sorted_modules)}",
            f"Number of lessons: {len(lesson_set)}",
            f"Number of nodes: {len(nodes)}",
            "Modules:",
            *module_lines,
            "",
            f"Main topics: {', '.join(top_cats)}",
        ])
        chunks.append({
            "id": _det_id("course", course),
            "text": text,
            "payload": {
                "chunk_type": "course",
                "course": course,
                "module_count": len(sorted_modules),
                "lesson_count": len(lesson_set),
                "node_count": len(nodes),
            },
        })
    return chunks


def _build_overview_chunk(rows: list[dict]) -> dict:
    tree: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    course_node_count: Counter = Counter()
    course_lesson_set: dict[str, set[tuple[str, str]]] = defaultdict(set)

    for r in rows:
        course = r["Course"]
        module = r["Module"]
        lesson = r["Lesson"]
        tree[course][module].add(lesson)
        course_node_count[course] += 1
        course_lesson_set[course].add((module, lesson))

    courses_sorted = sorted(tree.keys())

    lines = [
        "LOMA Overview",
        "LOMA stands for Life Office Management Association.",
        f"LOMA has {len(courses_sorted)} courses available in this knowledge base:",
    ]
    for course in courses_sorted:
        modules_count = len(tree[course])
        lessons_count = len(course_lesson_set[course])
        nodes_count = course_node_count[course]
        lines.append(
            f"- {course}: {modules_count} modules, {lessons_count} lessons, {nodes_count} nodes"
        )

    lines.append("")
    lines.append("Full Course Tree:")
    for course in courses_sorted:
        lines.append(f"+ {course}")
        modules_sorted = sorted(tree[course].keys(), key=_module_number)
        for module in modules_sorted:
            lines.append(f"  + {module}")
            lessons_sorted = sorted(tree[course][module], key=_lesson_number)
            for lesson in lessons_sorted:
                lines.append(f"    - {lesson}")

    return {
        "id": _det_id("overview"),
        "text": "\n".join(lines),
        "payload": {"chunk_type": "overview"},
    }


def build_all_chunks(rows: list[dict]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    chunks.extend(_build_node_chunks(rows))
    chunks.extend(_build_lesson_chunks(rows))
    chunks.extend(_build_module_chunks(rows))
    chunks.extend(_build_course_chunks(rows))
    chunks.append(_build_overview_chunk(rows))
    return chunks


def main() -> None:
    with open(METADATA_NODE_JSON, "r", encoding="utf-8") as f:
        rows = json.load(f)

    chunks = build_all_chunks(rows)

    OVERALL_INGEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OVERALL_INGEST_DIR / "overall_course.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    counts = Counter(c["payload"]["chunk_type"] for c in chunks)
    print(f"✅ Wrote {len(chunks)} chunks → {out_path}")
    for ctype, cnt in sorted(counts.items()):
        print(f"   {ctype}: {cnt}")


if __name__ == "__main__":
    main()
