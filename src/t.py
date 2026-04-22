"""
LOMA28 RAG Pipeline — Extraction → Cleaning → Chunking → Vector DB Ready
=========================================================================
File type  : LOMA eLearning .DOC (Word 2007+ internally)
Content    : LOMA 281 Lesson 2 — Life Insurance Policy Provisions
Destination: Vector DB (FAISS / Chroma / Qdrant / Weaviate …)
"""

import re
import json
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional


def count_tokens(text: str) -> int:
    """Approximate token count: ~0.75 words per token for English."""
    return int(len(text.split()) * 1.33)


# ─────────────────────────────────────────────
# STEP 1: EXTRACTION
# ─────────────────────────────────────────────

def extract_raw(path: str) -> str:
    """
    Dùng extract-text (markitdown) với --format docx vì file .DOC
    thực ra là Word 2007+ (OOXML), không cần chuyển qua LibreOffice.
    Output là Markdown với heading hierarchy giữ nguyên.
    """
    result = subprocess.run(
        ["extract-text", path, "--format", "docx"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"extract-text failed: {result.stderr}")
    return result.stdout


# ─────────────────────────────────────────────
# STEP 2: CLEANING — 7 lớp filter
# ─────────────────────────────────────────────

# Pattern theo thứ tự ưu tiên — áp dụng tuần tự
CLEANING_RULES = [

    # 2a. Xóa URL dạng Markdown link — giữ lại label text
    # VD: [Before Death](https://services.loma.org/...) → Before Death
    (re.compile(r'\[([^\]]+)\]\(https?://[^\)]+\)'), r'\1'),

    # 2b. Xóa JavaScript pseudo-links — không có nội dung hữu ích
    # VD: [Submit](javascript:Question_Comp_Submit(13);)
    (re.compile(r'\[([^\]]+)\]\(javascript:[^\)]+\)'), r''),

    # 2c. Xóa UI noise từ eLearning interactive
    # VD: "(Click or touch the arrows to the right.)"
    #     "(Click or touch each tab below.)"
    (re.compile(r'\(Click or touch[^\)]+\)\s*'), ''),

    # 2d. Xóa "Open Transcript" standalone lines
    (re.compile(r'^Open Transcript\s*$', re.MULTILINE), ''),

    # 2e. Chuẩn hóa bold: **text** giữ nguyên nếu inline,
    #     nhưng xóa bold trùng lặp ở heading đã có ##
    #     VD: ## **Incontestability Provision** → ## Incontestability Provision
    (re.compile(r'^(#{1,4})\s+\*\*(.+?)\*\*\s*$', re.MULTILINE), r'\1 \2'),

    # 2f. Xóa duplicate "## Learning Objective" headers (xuất hiện 2 lần mỗi section)
    #     Giữ lại lần đầu, drop lần 2 (cùng tên ngay sau ## Section header)
    # — được xử lý trong clean() dưới dạng logic, không phải regex đơn thuần

    # 2g. Collapse 3+ blank lines → 2 blank lines
    (re.compile(r'\n{3,}'), '\n\n'),
]


def clean(raw: str) -> str:
    text = raw

    # Áp dụng regex rules
    for pattern, replacement in CLEANING_RULES:
        text = pattern.sub(replacement, text)

    # 2f. Dedup "Learning Objective" headers (xuất hiện double vì structure file)
    lines = text.split('\n')
    seen_lo_at = {}  # section_name → index đầu tiên thấy LO header
    cleaned_lines = []
    prev_h2 = None
    for line in lines:
        if re.match(r'^## .+', line) and 'Learning Objective' not in line:
            prev_h2 = line.strip()
        if re.match(r'^## \*?Learning Objective\*?', line):
            key = prev_h2
            if key in seen_lo_at:
                # Đây là LO duplicate → drop
                continue
            seen_lo_at[key] = True
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # 2h. Strip image description placeholders
    text = re.sub(r'^Image Description\s*$', '', text, flags=re.MULTILINE)

    # 2i. Collapse lại blank lines sau khi clean
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ─────────────────────────────────────────────
# STEP 3: CHUNKING — Hierarchical Semantic
# ─────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str  # e.g. "loma281_l2_grace_period_001"
    content: str  # nội dung chunk thuần text
    token_count: int
    # ── Metadata cho retrieval & filtering ──
    source_file: str
    module: str  # "Module 3: Benefits, Provisions, and Ownership Rights"
    lesson: str  # "Lesson 2: Life Insurance Policy Provisions"
    section: str  # H2-level section name
    subsection: str  # H3-level (nếu có)
    chunk_type: str  # concept | quiz_qa | example | definition | objective
    breadcrumb: str  # "Module 3 > Lesson 2 > Grace Period > Reinstatement"


CHUNK_TARGET_TOKENS = 350  # optimal cho embedding (không quá dài, đủ context)
CHUNK_MAX_TOKENS = 512
CHUNK_OVERLAP_LINES = 3  # overlap giữa các chunk cùng subsection


def classify_chunk_type(content: str) -> str:
    """Phân loại loại nội dung để tag metadata."""
    lower = content.lower()
    if any(k in lower for k in ['true\nfalse', 'true or false', 'select each', 'choose all']):
        return 'quiz_qa'
    if re.search(r'\b(example|for example|e\.g\.)\b', lower):
        return 'example'
    if re.search(r'\*\*[A-Z][^*]{3,50}\*\*\s*[:—]', content):
        return 'definition'
    if 'learning objective' in lower or 'after studying' in lower:
        return 'objective'
    return 'concept'


def slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')[:40]


def split_into_chunks(text: str, source_file: str = "LOMA28_1.DOC") -> list[Chunk]:
    """
    Chiến lược chunking 3 tầng:
    ┌─ Tầng 1: Tách theo H2 (section chính) ─────────────────────────┐
    │  ┌─ Tầng 2: Tách theo H3/H4 (subsection) ──────────────────────┤
    │  │  ┌─ Tầng 3: Sliding window nếu subsection > MAX_TOKENS ─────┤
    │  │  └──────────────────────────────────────────────────────────┘
    │  └──────────────────────────────────────────────────────────────┘
    └──────────────────────────────────────────────────────────────────┘

    Prefix metadata được nhúng vào đầu mỗi chunk để embedding có context:
    "[Module 3 > Lesson 2 > Grace Period > Reinstatement] <content>"
    """
    lines = text.split('\n')

    module_name = ""
    lesson_name = ""
    current_h2 = ""
    current_h3 = ""
    buffer = []
    chunks = []
    chunk_index = [0]  # dùng list để có thể mutate trong nested func

    def flush(buf: list[str], h2: str, h3: str, ctype_hint: Optional[str] = None):
        content = '\n'.join(buf).strip()
        if not content or count_tokens(content) < 20:
            return

        breadcrumb = " > ".join(filter(None, [
            "Module 3", "Lesson 2", h2, h3
        ]))

        # Nhúng breadcrumb vào đầu chunk — giúp embedding hiểu context
        prefixed = f"[{breadcrumb}]\n\n{content}"

        ctype = ctype_hint or classify_chunk_type(content)
        tok = count_tokens(prefixed)

        # Nếu chunk quá lớn → sliding window
        if tok > CHUNK_MAX_TOKENS:
            sub_chunks = sliding_window(prefixed, breadcrumb, h2, h3, source_file, chunk_index)
            chunks.extend(sub_chunks)
        else:
            chunk_index[0] += 1
            cid = f"loma281_l2_{slugify(h2)}_{chunk_index[0]:03d}"
            chunks.append(Chunk(
                chunk_id=cid,
                content=prefixed,
                token_count=tok,
                source_file=source_file,
                module=module_name,
                lesson=lesson_name,
                section=h2,
                subsection=h3,
                chunk_type=ctype,
                breadcrumb=breadcrumb,
            ))

    def sliding_window(prefixed: str, breadcrumb: str, h2: str, h3: str,
                       source_file: str, idx: list) -> list[Chunk]:
        """Chia chunk lớn thành các cửa sổ trượt có overlap."""
        result = []
        words = prefixed.split()
        step = CHUNK_TARGET_TOKENS  # ~350 tokens per step
        win = CHUNK_MAX_TOKENS  # window = 512 tokens
        i = 0
        while i < len(words):
            segment = words[i:i + win]
            seg_text = ' '.join(segment)
            if count_tokens(seg_text) < 20:
                break
            idx[0] += 1
            cid = f"loma281_l2_{slugify(h2)}_{idx[0]:03d}"
            result.append(Chunk(
                chunk_id=cid,
                content=seg_text,
                token_count=count_tokens(seg_text),
                source_file=source_file,
                module=module_name,
                lesson=lesson_name,
                section=h2,
                subsection=h3,
                chunk_type=classify_chunk_type(seg_text),
                breadcrumb=breadcrumb,
            ))
            i += step  # overlap tự nhiên = win - step tokens
        return result

    # ── Quiz Q&A: tách riêng thành chunk độc lập ──
    # Pattern: câu hỏi + danh sách lựa chọn + answer nếu có
    def maybe_flush_as_qa(buf: list[str], h2: str, h3: str):
        """Nếu buffer chứa quiz Q&A, tạo chunk type='quiz_qa'."""
        flush(buf, h2, h3, ctype_hint='quiz_qa')

    # ── Main parsing loop ──
    for line in lines:
        h1 = re.match(r'^# (.+)', line)
        h2 = re.match(r'^## (.+)', line)
        h3 = re.match(r'^#{3,4} (.+)', line)

        if h1:
            module_name = h1.group(1).replace('Module 3: ', '', 1)
            continue

        if h2:
            # Flush buffer của section trước
            if buffer:
                flush(buffer, current_h2, current_h3)
                buffer = []
            current_h2 = h2.group(1)
            current_h3 = ""
            continue

        if h3:
            # Flush buffer của subsection trước
            if buffer:
                flush(buffer, current_h2, current_h3)
                buffer = []
            current_h3 = h3.group(1)
            # Thêm header vào buffer mới
            buffer = [f"### {current_h3}"]
            continue

        if line.startswith('## Lesson'):
            lesson_name = re.sub(r'^## ', '', line)
            continue

        buffer.append(line)

        # Flush sớm nếu buffer quá lớn (tránh accumulate vô hạn)
        if count_tokens('\n'.join(buffer)) > CHUNK_MAX_TOKENS * 1.5:
            flush(buffer, current_h2, current_h3)
            # Giữ lại OVERLAP_LINES dòng cuối cho context
            buffer = buffer[-CHUNK_OVERLAP_LINES:] if len(buffer) > CHUNK_OVERLAP_LINES else []

    # Flush phần cuối
    if buffer:
        flush(buffer, current_h2, current_h3)

    return chunks


def sliding_window(prefixed, breadcrumb, h2, h3, source_file, idx):
    pass  # defined inside split_into_chunks


# ─────────────────────────────────────────────
# STEP 4: RUN & REPORT
# ─────────────────────────────────────────────

def run(input_path: str, output_path: str):
    print(f"[1/4] Extracting from {input_path}...")
    raw = extract_raw(input_path)
    print(f"      Raw: {len(raw)} chars, {count_tokens(raw)} tokens")

    print("[2/4] Cleaning...")
    cleaned = clean(raw)
    print(f"      Cleaned: {len(cleaned)} chars, {count_tokens(cleaned)} tokens")
    removed_pct = (1 - len(cleaned) / len(raw)) * 100
    print(f"      Noise removed: {removed_pct:.1f}%")

    print("[3/4] Chunking...")
    chunks = split_into_chunks(cleaned, source_file=input_path.split('/')[-1])
    print(f"      Total chunks: {len(chunks)}")

    token_counts = [c.token_count for c in chunks]
    type_dist = {}
    for c in chunks:
        type_dist[c.chunk_type] = type_dist.get(c.chunk_type, 0) + 1

    print(f"      Token range: {min(token_counts)}–{max(token_counts)} "
          f"(avg {sum(token_counts) // len(token_counts)})")
    print(f"      Type distribution: {type_dist}")

    print(f"[4/4] Writing to {output_path}...")
    output = {
        "meta": {
            "source": input_path,
            "total_chunks": len(chunks),
            "avg_tokens": sum(token_counts) // len(token_counts),
            "type_distribution": type_dist,
        },
        "chunks": [asdict(c) for c in chunks]
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n✓ Done. Sample chunks:")
    for c in chunks[:3]:
        print(f"\n  [{c.chunk_id}] ({c.chunk_type}, {c.token_count} tokens)")
        print(f"  Breadcrumb: {c.breadcrumb}")
        print(f"  Content preview: {c.content[:120].replace(chr(10), ' ')}...")

    return chunks


if __name__ == "__main__":
    chunks = run(
        input_path=r"D:\Deverlopment\huudan.com\PythonProject\data\raw\LOMA 281 - Meeting Customer Needs with Insurance and Annuities - 3rd Edition\02_Document\Module 1 Risk and Insurance\Lesson 1 - Risky Business\LOMA281_M1L1_Knowledge File_Risk and Insurance.docx",
        output_path=r"D:\Deverlopment\huudan.com\PythonProject\data\raw\LOMA 281 - Meeting Customer Needs with Insurance and Annuities - 3rd Edition\02_Document\Module 1 Risk and Insurance\Lesson 1 - Risky Business\LOMA281_M1L1_Knowledge File_Risk and Insurance.json"
    )