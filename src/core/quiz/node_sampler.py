import asyncio
import random
from collections import defaultdict
from typing import Any

from qdrant_client import models

from src.constants.app_constant import METADATA_NODE_XLSX
from src.rag.db_vector import QdrantManager, get_qdrant_client
from src.utils.logger_utils import alog_method_call, logger

# ── Constants ─────────────────────────────────────────────────────────────────
_SCROLL_LIMIT = 100
_CATEGORY_SCROLL_LIMIT = 1_000  # category scroll chỉ lấy 1 field → dùng batch lớn
_CATEGORY_PAYLOAD = ["category"]
_CHUNK_FIELDS = (
    "chunk_index", "file_name", "node_name", "node_id",
    "category", "page_number", "total_pages", "module", "lesson", "course",
)


class NodeSampler:
    """Lấy mẫu ngẫu nhiên các category từ Qdrant và truy xuất chunk liền kề."""

    def __init__(
            self,
            xlsx_path: str | None = None,
            qdrant_manager: QdrantManager | None = None,
    ) -> None:
        self.xlsx_path = xlsx_path or str(METADATA_NODE_XLSX)
        self.qdrant = qdrant_manager or get_qdrant_client()

    # ── Filter builder ────────────────────────────────────────────────────────

    @staticmethod
    def _build_filter(
            knowledge_pack: str,
            level: str | None = None,
            value: str | None = None,
            category: str | None = None,
    ) -> models.Filter:
        """Build Qdrant filter. knowledge_pack so sánh với metadata 'course'; level/value/category là tuỳ chọn."""
        conditions = [
            models.FieldCondition(key="course", match=models.MatchValue(value=knowledge_pack))
        ]
        if level and value:
            conditions.append(
                models.FieldCondition(key=level, match=models.MatchValue(value=value))
            )
        if category:
            conditions.append(
                models.FieldCondition(key="category", match=models.MatchValue(value=category))
            )
        return models.Filter(must=conditions)

    # ── Category discovery ────────────────────────────────────────────────────

    async def _aget_categories(
            self,
            knowledge_pack: str,
            level: str | None = None,
            value: str | None = None,
    ) -> list[str]:
        """Trả về danh sách category duy nhất khớp filter (dùng batch lớn vì payload nhẹ)."""
        scroll_filter = self._build_filter(knowledge_pack, level, value)
        points = await self.qdrant.ascroll_all(
            scroll_filter=scroll_filter, with_payload=_CATEGORY_PAYLOAD, limit=_CATEGORY_SCROLL_LIMIT,
        )
        return list({pt.payload.get("category") for pt in points if pt.payload.get("category")})

    # ── Chunk fetching ────────────────────────────────────────────────────────

    async def _afetch_category_points(
            self,
            category: str,
            knowledge_pack: str,
            level: str | None = None,
            value: str | None = None,
    ) -> list:
        """Fetch toàn bộ point của một category."""
        scroll_filter = self._build_filter(knowledge_pack, level, value, category=category)
        return await self.qdrant.ascroll_all(scroll_filter=scroll_filter)

    # ── Adjacent window selection ─────────────────────────────────────────────

    @staticmethod
    def _pick_adjacent(all_points: list, window: int) -> list[dict[str, Any]]:
        """Chọn `window` chunk liên tiếp từ danh sách point, trả về list[dict]."""
        if not all_points:
            return []

        # Nhóm theo file_name → sort theo chunk_index → tìm consecutive runs
        by_file: dict[str, list] = defaultdict(list)
        for pt in all_points:
            by_file[pt.payload.get("file_name", "")].append(pt)

        runs: list[list] = []
        for pts in by_file.values():
            pts.sort(key=lambda p: p.payload.get("chunk_index", 0))
            run: list = [pts[0]]
            for pt in pts[1:]:
                if pt.payload.get("chunk_index", -1) == run[-1].payload.get("chunk_index", -2) + 1:
                    run.append(pt)
                else:
                    runs.append(run)
                    run = [pt]
            runs.append(run)

        # Chọn window liên tiếp; fallback 1 chunk nếu không đủ
        valid = [r for r in runs if len(r) >= window]
        if valid:
            chosen = random.choice(valid)
            start = random.randint(0, len(chosen) - window)
            selected = chosen[start: start + window]
        else:
            flat = [pt for r in runs for pt in r]
            selected = [random.choice(flat)] if flat else []

        # Convert points → dicts
        return [
            {
                "id": pt.id,
                **{f: pt.payload.get(f) for f in _CHUNK_FIELDS},
                "text": pt.payload.get("text", ""),
            }
            for pt in selected
        ]

    # ── Public API ────────────────────────────────────────────────────────────

    @alog_method_call
    async def asample_and_fetch(
            self,
            knowledge_pack: str,
            level: str | None = None,
            value: str | None = None,
            n: int = 10,
            window: int = 2,
            seed: int | None = None,
    ) -> list[tuple[str, list[dict[str, Any]]]]:
        """
        Random n categories → fetch adjacent chunks song song.
        Cho phép lặp lại category (chọn window khác nhau) khi n > số category khả dụng.
        """
        all_categories = await self._aget_categories(knowledge_pack, level, value)
        if not all_categories:
            filter_desc = f"knowledge_pack='{knowledge_pack}'"
            if level and value:
                filter_desc += f", {level}='{value}'"
            raise ValueError(f"Không tìm thấy category nào với {filter_desc}.")

        if seed is not None:
            random.seed(seed)

        random.shuffle(all_categories)
        needs_repeats = n > len(all_categories)
        # Chỉ fetch đủ dùng: tất cả nếu cần lặp, hoặc n×3 (buffer cho empty categories)
        fetch_count = len(all_categories) if needs_repeats else min(len(all_categories), n * 3)
        cats_to_fetch = all_categories[:fetch_count]

        logger.info(
            f"  📦 Found {len(all_categories)} unique categories, "
            f"fetching {fetch_count}, need {n} samples"
        )

        # Fetch song song chỉ các category cần thiết
        fetched_lists = await asyncio.gather(
            *[self._afetch_category_points(cat, knowledge_pack, level, value)
              for cat in cats_to_fetch]
        )

        cat_points: dict[str, list] = {}
        for cat, points in zip(cats_to_fetch, fetched_lists):
            if points:
                cat_points[cat] = points
            else:
                logger.info(f"  ⏭️  {cat:<40s}  →  no chunks, skipping…")

        if not cat_points:
            raise ValueError("Không có category nào chứa chunk trong Qdrant.")

        usable = list(cat_points.keys())
        logger.info(f"  ✅ {len(usable)} categories with data available")

        result: list[tuple[str, list[dict[str, Any]]]] = []
        round_num = 0
        while len(result) < n:
            round_num += 1
            random.shuffle(usable)
            for cat in usable:
                if len(result) >= n:
                    break
                chunks = self._pick_adjacent(cat_points[cat], window=window)
                if not chunks:
                    continue
                result.append((cat, chunks))
                idxs = [c["chunk_index"] for c in chunks]
                logger.info(
                    f"  📌 [{round_num}] {cat:<40s}  →  {len(chunks)} chunk(s) "
                    f"(chunk_index {idxs})"
                )

        if len(result) < n:
            logger.info(f"\n  ⚠️  Chỉ tìm được {len(result)}/{n} mẫu.")

        return result
