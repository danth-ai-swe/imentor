import asyncio
import random
from collections import defaultdict
from typing import Any

import pandas as pd
from qdrant_client import models

from src.constants.app_constant import METADATA_NODE_XLSX
from src.rag.db_vector import QdrantManager, get_qdrant_client
from src.utils.logger_utils import alog_method_call, logger

# ── Constants ─────────────────────────────────────────────────────────────────
_SCROLL_LIMIT = 100
_SCROLL_PAYLOAD_CATEGORY = ["category"]
_CHUNK_FIELDS = (
    "id", "chunk_index", "file_name", "node_name", "node_id",
    "category", "page_number", "total_pages", "module", "lesson", "course", "text",
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
        """Build Qdrant filter. course là bắt buộc; level/value/category là tuỳ chọn."""
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

    # ── Generic scroll (DRY core) ─────────────────────────────────────────────

    async def _ascroll(
        self,
        scroll_filter: models.Filter,
        with_payload: list[str] | bool = True,
    ) -> list:
        """Scroll toàn bộ Qdrant với filter cho trước, trả về list point."""
        all_points: list = []
        offset = None

        while True:
            scroll_kwargs: dict[str, Any] = dict(
                collection_name=self.qdrant.collection_name,
                scroll_filter=scroll_filter,
                limit=_SCROLL_LIMIT,
                with_payload=with_payload,
                with_vectors=False,
            )
            if offset is not None:
                scroll_kwargs["offset"] = offset

            points, next_offset = await self.qdrant.async_client.scroll(**scroll_kwargs)
            all_points.extend(points)

            if next_offset is None or not points:
                break
            offset = next_offset

        return all_points

    # ── Category discovery ────────────────────────────────────────────────────

    async def _aget_categories(
        self,
        knowledge_pack: str,
        level: str | None = None,
        value: str | None = None,
    ) -> list[str]:
        """Trả về danh sách category duy nhất khớp filter."""
        scroll_filter = self._build_filter(knowledge_pack, level, value)
        points = await self._ascroll(scroll_filter, with_payload=_SCROLL_PAYLOAD_CATEGORY)
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
        return await self._ascroll(scroll_filter)

    # ── Adjacent window selection ─────────────────────────────────────────────

    @staticmethod
    def _find_consecutive_runs(sorted_points: list) -> list[list]:
        """Tìm tất cả chuỗi chunk_index liên tiếp trong danh sách đã sort."""
        if not sorted_points:
            return []
        runs: list[list] = [[sorted_points[0]]]
        for pt in sorted_points[1:]:
            prev_idx = runs[-1][-1].payload.get("chunk_index", -999)
            curr_idx = pt.payload.get("chunk_index", -999)
            if curr_idx == prev_idx + 1:
                runs[-1].append(pt)
            else:
                runs.append([pt])
        return runs

    @staticmethod
    def _group_runs_by_file(points: list) -> list[list]:
        """Nhóm point theo file_name rồi tìm consecutive runs."""
        by_file: dict[str, list] = defaultdict(list)
        for pt in points:
            by_file[pt.payload.get("file_name", "")].append(pt)

        all_runs: list[list] = []
        for file_points in by_file.values():
            file_points.sort(key=lambda p: p.payload.get("chunk_index", 0))
            all_runs.extend(NodeSampler._find_consecutive_runs(file_points))
        return all_runs

    @staticmethod
    def _select_window(runs: list[list], window: int) -> list:
        """Chọn window chunk liên tiếp từ các run. Fallback sang 1 chunk nếu không đủ."""
        valid_runs = [r for r in runs if len(r) >= window]
        if valid_runs:
            chosen = random.choice(valid_runs)
            start = random.randint(0, len(chosen) - window)
            return chosen[start: start + window]
        # Fallback: bất kỳ chunk nào
        flat = [pt for run in runs for pt in run]
        return [random.choice(flat)] if flat else []

    @staticmethod
    def _point_to_dict(pt) -> dict[str, Any]:
        payload = pt.payload
        return {
            "id": pt.id,
            **{field: payload.get(field) for field in _CHUNK_FIELDS if field != "id"},
            "text": payload.get("text", ""),
        }

    def _pick_adjacent(self, all_points: list, window: int) -> list[dict[str, Any]]:
        """Chọn `window` chunk liên tiếp từ danh sách point."""
        if not all_points:
            return []
        runs = self._group_runs_by_file(all_points)
        selected = self._select_window(runs, window)
        return [self._point_to_dict(pt) for pt in selected]

    # ── Public API ────────────────────────────────────────────────────────────

    @alog_method_call
    async def asample_and_fetch(
        self,
        knowledge_pack: str = "LOMA281",
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
            filter_desc = f"course='{knowledge_pack}'"
            if level and value:
                filter_desc += f", {level}='{value}'"
            raise ValueError(f"Không tìm thấy category nào với {filter_desc}.")

        if seed is not None:
            random.seed(seed)

        pool = random.sample(all_categories, len(all_categories))  # shuffle copy
        logger.info(
            f"  📦 Found {len(pool)} unique categories, need {n} samples "
            f"(repeats allowed: {n > len(pool)})"
        )

        # Fetch tất cả points song song (mỗi category chỉ fetch 1 lần)
        fetched_lists = await asyncio.gather(
            *[self._afetch_category_points(cat, knowledge_pack, level, value) for cat in pool]
        )

        cat_points: dict[str, list] = {}
        for cat, points in zip(pool, fetched_lists):
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