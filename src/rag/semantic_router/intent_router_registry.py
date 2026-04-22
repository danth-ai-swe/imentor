from typing import Optional

from src.rag.semantic_router.router import SemanticRouter

_intent_router: Optional[SemanticRouter] = None


def get_intent_router() -> SemanticRouter:
    if _intent_router is None:
        raise RuntimeError("Intent router chưa được khởi tạo. Gọi init_intent_router() trong lifespan.")
    return _intent_router


def set_intent_router(router: SemanticRouter) -> None:
    global _intent_router
    _intent_router = router
