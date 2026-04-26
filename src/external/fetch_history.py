import httpx

from src.config.app_config import get_app_config
from src.constants.app_constant import CHAT_HISTORY_TIMEOUT
from src.utils.logger_utils import alog_function_call


@alog_function_call
async def fetch_raw_chat_history(
        conversation_id: str,
) -> list[dict] | None:
    config = get_app_config()
    url = f"{config.CHAT_HISTORY_API_BASE}/api/conversations/{conversation_id}/history-for-ai"
    params = {"limit": 20}

    try:
        async with httpx.AsyncClient(timeout=CHAT_HISTORY_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            body = resp.json()
    except httpx.HTTPError:
        return None

    if not body.get("success"):
        return None

    return body.get("data", {}).get("messages", [])
