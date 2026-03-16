from datetime import datetime

import uvicorn
from fastapi import FastAPI

from src.config import get_app_config

app = FastAPI()
config = get_app_config()


@app.get("/health", tags=["Health Check"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "git_commit_id": config.GIT_COMMIT_ID,
        "definition_name": config.DEFINITION_NAME,
        "app_build_number": config.APP_BUILD_NUMBER
    }


if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)
