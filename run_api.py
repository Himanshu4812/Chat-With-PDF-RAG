"""Run the FastAPI server."""
import uvicorn

from src.api.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8001,
        reload=settings.RELOAD,
        log_level="info"
    )
