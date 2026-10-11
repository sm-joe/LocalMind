from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ollama import Client as OllamaClient
from qdrant_client import QdrantClient

from app.config import get_settings
from app.routes.chat import router as chat_router
from app.routes.documents import router as documents_router


app = FastAPI(
    title="LocalRAG API",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
        "DELETE",
    ],
    allow_headers=[
        "Content-Type",
    ],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check; must not depend on external services."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness check for Qdrant and Ollama."""
    settings = get_settings()

    dependencies = {
        "qdrant": "unavailable",
        "ollama": "unavailable",
    }

    try:
        qdrant = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        qdrant.get_collections()
        dependencies["qdrant"] = "ok"
    except Exception:
        pass

    try:
        ollama = OllamaClient(
            host=settings.ollama_base_url,
        )
        ollama.list()
        dependencies["ollama"] = "ok"
    except Exception:
        pass

    if any(
        status != "ok"
        for status in dependencies.values()
    ):
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "dependencies": dependencies,
            },
        )

    return {
        "status": "ready",
        "dependencies": dependencies,
    }


app.include_router(documents_router)
app.include_router(chat_router)