from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ollama import Client as OllamaClient
from qdrant_client import QdrantClient

from app.auth import require_api_key
from app.config import get_settings
from app.db import check_database
from app.routes.chat import router as chat_router
from app.routes.conversations import router as conversations_router
from app.routes.documents import router as documents_router


app = FastAPI(
    title="LocalRAG API",
    version="0.2.0",
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
        "PATCH",
        "DELETE",
    ],
    allow_headers=[
        "X-API-Key",
        "Content-Type",
        "Idempotency-Key",
    ],
)


def _model_identifier(model: object) -> str:
    """Return a model name from an Ollama model-list entry."""
    if isinstance(model, dict):
        value = model.get("model") or model.get("name")
    else:
        value = getattr(model, "model", None) or getattr(
            model, "name", None
        )

    return value.strip() if isinstance(value, str) else ""


def _model_is_available(
    configured_model: str,
    available_models: set[str],
) -> bool:
    """Match a configured model, treating an omitted tag as :latest."""
    requested = configured_model.strip()

    if not requested:
        return False

    if requested in available_models:
        return True

    if ":" not in requested:
        return f"{requested}:latest" in available_models

    return False


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check; must not depend on external services."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Check the database, Qdrant, Ollama, and configured models."""
    settings = get_settings()

    dependencies = {
        "database": "ok" if check_database() else "unavailable",
        "qdrant": "unavailable",
        "ollama": "unavailable",
        "llm_model": "unavailable",
        "embedding_model": "unavailable",
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
        response = ollama.list()
        model_entries = getattr(response, "models", None)

        if model_entries is None:
            raise RuntimeError("Ollama returned no model list.")

        available_models = {
            name
            for name in (
                _model_identifier(model)
                for model in model_entries
            )
            if name
        }

        dependencies["ollama"] = "ok"
        dependencies["llm_model"] = (
            "ok"
            if _model_is_available(settings.llm_model, available_models)
            else "missing"
        )
        dependencies["embedding_model"] = (
            "ok"
            if _model_is_available(settings.embedding_model, available_models)
            else "missing"
        )
    except Exception:
        # Ollama is unavailable, so its models cannot be verified.
        pass

    if any(status != "ok" for status in dependencies.values()):
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "dependencies": dependencies,
            },
        )

    return {"status": "ready", "dependencies": dependencies}


app.include_router(documents_router, dependencies=[Depends(require_api_key)])
app.include_router(chat_router, dependencies=[Depends(require_api_key)])
app.include_router(conversations_router, dependencies=[Depends(require_api_key)])
