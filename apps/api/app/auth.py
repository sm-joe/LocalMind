"""Single-instance API credential gate.

The web server holds the API secret. Never embed it in browser JavaScript.
Local development can run without a key; Compose explicitly requires one.
"""
import secrets

from fastapi import Header, HTTPException

from app.config import get_settings


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    configured = get_settings().localrag_api_key
    if not configured:
        # Fail closed when the instance is configured as production.
        if get_settings().app_env.lower() == "production":
            raise HTTPException(status_code=503, detail="API authentication is not configured.")
        return

    if not x_api_key or not secrets.compare_digest(x_api_key, configured):
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
