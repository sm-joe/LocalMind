# LocalMind — Historical Conversation Persistence Notes

This file is retained as a **historical note** for the original PostgreSQL conversation migration work. Its original PowerShell setup instructions and older unauthenticated API samples are obsolete.

The current LocalMind application has:

- A password-protected Next.js frontend and signed HttpOnly sessions.
- FastAPI endpoints requiring an API key.
- PostgreSQL migrations applied before Uvicorn starts.
- Conversation creation, retrieval, rename, archive, restore and deletion.
- Document-first dashboard and research workspace.

For current installation and acceptance procedures, see [README.md](README.md) and [docs/MILESTONE_TESTING.md](docs/MILESTONE_TESTING.md).

## Bash Setup And Verification

From the repository root:

~~~bash
cp -n .env.example .env
# Edit .env and provide unique, strong values for all required secrets.
docker compose config --quiet
docker compose up -d --build
docker compose ps
docker compose exec api alembic current
docker compose exec api pytest -q tests/test_conversations.py
curl -fsS http://127.0.0.1:8000/ready
~~~

The runtime uses legacy LOCALRAG_* configuration keys for backward compatibility; the product branding is **LocalMind**. Use the administrator login in the browser at http://localhost:3000. Do not call protected API endpoints without their X-API-Key header.

**Data safety:** Do not use docker compose down -v on an installation whose conversations, documents, and models need to be retained. Do not rename an existing Compose project directory without first verifying its persistent volume mappings.

This historical reference is not a complete release or migration guide.
