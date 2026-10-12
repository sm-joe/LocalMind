# LocalRAG Phase A/B — Apply and Validate

This patch adds PostgreSQL-backed conversation persistence and conversation APIs.
It does not modify the frontend; Phase C will connect the UI to these APIs.

## Apply

Extract this ZIP into the repository root, preserving the directory structure.
It intentionally does not include `.env`, runtime data, `.git`, `node_modules`, or `.next`.

## Configure the database password

From PowerShell at the repository root:

```powershell
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
}
```

Open the repository-root `.env` and set `POSTGRES_PASSWORD` to a strong random value.
If `.env` already exists, preserve its other settings. Do not commit `.env`.

## Build and start

```powershell
docker compose config
docker compose build api
docker compose up -d --build
docker compose ps
```

The API container runs `alembic upgrade head` before starting Uvicorn. If migration fails, the API does not start.

## Verify persistence and API

```powershell
docker compose logs --no-color postgres
docker compose logs --no-color api
docker compose exec api alembic current
docker compose exec api pytest -q
Invoke-RestMethod http://localhost:8000/ready
```

Expected: the migration revision is `0001_conversation_history`, and `/ready` reports `database`, `qdrant`, `ollama`, `llm_model`, and `embedding_model` as `ok`.

## API smoke test (PowerShell)

```powershell
$conversation = Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/conversations `
  -ContentType 'application/json' `
  -Body '{"title":"Persistence smoke test"}'

$conversationId = $conversation.id
$conversationId
```

A persisted chat request must include that `conversation_id`. For example, using a stable idempotency key:

```powershell
$body = @{
    question = "Summarize the uploaded documents"
    top_k = 5
    conversation_id = $conversationId
} | ConvertTo-Json

$headers = @{ "Idempotency-Key" = "smoke-test-0001" }

Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/chat `
  -ContentType 'application/json' `
  -Headers $headers `
  -Body $body

Invoke-RestMethod `
  -Uri "http://localhost:8000/conversations/$conversationId/messages"
```

The answer may legitimately say that there is insufficient evidence if no relevant documents are indexed.

## Important data-safety notes

- `docker compose down` preserves `postgres_data`.
- **Do not use `docker compose down -v`** if you want to keep conversation history.
- The database is not exposed on a host port; it is reachable only on the internal backend network.
- The current application remains single-user/local. The `owner_id` field is groundwork, not authentication. Do not expose it to untrusted networks until authentication and authorization are implemented.
- The existing frontend still calls `/chat` without a `conversation_id`; chat history will become visible in the UI in Phase C.
- The uploaded source archive contained `apps/api/.env`. This patch excludes all `.env` files. If that file contained credentials, rotate them and ensure `.env` files are not committed.
