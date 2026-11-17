# LocalRAG milestone verification — Windows PowerShell

These commands verify a **running local Docker installation**, not just CI. Run them
from the repository root. The features below are on stacked PR branches until
merged into `main`. Before running, use the branch containing the feature under
test (the latest branch includes its ancestors).

## Prerequisites and startup

```powershell
git status
Copy-Item .env.example .env   # Only if .env does not already exist
notepad .env
```

Set **distinct, private** `POSTGRES_PASSWORD`, `LOCALRAG_API_KEY` (32+ characters),
`LOCALRAG_ADMIN_PASSWORD` (12+ characters), and
`LOCALRAG_SESSION_SECRET` (32+ characters) in `.env`. Never commit this file.
To generate high-entropy secrets on Windows:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Repeat for the API key, admin password, and session secret. Never paste a
production secret into chat, CI logs, or a GitHub issue. Check/start services:

```powershell
docker compose config --quiet
docker compose up -d --build
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
```

**Expected:** Compose config is valid; services become healthy; health status is
`ok` and readiness status is `ready` once Ollama models are installed. The
initial model download can require an internet connection. This build is **local
and explicit**; PR CI deliberately skips its separate Docker image build job.

For API tests: enter a private key at the prompt (do not print it):

```powershell
$apiKey = Read-Host "Enter LOCALRAG_API_KEY from .env"
$headers = @{ "X-API-Key" = $apiKey }
```

## Milestone 1 — Authentication and API access control (PR #15)

```powershell
# Unauthenticated business endpoints should reject the request.
try {
    Invoke-WebRequest http://127.0.0.1:8000/documents -ErrorAction Stop
    throw "FAIL: missing API key was accepted"
} catch {
    if ($_.Exception.Response.StatusCode.value__ -ne 401) { throw }
    "PASS: unauthenticated document request rejected with 401"
}

# Authenticated requests should work.
Invoke-RestMethod http://127.0.0.1:8000/documents -Headers $headers

# Same-origin web login establishes an HttpOnly session.
$adminPassword = Read-Host "Enter LOCALRAG_ADMIN_PASSWORD from .env"
$login = Invoke-RestMethod -Uri http://127.0.0.1:3000/api/auth/login `
    -Method Post -ContentType "application/json" `
    -Headers @{ Origin = "http://127.0.0.1:3000" } `
    -Body (@{ password = $adminPassword } | ConvertTo-Json) `
    -SessionVariable webSession
$login
Invoke-RestMethod http://127.0.0.1:3000/api/auth/session -WebSession $webSession
Invoke-RestMethod http://127.0.0.1:3000/api/backend/documents -WebSession $webSession
```

**Expected:** unauthenticated API request returns 401; authenticated API returns
`documents`; login/session responses report `authenticated: true`; proxied
document listing works. Manually visit http://localhost:3000: the application
must show sign-in before the chatbot workspace. In a fresh private/incognito
window without the session, the API proxy must reject access.

**Important:** Current auth protects one local instance. It is **not** per-user
or per-document multi-tenant authorization. Do not expose publicly without
HTTPS, brute-force/rate-limiting defenses, and additional hardening.

## Milestone 2 — Document ingestion states and recovery (PR #18)

Create a safe temporary test file and upload twice:

```powershell
"LocalRAG verifies resumable retry states, citations, and document deduplication." |
    Set-Content -Encoding utf8 .\milestone-test.txt
$first = curl.exe -sS -H "X-API-Key: $apiKey" `
    -F "file=@milestone-test.txt;type=text/plain" `
    http://127.0.0.1:8000/documents/upload | ConvertFrom-Json
$first
$duplicate = curl.exe -sS -H "X-API-Key: $apiKey" `
    -F "file=@milestone-test.txt;type=text/plain" `
    http://127.0.0.1:8000/documents/upload | ConvertFrom-Json
$duplicate
Invoke-RestMethod "http://127.0.0.1:8000/documents/status/$($first.document_id)" `
    -Headers $headers
```

**Expected:** initial upload returns `indexed`, duplicate returns
`already_indexed` with the same `document_id`, and status returns `ready`
with a positive `chunks` count. For active concurrent uploads of identical
bytes, the second request should receive HTTP 409. A failed ingest can be
retried by uploading the file again. This does **not** imply persisted upload
bytes or automatic background resumption.

To verify deletion (only the new test file):

```powershell
Invoke-RestMethod -Method Delete `
    "http://127.0.0.1:8000/documents/$($first.document_id)" -Headers $headers
Remove-Item .\milestone-test.txt
```

**Expected:** response status `deleted`; the test document disappears from
the listing and its registry record is cleared. Do not delete real documents.

## Milestone 3 — Retrieval relevance (PR #20)

```powershell
docker compose exec api pytest -q tests/test_lexical_reranking.py `
    tests/test_retrieval.py tests/test_rag.py
```

**Expected:** tests pass, proving lexical reranking can improve the order of
dense-search candidates and original Qdrant similarity scores are preserved.
For live quality, upload the fixture and run the authenticated evaluation:

```powershell
$env:LOCALRAG_API_KEY = $apiKey
$env:LOCALRAG_EVAL_API_URL = "http://127.0.0.1:8000/chat"
python .\evals\run_evaluation.py
Remove-Item Env:LOCALRAG_API_KEY
```

**Expected:** per-case retrieval/concept/citation scores and an evaluation
JSON output. Evaluation can fail until the expected fixture document has been
uploaded. This is **lexical-aware reranking**, not native sparse+dense hybrid
search or a tokenizer-exact context budget.

## Milestone 4 — Chat UX and source previews (PR #16)

After uploading a document, use the signed-in web UI at
http://localhost:3000 and ask a question grounded in it. Click
**Read source passage** on the right evidence panel.

To verify directly against an indexed chunk:

```powershell
# Set $docId to a document_id returned by /documents/upload.
$docId = Read-Host "Enter indexed document_id"
Invoke-RestMethod `
    "http://127.0.0.1:8000/documents/$docId/chunks/0" `
    -Headers $headers
docker compose exec api pytest -q tests/test_source_preview.py
```

**Expected:** authenticated API returns the indexed `text`, `filename`,
`chunk_id`, `start_char`, `end_char` for the specified source; the UI
displays the passage without interpreting it as HTML. Nonexistent chunks return
404. Streaming token responses are **not yet implemented**.

## Milestone 5 — Backup integrity, evaluation and release checks (PR #21)

Pause document uploads and chat writes for a consistent backup.

```powershell
python .\scripts\backup_localrag.py --output .\backups
$latest = Get-ChildItem .\backups -Directory |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
python .\scripts\backup_localrag.py --verify $latest.FullName
python -m unittest discover -s scripts/tests -p "test_*.py"
python -m unittest discover -s evals/tests -p "test_*.py"
docker compose exec api pytest -q
docker compose ps
```

**Expected:** backup contains `postgres.dump`, `qdrant.snapshot`,
and `manifest.json`; integrity verification and deterministic tests pass;
API tests pass and services remain healthy. The backup directory contains
**private** documents and conversations — encrypt it before transferring
off-machine. Backups are not cross-database atomic, and the **restore drill
has not been verified**.

## Run the heavyweight image-security gate only when needed

Pull requests skip the separate Docker Build job. To run it on demand:
GitHub repository **Actions → LocalRAG CI Pipeline → Run workflow**. The same
image build and Trivy HIGH/CRITICAL checks remain active for relevant pushes
to `main`. A passing PR is **not** proof of a passing container vulnerability
scan or live end-to-end behavior; run those before merging a release candidate.

When testing is complete, do not destroy your persisted data with
`docker compose down -v`. For a normal shutdown use:

```powershell
docker compose down
```
