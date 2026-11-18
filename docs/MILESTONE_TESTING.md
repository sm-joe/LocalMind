# LocalMind — Bash Acceptance And Milestone Testing

This runbook validates a **live Docker Compose installation**, not merely GitHub Actions. Commands use Bash (Linux, macOS or WSL), GNU-compatible curl and Python 3. Run from your repository root.

> Never paste passwords or API keys into shell history, chat transcripts, CI logs or issue comments. Never use docker compose down -v unless intentionally deleting persistent data.

## Prepare The Environment

~~~bash
git status --short
cp -n .env.example .env
python3 -c 'import secrets; print(secrets.token_urlsafe(48))'
~~~

Set distinct secrets in .env: POSTGRES_PASSWORD, LOCALRAG_API_KEY (32+ characters), LOCALRAG_ADMIN_PASSWORD (12+ characters), and LOCALRAG_SESSION_SECRET (32+ characters). The LOCALRAG_* keys remain for runtime backwards compatibility after branding becomes LocalMind.

~~~bash
docker compose config --quiet
docker compose up -d --build
docker compose ps
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/ready
~~~

Expected: healthy services, status ok, and readiness ready once Ollama provisioning is complete. First-start readiness can return HTTP 503 while models are downloading.

## Authentication And API Access

Read the API key and login password without printing them to the terminal:

~~~bash
read -r -s -p 'API Key: ' API_KEY; printf '\n'
read -r -s -p 'Administrator Password: ' ADMIN_PASSWORD; printf '\n'
~~~

Unauthenticated calls must be rejected:

~~~bash
curl -sS -o /dev/null -w 'Unauthenticated HTTP %{http_code}\n' http://127.0.0.1:8000/documents
curl -fsS -H "X-API-Key: $API_KEY" http://127.0.0.1:8000/documents
~~~

Expected: first response HTTP 401; second response returns the document collection.

Verify the signed web session with curl's cookie jar:

~~~bash
cookie_jar=$(mktemp)
ADMIN_PASSWORD="$ADMIN_PASSWORD" python3 -c 'import json,os; print(json.dumps({"password":os.environ["ADMIN_PASSWORD"]}))' | \
curl -fsS -c "$cookie_jar" -H 'Origin: http://localhost:3000' \
  -H 'Content-Type: application/json' --data-binary @- \
  http://localhost:3000/api/auth/login
curl -fsS -b "$cookie_jar" http://localhost:3000/api/auth/session
curl -fsS -b "$cookie_jar" http://localhost:3000/api/backend/documents
rm -f "$cookie_jar"
~~~

Expected: authenticated true, an HttpOnly session cookie, and accessible proxied documents. The application is **single-user**, not tenant-isolated.

## Document Upload, Deduplication And Indexing

Create a harmless fixture and upload it:

~~~bash
printf 'LocalMind Tests Document Retrieval, Citations, And Duplicate Indexing.\n' > milestone-test.txt
first=$(curl -fsS -H "X-API-Key: $API_KEY" \
  -F 'file=@milestone-test.txt;type=text/plain' \
  http://127.0.0.1:8000/documents/upload)
printf '%s\n' "$first" | python3 -m json.tool
doc_id=$(printf '%s' "$first" | python3 -c 'import json,sys; print(json.load(sys.stdin)["document_id"])')
curl -fsS -H "X-API-Key: $API_KEY" \
  -F 'file=@milestone-test.txt;type=text/plain' \
  http://127.0.0.1:8000/documents/upload | python3 -m json.tool
curl -fsS -H "X-API-Key: $API_KEY" \
  "http://127.0.0.1:8000/documents/status/$doc_id" | python3 -m json.tool
~~~

Expected: initial upload indexed, duplicate already_indexed, and a ready document record with a nonzero chunk count. A parallel conflicting upload may return HTTP 409. The server does not persist every uploaded file as a reusable original.

Delete only the test fixture:

~~~bash
curl -fsS -X DELETE -H "X-API-Key: $API_KEY" \
  "http://127.0.0.1:8000/documents/$doc_id"
rm -f milestone-test.txt
~~~

## Retrieval, Citations And Hallucination Acceptance

~~~bash
docker compose exec api pytest -q tests/test_retrieval.py tests/test_lexical_reranking.py tests/test_comparison_qa.py tests/test_named_entity_qa.py tests/test_rag_streaming.py
~~~

Upload your actual document first, then audit stored mentions without mutating Qdrant:

~~~bash
docker compose exec api python -m app.tools.index_audit Interview_Questions.docx EC2 IAM Bedrock
~~~

If the indexed document has EC2 and IAM passages but no Bedrock mention, the expected behavior is:

1. **Compare EC2 And IAM:** retrieve relevant evidence for both subjects, preferably definition passages; do not invent unsupported differences.
2. **What Is Amazon Bedrock Mentioned In The Document?:** explicitly say there is no indexed evidence; do not answer from pretrained knowledge or make up citations.
3. **Summarize The Document:** synthesize the document's major sections. If inline summary citations are absent, don't claim that individual statements have been verified.
4. **Inspect Sources:** use the UI's Read Original Passage action to verify retrieved text and citation mapping.

These acceptance criteria require real Ollama and the actual indexed data. Passing mock-based tests is not sufficient proof.

## Research UI And Source Inspection

1. Visit http://localhost:3000 in a private browser window; check the **LocalMind** sign-in page.
2. Sign in, confirm **Documents** loads as the default dashboard and files/search/filters are visible.
3. Choose **Ask Questions** for an indexed document. Confirm the **Research** page opens, with the selected source and an independent source inspector.
4. Send a question and check streaming, final answer, source citations and **Read Original Passage**.
5. Use **Conversations** to test resume, rename, archive, restore and delete.
6. Test the layout at narrow desktop and mobile browser sizes; review overflow, keyboard focus and accessibility.

API source preview example:

~~~bash
read -r -p 'Indexed Document ID: ' DOC_ID
curl -fsS -H "X-API-Key: $API_KEY" \
  "http://127.0.0.1:8000/documents/$DOC_ID/chunks/0"
docker compose exec api pytest -q tests/test_source_preview.py
~~~

## Persistence And Backup

Pause uploads and chat writes while backing up:

~~~bash
python3 scripts/backup_localmind.py --output backups
latest_backup=$(find backups -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
test -n "$latest_backup" && python3 scripts/backup_localmind.py --verify "$latest_backup"
python3 -m unittest discover -s scripts/tests -p 'test_*.py'
python3 -m unittest discover -s evals/tests -p 'test_*.py'
docker compose exec api pytest -q
~~~

The LocalMind command uses a backward-compatible implementation that retains legacy storage identifiers. Encrypt backup output and validate a real restore. Do not mistake a SHA-256 integrity check for a disaster-recovery acceptance test.

## CI Versus Runtime Proof

GitHub Actions verifies source checks and build jobs on eligible branches; some integration, image and security jobs may be skipped. Manual verification of full-stack Docker operation, live Ollama quality, and recovery remains necessary.

~~~bash
docker compose ps
docker compose logs --tail=120 api web ollama postgres qdrant
~~~

After the GitHub rename, the workflow will appear under **Actions → LocalMind CI Pipeline**. Update your Git remote rather than relying indefinitely on GitHub redirect behavior:

~~~bash
git remote set-url origin https://github.com/sm-joe/localmind.git
git remote -v
~~~

## Cleanup

~~~bash
unset API_KEY ADMIN_PASSWORD DOC_ID
~~~

**Do not use docker compose down -v while retaining saved documents or conversations.**
