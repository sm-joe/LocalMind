# LocalMind

**A private, document-first research workspace.** Upload documents, search your indexed library, ask questions, and inspect source passages behind answers using local models.

LocalMind is a self-hosted Next.js + React, FastAPI, PostgreSQL, Qdrant, Ollama and Docker Compose project. It is designed for a local single-user installation, not hosted multi-tenant SaaS.

## Features

| Screen | Capabilities |
| --- | --- |
| **Documents** | Document-first homepage, drag-and-drop uploads, file-type filters, searchable file table, chunk counts, deletion and one-click Ask Questions |
| **Research** | Document scope selector, streaming answers, summaries, comparisons, cited passage inspector and copy tools |
| **Conversations** | Saved threads, search, resume, rename, archive, restore and delete |
| **Sign In** | Administrator password, signed HttpOnly session and server-side API key |
| **Local Data** | PostgreSQL history, Qdrant searchable chunks and Ollama inference/embeddings |

**Evidence policy:** LocalMind should not invent source-dependent answers when the indexed document lacks the requested information. Summaries without verified inline citations disclose that limitation. Tests alone do not guarantee factual grounding or live LLM correctness.

## Quick Start — Bash

Prerequisites: Git, Docker Engine/Desktop with Compose v2, and sufficient RAM and disk for your chosen Ollama models. Run these commands in **Bash** on Linux, macOS or WSL.

### 1. Clone

After the GitHub repository is renamed:

~~~bash
git clone https://github.com/sm-joe/localmind.git
cd localmind
~~~

### 2. Configure

~~~bash
cp -n .env.example .env
python3 -c 'import secrets; print(secrets.token_urlsafe(48))'
~~~

Set **distinct** private values in .env for:

| Variable | Purpose |
| --- | --- |
| POSTGRES_PASSWORD | Strong PostgreSQL credential |
| LOCALMIND_API_KEY | Backend API key, at least 32 characters |
| LOCALMIND_ADMIN_PASSWORD | Browser sign-in password, at least 12 characters |
| LOCALMIND_SESSION_SECRET | Signing secret, at least 32 characters |

Generate independent values for each secret. Never commit .env or share secrets in logs or chat.

**Backward compatibility:** The application and repository branding are **LocalMind**, but existing LOCALRAG_* environment keys, container names, database/collection names and volume identifiers are intentionally preserved. Changing those identifiers without a controlled migration would disrupt existing installations. They are internal runtime names, not user-facing branding.

### 3. Build And Start

~~~bash
docker compose config --quiet
docker compose up -d --build
docker compose ps
~~~

The one-shot ollama-init service provisions configured models during initial startup. Models are stored in a persistent Docker volume.

### 4. Sign In And Check Readiness

Open **http://localhost:3000** and use the LOCALMIND_ADMIN_PASSWORD configured in .env.

~~~bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/ready
docker compose exec ollama ollama list
~~~

A health response confirms liveness; readiness checks the database, Qdrant, Ollama and models. Readiness can return HTTP 503 while models are provisioning. The protected API at http://127.0.0.1:8000 requires an X-API-Key header; browsers use the authenticated Next.js proxy.

## How To Use LocalMind

1. **Documents** is the default workspace page. Upload a PDF, DOCX, TXT or MD file (10 MiB max), then search or filter the library.
2. Choose **Ask Questions** on a document to open document-oriented research; alternatively open **Research** and select **All Documents**.
3. Ask for summaries, comparisons or facts. Use **Read Original Passage** to inspect retrieved chunks; do not treat a plausible answer as independently verified.
4. Visit **Conversations** to resume threads or rename, archive, restore and delete them.
5. To refresh extraction after a parser fix, re-upload a file and explicitly accept the reindex prompt. Old vectors remain until replacement indexing succeeds.

The current UI has a browser-based document-management dashboard, an independent research view and a separate conversation-history page. The sign-in screen uses a split layout. Old screenshot mockups in the repo are not screenshots of this release.

## Architecture

~~~text
Browser / localhost:3000
       |
       v
Next.js Web UI -- signed session and API proxy
       |
       v
FastAPI / localhost:8000
       |                   |
       +--- PostgreSQL     +--- Qdrant
       |    conversations       document chunks
       |
       +--- Ollama local LLM and embeddings
~~~

PostgreSQL stores conversation and document-index metadata; Qdrant stores chunks and vectors. Retrieved content is treated as untrusted evidence, not as executable instructions.

## Bash Operator Commands

~~~bash
# Start, stop, and inspect services
docker compose up -d
docker compose ps
docker compose logs --tail=100 api web ollama
docker compose down

# Rebuild an individual app (preserves volumes)
docker compose up -d --build --no-deps web
docker compose up -d --build --no-deps api

# Verify backend tests
docker compose exec api pytest -q

# Inspect the indexed chunks of an existing document (read-only)
docker compose exec api python -m app.tools.index_audit Interview_Questions.docx EC2 IAM Bedrock

# Check frontend locally with Node.js installed
cd apps/web
npm ci
npm run lint
npm run typecheck
npm run build
cd ../..
~~~

**Never run docker compose down -v on an installation whose data you want to keep.** It removes persistent Compose volumes.

To download an additional Ollama model into the existing persistent model volume, use the Bash helper. It temporarily grants outbound network access, then disconnects the runtime Ollama container:

~~~bash
bash scripts/pull-ollama-model.sh qwen3:8b
~~~

Only run this after the stack is healthy. This does not change the active LLM_MODEL unless you update .env and restart the relevant services.

## Backup And Evaluation

Pause document uploads and chat writes before taking backups.

~~~bash
python3 scripts/backup_localmind.py --output backups
python3 scripts/backup_localmind.py --verify backups/<backup-folder>
~~~

The LocalMind backup command delegates to the existing, backward-compatible backup implementation. Backups contain private data. Encrypt them and verify a real restore separately; a passing hash check is not a full disaster-recovery test.

After uploading the evaluation fixture, run a real model-based evaluation:

~~~bash
export LOCALMIND_API_KEY='your-private-api-key'
export LOCALMIND_EVAL_API_URL='http://127.0.0.1:8000/chat'
python3 evals/run_evaluation.py
unset LOCALMIND_API_KEY LOCALMIND_EVAL_API_URL
~~~

Unit tests, CI checks and model-based acceptance are separate quality gates.

## GitHub Repository Rename And Existing Installs

A repository administrator must rename **sm-joe/localrag** to **sm-joe/localmind** in **GitHub → Repository Settings → General → Repository Name → Rename**. Once GitHub confirms the rename, update remotes:

~~~bash
git remote set-url origin https://github.com/sm-joe/localmind.git
git remote -v
~~~

LocalMind pins Docker Compose's default project identity to the historical `localrag` name, so moving a default installation to a directory called `LocalMind` does **not** silently create new Qdrant and Ollama volumes. Existing database and container identifiers are retained for compatibility. If your original installation used a custom `COMPOSE_PROJECT_NAME`, continue setting that exact value whenever you run Compose. Verify mounted volumes and backups before changing any paths; do not delete existing volumes.

~~~bash
# Example: reuse an established localrag checkout
cd localrag
git remote set-url origin https://github.com/sm-joe/localmind.git
docker compose ps
~~~

The existing legacy secret variables, API keys and database credentials remain valid; do not regenerate them solely for rebranding.

## Runtime Configuration

| Setting | Default / Notes |
| --- | --- |
| LLM_MODEL | llama3.2:3b |
| EMBEDDING_MODEL | nomic-embed-text when unset in Compose |
| OLLAMA_NUM_CTX | 6144 |
| QDRANT_HOST / QDRANT_PORT | qdrant / 6333 |
| OLLAMA_BASE_URL | http://ollama:11434 |
| LOCALMIND_API_KEY | Required server-side API key |
| LOCALMIND_ADMIN_PASSWORD | Required single-user login |
| LOCALMIND_SESSION_SECRET | Signed session secret |

Uploaded files are limited to 10 MiB. Additional parser limits include 200 PDF pages, 10,000 DOCX paragraphs and 2 million extracted characters.

## Security And Known Limits

LocalMind is a local, password-protected single-user app, **not** multi-user authorization. Ports are bound to loopback and infrastructure services are private. Do not expose the app to untrusted networks without HTTPS, rate limiting and further review. Prompt-injection and hallucination risks cannot be eliminated by prompting alone; inspect the original evidence for consequential answers.

GitHub Actions run staged tests and security checks; a successful unit test or frontend build is not proof of a working full-stack Docker deployment. Read [SECURITY.md](SECURITY.md) and [the milestone acceptance guide](docs/MILESTONE_TESTING.md).

## Repository Layout

~~~text
localmind/
├── apps/
│   ├── web/              # Next.js UI
│   └── api/              # FastAPI and tests
├── docs/
├── evals/
├── scripts/
├── ollama/
├── docker-compose.yml
├── .env.example
├── README.md
└── SECURITY.md
~~~

**LocalMind — Your Documents, Your Research, Your Evidence.**
