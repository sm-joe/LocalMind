# LocalRAG Evaluation

The evaluation dataset is self-contained in `datasets/localrag_baseline.json`. Its `documents` section stores benchmark content directly; there is no required standalone fixture file and no manually uploaded filename.

## Safety model

- Run the evaluation against the dedicated evaluation Compose stack, not the normal user stack.
- The stack uses a separate Qdrant collection (`localrag_evaluation_documents`), a separate PostgreSQL volume, distinct container names, and API port `18000`.
- The runner assigns a unique filename and content marker to each benchmark document, then records the runtime `document_id` returned by the API.
- Source assertions use those runtime IDs mapped to logical dataset IDs. Filenames are display metadata, not correctness criteria.
- The runner deletes only document IDs returned by uploads from its current run. It never clears a collection or deletes documents by filename.
- If cleanup fails, the run fails and records the affected document IDs in its JSON report.

## Run locally (PowerShell)

Run these commands from the repository root. Your normal root `.env` must contain the required `POSTGRES_PASSWORD` setting. This stack uses separate named volumes and does not reuse the normal application's Qdrant collection or PostgreSQL data volume.

```powershell
docker compose -p localrag-eval -f docker-compose.yml -f .github/ci-compose.yml -f docker-compose.eval.yml up -d --build api
```

Wait until the evaluation API is ready:

```powershell
Invoke-RestMethod -Uri "http://localhost:18000/health"
Invoke-RestMethod -Uri "http://localhost:18000/ready"
```

Run the evaluation harness tests and benchmark:

```powershell
python -m unittest discover -s evals/tests -v
$env:LOCALRAG_EVAL_API_URL = "http://localhost:18000"
python .\evals\run_evaluation.py
```

The report is written to `evals/results/localrag-evaluation.json`.

Stop and remove the isolated evaluation stack when finished:

```powershell
docker compose -p localrag-eval -f docker-compose.yml -f .github/ci-compose.yml -f docker-compose.eval.yml down --volumes --remove-orphans
```

Do not add `--volumes` to commands targeting your normal LocalRAG project unless you intentionally want to delete that project's persistent data.
