"""Back up LocalMind PostgreSQL and Qdrant data without deleting volumes.

Usage (Bash):
    python3 scripts/backup_localrag.py --output backups

Requires Docker Compose services api, postgres, and qdrant to be running.
For a consistent backup, pause document uploads and chat writes while running.
Backups contain private data and must be encrypted at rest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


def run(*args: str) -> str:
    result = subprocess.run(
        list(args),
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_backup(output_directory: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    identifier = uuid.uuid4().hex[:8]
    destination = output_directory / f"localmind-{timestamp}-{identifier}"
    destination.mkdir(parents=True, exist_ok=False)

    temp_db_path = f"/tmp/localrag-backup-{identifier}.dump"
    postgres_dump = destination / "postgres.dump"
    qdrant_snapshot = destination / "qdrant.snapshot"

    try:
        # Copy the binary pg_dump directly to avoid text encoding conversions.
        run(
            "docker", "exec", "localrag-postgres",
            "pg_dump", "-U", "localrag", "-Fc",
            "-f", temp_db_path, "localrag",
        )
        run(
            "docker", "cp",
            f"localrag-postgres:{temp_db_path}",
            str(postgres_dump),
        )
    finally:
        subprocess.run(
            ["docker", "exec", "localrag-postgres", "rm", "-f", temp_db_path],
            check=False, capture_output=True,
        )

    snapshot_script = (
        "from app.config import get_settings;"
        "from qdrant_client import QdrantClient;"
        "s=get_settings();"
        "c=QdrantClient(host=s.qdrant_host,port=s.qdrant_port);"
        "r=c.create_snapshot(collection_name=s.qdrant_collection,wait=True);"
        "print('LOCALRAG_COLLECTION='+s.qdrant_collection);"
        "print('LOCALRAG_SNAPSHOT='+r.name)"
    )
    output = run("docker", "exec", "localrag-api", "python", "-c", snapshot_script)
    values = dict(
        line.split("=", 1)
        for line in output.splitlines()
        if line.startswith(("LOCALRAG_COLLECTION=", "LOCALRAG_SNAPSHOT="))
    )
    collection = values.get("LOCALRAG_COLLECTION", "")
    snapshot = values.get("LOCALRAG_SNAPSHOT", "")
    safe_part = re.compile(r"^[a-zA-Z0-9_.-]+$")
    if not safe_part.fullmatch(collection) or not safe_part.fullmatch(snapshot):
        raise RuntimeError("Qdrant returned an invalid snapshot identifier.")

    run(
        "docker", "cp",
        f"localrag-qdrant:/qdrant/snapshots/{collection}/{snapshot}",
        str(qdrant_snapshot),
    )

    manifest = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collection": collection,
        "qdrant_snapshot_name": snapshot,
        "files": {
            path.name: {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (postgres_dump, qdrant_snapshot)
        },
        "warning": "Postgres and Qdrant snapshots are not atomic across databases.",
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def verify_backup(path: Path) -> None:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported backup manifest format.")
    for filename in ("postgres.dump", "qdrant.snapshot"):
        expected = manifest["files"][filename]
        file = path / filename
        if file.stat().st_size != expected["size"] or sha256_file(file) != expected["sha256"]:
            raise ValueError(f"Backup integrity check failed for {filename}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("backups"))
    parser.add_argument("--verify", type=Path, help="Verify an existing backup directory")
    args = parser.parse_args()
    try:
        if args.verify:
            verify_backup(args.verify)
            print("Backup manifest and file hashes verified.")
        else:
            destination = create_backup(args.output)
            verify_backup(destination)
            print(f"LocalMind backup created and verified: {destination}")
        return 0
    except (subprocess.CalledProcessError, OSError, KeyError, ValueError, RuntimeError) as exc:
        print(f"Backup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
