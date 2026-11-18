"""Read-only audit of currently indexed document text.

Run inside API container:
    python -m app.tools.index_audit Interview_Questions.docx EC2 IAM Bedrock

Prints only small matching snippets. Never alters Qdrant, PostgreSQL or Ollama.
"""
import argparse
import re

from app.config import get_settings
from app.services.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", help="Exact indexed filename")
    parser.add_argument("terms", nargs="+", help="Terms to look for, case-insensitive")
    args = parser.parse_args()

    settings = get_settings()
    store = VectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection,
    )
    indexed = []
    for point in store.iter_points():
        payload = point.payload or {}
        if payload.get("filename") == args.filename:
            indexed.append(payload)
    indexed.sort(key=lambda item: int(item.get("chunk_id", 0)))
    print(f"File: {args.filename} | indexed chunks: {len(indexed)}")
    if not indexed:
        print("No chunks found with that exact filename in the configured collection.")
        return
    for term in args.terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        hits = []
        for item in indexed:
            text = str(item.get("text", ""))
            match = pattern.search(text)
            if match:
                start = max(0, match.start() - 65)
                end = min(len(text), match.end() + 100)
                snippet = " ".join(text[start:end].split())
                hits.append((item.get("chunk_id"), snippet))
        print(f"\n{term}: {len(hits)} matching chunks")
        for chunk_id, snippet in hits[:4]:
            print(f"  chunk {chunk_id}: {snippet}")
    print("\nRead-only audit complete. Missing matches indicate absent indexed text,")
    print("not necessarily absent source-document text.")


if __name__ == "__main__":
    main()
