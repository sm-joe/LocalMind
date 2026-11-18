import json
import tempfile
import unittest
from pathlib import Path

from scripts.backup_localrag import sha256_file, verify_backup


class BackupIntegrityTests(unittest.TestCase):
    def test_manifest_detects_tampered_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            files = {}
            for name, data in (("postgres.dump", b"pg-data"), ("qdrant.snapshot", b"qdrant-data")):
                file = path / name
                file.write_bytes(data)
                files[name] = {"size": file.stat().st_size, "sha256": sha256_file(file)}
            (path / "manifest.json").write_text(
                json.dumps({"format_version": 1, "files": files}), encoding="utf-8"
            )
            verify_backup(path)
            (path / "qdrant.snapshot").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "integrity check failed"):
                verify_backup(path)


if __name__ == "__main__":
    unittest.main()
