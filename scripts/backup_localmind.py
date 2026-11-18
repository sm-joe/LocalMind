"""LocalMind backup CLI.

This wrapper preserves the proven backup implementation and existing storage
identifiers while exposing the new product name to operators.

Usage (Bash):
    python3 scripts/backup_localmind.py --output backups
    python3 scripts/backup_localmind.py --verify backups/<backup-folder>
"""
import sys

from backup_localrag import main


if __name__ == "__main__":
    sys.exit(main())
