"""Global storage lock mechanism to prevent concurrent storage modifications."""

from __future__ import annotations

import json
import os
import sys
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


class StorageLockBusyError(RuntimeError):
    """Raised when the storage lock is held by another process or operation."""


class StorageLock:
    """Cross-platform non-blocking file lock."""

    def __init__(self, lock_path: Path, operation: str = "generic"):
        self.lock_path = lock_path
        self.operation = operation
        self._file: Any = None
        self._locked = False

    async def __aenter__(self) -> "StorageLock":
        self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def acquire(self) -> None:
        if self._locked:
            return

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        # Open in r+ or w+ mode
        try:
            self._file = open(self.lock_path, "a+b")
        except OSError as e:
            raise StorageLockBusyError(f"Cannot open lock file {self.lock_path}: {e}") from e

        try:
            if sys.platform == "win32":
                self._file.seek(0)
                # Lock 1 byte at position 0, LK_NBLCK is non-blocking lock
                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError) as e:
            if self._file:
                try:
                    self._file.close()
                except Exception:
                    pass
                self._file = None
            raise StorageLockBusyError(
                f"Storage lock is currently held by another process: {self.lock_path}"
            ) from e

        # Write lock metadata
        self._file.seek(0)
        self._file.truncate()
        lock_meta = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "operation": self.operation,
            "acquired_at": datetime.now(timezone.utc).isoformat(),
        }
        self._file.write(json.dumps(lock_meta).encode("utf-8"))
        self._file.flush()
        self._locked = True

    def get_metadata(self) -> dict | None:
        """Read metadata while holding the lock."""
        if not self._locked or self._file is None:
            return None
        self._file.seek(0)
        try:
            return json.loads(self._file.read().decode("utf-8"))
        except Exception:
            return None

    def release(self) -> None:
        if not self._locked or self._file is None:
            return

        try:
            if sys.platform == "win32":
                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        finally:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
            self._locked = False
