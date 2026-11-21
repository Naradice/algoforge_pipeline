"""Unit tests for StorageLock."""

import pytest
from pathlib import Path
from research_pipeline.storage.lock import StorageLock, StorageLockBusyError


@pytest.mark.asyncio
async def test_storage_lock_acquire_and_release(tmp_path: Path):
    lock_path = tmp_path / "test.lock"
    async with StorageLock(lock_path, operation="test_op") as lock:
        assert lock._locked
        assert lock_path.exists()
        meta = lock.get_metadata()
        assert meta is not None
        assert meta["operation"] == "test_op"

    # Should be unlocked now
    assert not lock._locked


@pytest.mark.asyncio
async def test_storage_lock_prevents_concurrent_access(tmp_path: Path):
    lock_path = tmp_path / "test.lock"
    lock1 = StorageLock(lock_path, operation="holder")
    lock1.acquire()

    lock2 = StorageLock(lock_path, operation="intruder")
    with pytest.raises(StorageLockBusyError):
        lock2.acquire()

    lock1.release()

    # Now lock2 can acquire
    lock2.acquire()
    assert lock2._locked
    lock2.release()
