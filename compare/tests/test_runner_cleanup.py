from __future__ import annotations

from pathlib import Path

import pytest

from physeval_compare.runner import CompareRunner, select_shard


def _runner(cache_root: Path) -> CompareRunner:
    runner = object.__new__(CompareRunner)
    runner.cache_root = cache_root
    return runner


def test_cleanup_frames_removes_only_prompt_cache(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    frames_dir = cache_root / "frames" / "model" / "V001"
    frames_dir.mkdir(parents=True)
    (frames_dir / "00000.jpg").write_bytes(b"frame")

    _runner(cache_root)._cleanup_frames(frames_dir)

    assert not frames_dir.exists()
    assert cache_root.exists()


def test_cleanup_frames_refuses_outside_path(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    outside = tmp_path / "video"
    outside.mkdir()

    with pytest.raises(RuntimeError, match="outside the frame cache"):
        _runner(cache_root)._cleanup_frames(outside)

    assert outside.exists()


def test_select_shard_is_disjoint_and_complete() -> None:
    rows = [{"Index": index} for index in range(10)]
    shards = [select_shard(rows, 4, shard_index) for shard_index in range(4)]

    flattened = [row["Index"] for shard in shards for row in shard]
    assert sorted(flattened) == list(range(10))
    assert len(flattened) == len(set(flattened))


def test_select_shard_validates_index() -> None:
    with pytest.raises(ValueError, match="shard-index"):
        select_shard([], 4, 4)
