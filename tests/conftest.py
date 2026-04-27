"""Shared pytest fixtures for spost tests."""

from __future__ import annotations

import pathlib
import shutil

import pytest

TESTS_DIR = pathlib.Path(__file__).parent
DATA_DIR = TESTS_DIR / "data"
PIMESH0 = DATA_DIR / "pimesh0"
SEGMENTS = ("20200101.00", "20200131.00")


def _segment_outputs(segment: str) -> pathlib.Path:
    return PIMESH0 / segment / "outputs"


@pytest.fixture(scope="session")
def pimesh0_root() -> pathlib.Path:
    """Path to the bundled SCHISM run used by the heavier tests."""
    if not PIMESH0.exists():
        pytest.skip(f"Test data not available: {PIMESH0}")
    return PIMESH0


@pytest.fixture(scope="session")
def segment_dirs(pimesh0_root: pathlib.Path) -> list[pathlib.Path]:
    """The available outputs/ directories, sorted chronologically."""
    out: list[pathlib.Path] = []
    for seg in SEGMENTS:
        d = _segment_outputs(seg)
        if d.is_dir():
            out.append(d)
    if len(out) < 2:
        pytest.skip(
            f"Need both segments under {PIMESH0}; found {[p.parent.name for p in out]}"
        )
    return out


@pytest.fixture
def staged_run(tmp_path: pathlib.Path, segment_dirs: list[pathlib.Path]):
    """Helper to copy a subset of segments into an isolated working tree.

    Returns a callable ``stage(*segments)`` that copies the named segment
    directories (e.g. ``stage("20200101.00")``) into ``tmp_path/run/`` and
    returns the ``tmp_path/run`` path. Calling it again with more segments
    appends them.
    """
    base = tmp_path / "run"
    base.mkdir(exist_ok=True)

    def stage(*segments: str) -> pathlib.Path:
        for seg in segments:
            src = PIMESH0 / seg
            dst = base / seg
            if dst.exists():
                continue
            shutil.copytree(src, dst)
        return base

    return stage
