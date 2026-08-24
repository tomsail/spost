"""Tests for :func:`spost._tide.compute_sal` source dispatch."""
from __future__ import annotations

import pathlib

import pytest

from spost._tide import compute_sal


def test_compute_sal_requires_a_source(tmp_path: pathlib.Path):
    # Neither `fes` nor `tides` -> ambiguous, must raise before any IO.
    with pytest.raises(ValueError, match="exactly one SAL source"):
        compute_sal(output_dir=tmp_path)


def test_compute_sal_rejects_both_sources(tmp_path: pathlib.Path):
    # Both `fes` and `tides` -> ambiguous, must raise before any IO.
    fes = tmp_path / "fes"
    tides = tmp_path / "tides.zarr"
    fes.mkdir()
    tides.mkdir()
    with pytest.raises(ValueError, match="exactly one SAL source"):
        compute_sal(fes=fes, tides=tides, output_dir=tmp_path)
