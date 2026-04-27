"""Zarr-extraction tests against the bundled SCHISM run.

Skips automatically when ``tests/data/pimesh0/`` is not present.

Two scenarios are exercised:

- **Full span** — pointing ``to_zarr`` at the run root (which contains both
  hotstart segments) produces a single zarr covering the full timeline.
- **Incremental** — extracting after only the first segment is on disk, then
  re-running once the second segment has been added, must yield the same
  contents as the full-span run. This mimics the way SCHISM produces output:
  one segment at a time, with post-processing extending the zarr each pass.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import xarray as xr

from spost._to_zarr import to_zarr


# Keep the heavy work small: one variable, single worker, no compression
# tuning. Each zarr extraction across two segments still touches all the
# out2d/salinity/temperature files, so we lean on `workers=1` to make any
# failure modes deterministic.
_VARIABLES = ["elevation"]
_WORKERS = 1


def _open_zarr(path: pathlib.Path) -> xr.Dataset:
    return xr.open_zarr(path, chunks={}, consolidated=False)


def _assert_static_vars(ds: xr.Dataset) -> None:
    for var in (
        "SCHISM_hgrid_node_x",
        "SCHISM_hgrid_node_y",
        "SCHISM_hgrid_face_nodes",
        "depth",
    ):
        assert var in ds, f"static variable {var} missing from zarr"


def test_full_span_extraction(pimesh0_root, tmp_path):
    """Run ``to_zarr`` on the run root containing both segments."""
    store_path = tmp_path / "full.zarr"

    to_zarr(
        base_path=pimesh0_root,
        store_path=store_path,
        variables=_VARIABLES,
        workers=_WORKERS,
        clevel=1,
        overwrite=True,
    )

    ds = _open_zarr(store_path)
    _assert_static_vars(ds)
    assert "elevation" in ds.data_vars

    # Time axis must be strictly increasing and cover both segments.
    times = ds["time"].values
    assert times.size > 0
    assert np.all(np.diff(times.astype("datetime64[ns]")) > np.timedelta64(0, "ns"))

    # Second segment starts ~30 days after the first; combined timeline must
    # span at least the gap.
    span_days = (times[-1] - times[0]) / np.timedelta64(1, "D")
    assert span_days >= 29, f"full-span timeline too short: {span_days:.2f} days"

    # Elevation has the expected shape (time, nNodes) and is finite somewhere.
    elev = ds["elevation"].values
    assert elev.shape[0] == times.size
    assert np.isfinite(elev).any()


def test_incremental_extraction_matches_full_span(
    pimesh0_root, tmp_path, staged_run
):
    """Extract from one segment, then re-extract once the second arrives.

    The post-processing workflow is: as each hotstart segment lands, re-run
    ``to_zarr`` to refresh the store. The end result must match what we'd
    get from a single pass on the full set of segments.
    """
    store_path = tmp_path / "incremental.zarr"

    # Step 1 — only the first segment is on disk.
    run_dir = staged_run("20200101.00")
    to_zarr(
        base_path=run_dir,
        store_path=store_path,
        variables=_VARIABLES,
        workers=_WORKERS,
        clevel=1,
        overwrite=True,
    )
    first = _open_zarr(store_path)
    n_first = first["time"].size
    first.close()
    assert n_first > 0

    # Step 2 — second segment becomes available (simulated by copying it in).
    run_dir = staged_run("20200131.00")
    to_zarr(
        base_path=run_dir,
        store_path=store_path,
        variables=_VARIABLES,
        workers=_WORKERS,
        clevel=1,
        overwrite=True,
    )
    extended = _open_zarr(store_path)

    # Reference: full-span extraction directly from the data tree.
    full_path = tmp_path / "full_ref.zarr"
    to_zarr(
        base_path=pimesh0_root,
        store_path=full_path,
        variables=_VARIABLES,
        workers=_WORKERS,
        clevel=1,
        overwrite=True,
    )
    full = _open_zarr(full_path)

    # Same timeline.
    np.testing.assert_array_equal(extended["time"].values, full["time"].values)
    # Adding the second segment must extend (not shrink) the timeline.
    assert extended["time"].size > n_first
    # And the elevation field must agree across both runs.
    np.testing.assert_allclose(
        extended["elevation"].values,
        full["elevation"].values,
        equal_nan=True,
    )


def test_overwrite_required_when_store_exists(
    pimesh0_root, tmp_path, staged_run
):
    """Re-running without ``overwrite=True`` against an existing store fails.

    This pins down the current behaviour: callers must opt in to overwrite
    when extending after a segment lands.
    """
    store_path = tmp_path / "fail.zarr"
    run_dir = staged_run("20200101.00")
    to_zarr(
        base_path=run_dir,
        store_path=store_path,
        variables=_VARIABLES,
        workers=_WORKERS,
        clevel=1,
        overwrite=True,
    )
    with pytest.raises(Exception):
        to_zarr(
            base_path=run_dir,
            store_path=store_path,
            variables=_VARIABLES,
            workers=_WORKERS,
            clevel=1,
            overwrite=False,
        )
