"""Tests for :func:`spost._to_zarr.derive_velocity_polar`."""
from __future__ import annotations

import pathlib

import numpy as np
import zarr

from spost._to_zarr import derive_velocity_polar
from spost._to_zarr import VELOCITY_X
from spost._to_zarr import VELOCITY_Y


def _make_store(path: pathlib.Path, vx: np.ndarray, vy: np.ndarray) -> None:
    group = zarr.create_group(store=path, overwrite=True, zarr_format=3)
    for name, data in ((VELOCITY_X, vx), (VELOCITY_Y, vy)):
        group.create_array(
            name=name,
            shape=data.shape,
            dtype="float32",
            dimension_names=("time", "nSCHISM_hgrid_node"),
            attributes={"grid_mapping": "crs", "mesh": "SCHISM_hgrid", "location": "node"},
            chunks=(data.shape[0], 2),
            fill_value=None,
        )
        group[name][:] = data


def test_derive_velocity_polar_values(tmp_path: pathlib.Path):
    store = tmp_path / "run.zarr"
    vx = np.array([[3.0, 0.0, -1.0]], dtype="float32")
    vy = np.array([[4.0, 2.0, 0.0]], dtype="float32")
    _make_store(store, vx, vy)

    derive_velocity_polar(store, workers=1)

    group = zarr.open_group(store)
    np.testing.assert_allclose(group["velocity_magnitude"][:], np.hypot(vx, vy), rtol=1e-6)
    np.testing.assert_allclose(
        group["velocity_angle"][:], np.degrees(np.arctan2(vy, vx)), rtol=1e-6
    )
    # Inherited attributes are carried over for downstream plotting.
    assert group["velocity_magnitude"].attrs["grid_mapping"] == "crs"
    assert group["velocity_angle"].attrs["units"] == "degree"
