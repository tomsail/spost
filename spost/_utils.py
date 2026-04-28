import pathlib

import natsort
import numpy as np
import pandas as pd
import xarray as xr


def open_schism_output(
    base_path: pathlib.Path,
    pattern: str,
    exclude_last: int = 0,
) -> xr.Dataset:
    files = natsort.natsorted(base_path.glob(f"**/{pattern}"))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {base_path}")
    if exclude_last:
        if exclude_last >= len(files):
            raise ValueError(
                f"exclude_last={exclude_last} would drop all {len(files)} files "
                f"matching {pattern!r}"
            )
        files = files[:-exclude_last]
    ds = xr.open_mfdataset(
        files,
        data_vars="minimal",
        coords="minimal",
        chunks={},
        compat="override",
        mask_and_scale=False,
    )
    return ds


def sanitize_attrs(attrs: dict) -> dict:
    sanitized = {}
    for k, v in attrs.items():
        if isinstance(v, np.generic):
            sanitized[k] = v.item()
        elif isinstance(v, np.ndarray):
            sanitized[k] = v.tolist()
        else:
            sanitized[k] = v
    return sanitized


def build_simplices(ds: xr.Dataset) -> pd.DataFrame:
    """Extract triangle simplices from SCHISM mesh for datashader rendering."""
    faces = ds["SCHISM_hgrid_face_nodes"].values.astype("int64") - 1
    tris = faces[:, :3]
    return pd.DataFrame(tris, columns=["v0", "v1", "v2"])


def open_zarr_store(store_path: pathlib.Path) -> xr.Dataset:
    """Open a zarr store as an xarray Dataset."""
    return xr.open_zarr(store_path, chunks={})
