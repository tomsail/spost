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


def is_overlapping(tris, meshx):
    PIR = 180
    x1, x2, x3 = meshx[tris].T
    return np.logical_or(abs(x2 - x1) > PIR, abs(x3 - x1) > PIR, abs(x3 - x3) > PIR)


def build_simplices(ds: xr.Dataset) -> pd.DataFrame:
    """Extract triangle simplices from SCHISM mesh for datashader rendering."""
    faces = ds["SCHISM_hgrid_face_nodes"].values.astype("int64") - 1
    faces = faces[:, :3][~is_overlapping(faces[:, :3], ds["SCHISM_hgrid_node_x"].values)]
    return pd.DataFrame(faces, columns=["v0", "v1", "v2"])


def build_edge_df(coords_df: pd.DataFrame, simplices_df: pd.DataFrame, x: str = "SCHISM_hgrid_node_x", y: str = "SCHISM_hgrid_node_y") -> pd.DataFrame:
    """Build NaN-separated edge segments from mesh triangles for datashader line rendering."""
    xs, ys = coords_df[x].values, coords_df[y].values
    v = simplices_df[["v0", "v1", "v2"]].values  # (N, 3)
    nan = np.full(len(v), np.nan)

    edges = [(v[:, i], v[:, j]) for i, j in ((0, 1), (1, 2), (2, 0))]
    edge_x = np.stack([c for a, b in edges for c in (xs[a], xs[b], nan)], axis=1).ravel()
    edge_y = np.stack([c for a, b in edges for c in (ys[a], ys[b], nan)], axis=1).ravel()

    return pd.DataFrame({"x": edge_x, "y": edge_y})


def open_zarr_store(store_path: pathlib.Path) -> xr.Dataset:
    """Open a zarr store as an xarray Dataset."""
    return xr.open_zarr(store_path, chunks={})
