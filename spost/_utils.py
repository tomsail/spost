from __future__ import annotations

import pathlib

import multifutures
import natsort
import numpy as np
import pandas as pd
import xarray as xr
from pytides2.tide import Tide
from tqdm.auto import tqdm


FULL = [
    "M2", "S2", "N2", "K2", "2N2", "L2", "T2", "R2", "NU2", "MU2", "EPS2", "LAMBDA2",  # Semi-diurnal (twice daily)
    "K1", "O1", "P1", "Q1", "J1", "S1",  # Diurnal (once daily)
    "MF", "MM", "MSF", "SA", "SSA", "MSQM", "MTM",  # Long period (fortnightly to annual)
    "M4", "MS4", "M6", "MN4", "N4", "S4", "M8", "M3", "MKS2",  # Short period (higher harmonics)
]
METRICS = ["amplitude", "phase"]


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
        chunks={"time":-1},
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


def pytides_to_df(pytides_tide: Tide) -> pd.DataFrame:
    constituent_names = [c.name.upper() for c in pytides_tide.model["constituent"]]
    return pd.DataFrame(pytides_tide.model, index=constituent_names).drop(
        "constituent",
        axis=1,
    )


def pytide_get_coefs(ts: pd.Series, resample: int = None) -> dict:
    if resample is not None:
        ts = ts.resample(f"{resample}min").mean()
        ts = ts.shift(freq=f"{resample / 2}min")  # Center the resampled points
    ts = ts.dropna()
    return Tide.decompose(ts.values, ts.index.to_pydatetime())[0]


def reduce_coef_to_fes(df: pd.DataFrame, cnst: list, verbose: bool = False):
    res = pd.DataFrame(0.0, index=cnst, columns=df.columns)
    common_constituents = df.index.intersection(cnst)
    res.loc[common_constituents] = df.loc[common_constituents]

    not_in_fes_df = df[~df.index.isin(cnst)]
    not_in_fes = not_in_fes_df.index.tolist()
    not_in_fes_amps = not_in_fes_df["amplitude"].round(3).tolist()
    missing_fes = set(cnst) - set(df.index)

    if verbose:
        print(f"Constituents found but not in FES: {not_in_fes}")
        print(f"Their amplitudes: {not_in_fes_amps}")
        if missing_fes:
            print(
                f"FES constituents missing from analysis (set to 0): {sorted(missing_fes)}",
            )

    return res


def analyze_node(ts_np: np.ndarray, time_index: pd.DatetimeIndex) -> np.ndarray:
    ts = pd.Series(ts_np, index=time_index, name="elev")
    df = pytides_to_df(pytide_get_coefs(ts, 60))
    df = reduce_coef_to_fes(df, cnst=FULL)
    return df.to_numpy()


def analyze_block(ds_block: xr.DataArray) -> np.ndarray:
    time_index = pd.DatetimeIndex(ds_block["time"].values)
    data = ds_block.values  # shape (Nt, Nblock_nodes)
    results = np.stack(
        [analyze_node(data[:, i], time_index) for i in range(data.shape[1])],
        axis=0  # (Nblock_nodes, Nconstituents, Nmetrics)
    )
    return results


def detide(data_array: xr.DataArray, chunk_size: int = 50) -> np.ndarray:
    n_nodes = data_array.sizes['nSCHISM_hgrid_node']
    chunks = []
    for i in range(0, n_nodes, chunk_size):
        end_idx = min(i + chunk_size, n_nodes)
        chunk = {"ds_block": data_array.isel(nSCHISM_hgrid_node=slice(i, end_idx))}
        chunks.append(chunk)
    future_results = multifutures.multiprocess(analyze_block, chunks)
    results = [fr.result for fr in future_results if fr.exception is None]
    final_result = np.concatenate(results, axis=0)
    return final_result
