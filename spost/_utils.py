from __future__ import annotations

import pathlib

import multifutures
import natsort
import numpy as np
import pandas as pd
import xarray as xr

FULL = [
    "M2", "S2", "N2", "K2", "2N2", "L2", "T2", "R2", "NU2", "MU2", "EPS2", "LAMBDA2",  # Semi-diurnal (twice daily)
    "K1", "O1", "P1", "Q1", "J1", "S1",  # Diurnal (once daily)
    "MF", "MM", "MSF", "SA", "SSA", "MSQM", "MTM",  # Long period (fortnightly to annual)
    "M4", "MS4", "M6", "MN4", "N4", "S4", "M8", "M3", "MKS2",  # Short period (higher harmonics)
]
SAL = ["M2", "S2", "K2", "N2", "O1", "P1", "Q1", "K1"]


def open_schism_output(
    base_path: pathlib.Path,
    pattern: str,
    exclude_last: int = 0,
) -> xr.Dataset:
    # NOTE: recurse_symlinks=True is required (Python 3.13+): SCHISM `outputs/`
    # directories on HPC are commonly symlinks to scratch storage, and by
    # default Path.glob("**/...") does NOT descend into symlinked directories,
    # which would silently find zero files.
    files = natsort.natsorted(base_path.glob(f"**/{pattern}", recurse_symlinks=True))

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


def build_faces(ds: xr.Dataset) -> np.ndarray:
    """Full triangle connectivity (0-based) from a SCHISM mesh.

    Returns the complete ``(ne, 3)`` node-index array with no filtering, so it
    is safe for writing mesh files. Use :func:`build_simplices` instead for
    datashader rendering, which additionally drops dateline-crossing elements.
    """
    faces = ds["SCHISM_hgrid_face_nodes"].values[:, :3]
    return faces.astype("int64") - 1


def build_simplices(ds: xr.Dataset) -> pd.DataFrame:
    """Extract triangle simplices from SCHISM mesh for datashader rendering."""
    faces = build_faces(ds)
    faces = faces[~is_overlapping(faces, ds["SCHISM_hgrid_node_x"].values)]
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


def pytides_to_df(pytides_tide) -> pd.DataFrame:
    constituent_names = [c.name.upper() for c in pytides_tide.model["constituent"]]
    df = pd.DataFrame(pytides_tide.model, index=constituent_names).drop(
        "constituent",
        axis=1,
    )
    df["z"] = df["amplitude"] * np.exp(-1j*np.deg2rad(df["phase"]))
    return df[["z"]]


def pytides_get_coefs(ts: pd.Series, resample: int = None) -> dict:
    from pytides2.tide import Tide

    if resample is not None:
        ts = ts.resample(f"{resample}min").mean()
        ts = ts.shift(freq=f"{resample / 2}min")  # Center the resampled points
    ts = ts.dropna()
    return Tide.decompose(ts.values, ts.index.to_pydatetime())[0]


def keep_common_consituents(df: pd.DataFrame, cnst: list) -> pd.DataFrame:
    res = pd.DataFrame(0.0 + 0.0j, index=cnst, columns=df.columns)
    common_constituents = df.index.intersection(cnst)
    res.loc[common_constituents] = df.loc[common_constituents]
    return res


def analyze_node(ts_np: np.ndarray, time_index: pd.DatetimeIndex) -> np.ndarray:
    ts = pd.Series(ts_np, index=time_index, name="elev")
    df = pytides_to_df(pytides_get_coefs(ts, 60))
    df = keep_common_consituents(df, FULL)
    return df["z"].to_numpy()


def analyze_block(ds_block: xr.DataArray, start: int) -> tuple[int, np.ndarray]:
    time_index = pd.DatetimeIndex(ds_block["time"].values)
    data = ds_block.values  # shape (Nt, Nblock_nodes)
    results = np.stack(
        [analyze_node(data[:, i], time_index) for i in range(data.shape[1])],
        axis=0  # (Nblock_nodes, Nconstituents)
    )
    return start, results


def detect_tide_model(directory: pathlib.Path, candidates: list[str]):
    import pyTMD

    for name in candidates:
        try:
            m = pyTMD.io.model(directory).from_database(name)
        except (FileNotFoundError, ValueError, KeyError):
            continue
        return name, m

    raise FileNotFoundError(f"No known tide model found under {directory} (tried {candidates})")


def load_tide_model(directory: pathlib.Path, name: str):
    """Load a single pyTMD tide model by its database ``name``.

    ``directory`` is the root of a pyTMD-style store (the directory that holds
    the per-model sub-directories, e.g. ``fes2014/``, ``fes2022b/``,
    ``TPXO10_atlas_v2/`` ...). ``name`` must be a pyTMD database model name such
    as ``FES2014``, ``FES2022_extrapolated`` or ``TPXO10-atlas-v2-nc``; pyTMD
    resolves the expected sub-directory/file layout under ``directory`` for us.
    """
    import pyTMD

    try:
        return pyTMD.io.model(directory).from_database(name)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise FileNotFoundError(
            f"Could not load tide model {name!r} under {directory}. "
            "Check that the name is a valid pyTMD database model and that the "
            "pyTMD-style sub-directories exist under that root."
        ) from exc


def interpolate_tide_model(
    directory: pathlib.Path,
    name: str,
    lons: np.ndarray,
    lats: np.ndarray,
    constituents: list[str] = FULL,
) -> np.ndarray:
    """Interpolate a named pyTMD tide model onto ``(lons, lats)`` mesh nodes.

    Returns a complex ``(n_nodes, n_constituents)`` array aligned with
    ``constituents``; entries for constituents missing from the model are left
    as ``nan``.
    """
    m = load_tide_model(directory, name)
    model_name = name
    ds = m.open_dataset(group="z", use_default_units=True)

    name_map = {c.upper(): c for c in ds.tmd.constituents}
    available = [name_map[c] for c in constituents if c in name_map]
    if not available:
        raise ValueError(
            f"None of the requested constituents {constituents} are available "
            f"in tide model {model_name!r} (has {sorted(name_map)})"
        )
    ds = ds[available]

    lon = np.asarray(lons, dtype=float)
    if float(ds["x"].max()) > 180.0:
        lon = np.where(lon < 0, lon + 360.0, lon)
    lat = np.asarray(lats, dtype=float)

    node_dim = "node"
    interpolated = ds.interp(
        x=xr.DataArray(lon, dims=node_dim),
        y=xr.DataArray(lat, dims=node_dim),
    )

    result = np.full((len(lon), len(constituents)), np.nan + 0.0j, dtype=complex)
    col = {str(c).upper(): i for i, c in enumerate(constituents)}
    for c in available:
        result[:, col[c.upper()]] = interpolated[c].values
    return result


def harmonic_analysis(data_array: xr.DataArray, chunk_size: int = 50) -> np.ndarray:
    n_nodes = data_array.sizes['nSCHISM_hgrid_node']
    chunks = []
    for i in range(0, n_nodes, chunk_size):
        end_idx = min(i + chunk_size, n_nodes)
        chunk = {"ds_block": data_array.isel(nSCHISM_hgrid_node=slice(i, end_idx)), "start": i}
        chunks.append(chunk)
    future_results = multifutures.multiprocess(analyze_block, chunks, check=True, include_kwargs=False)
    # multiprocess returns blocks in completion order, not submission order
    blocks = sorted((fr.result for fr in future_results), key=lambda block: block[0])
    final_result = np.concatenate([result for _, result in blocks], axis=0)
    return final_result


def interpolate_load_tide(
    directory: pathlib.Path,
    lons: np.ndarray,
    lats: np.ndarray,
    constituents: list[str],
) -> np.ndarray:
    from pyTMD.io import FES

    directory = pathlib.Path(directory)
    file_map: dict[str, pathlib.Path] = {}
    for f in sorted(directory.glob("*.nc")):
        cname = f.stem.split("_")[0].upper()
        file_map.setdefault(cname, f)

    lon = np.asarray(lons, dtype=float)
    lat = np.asarray(lats, dtype=float)
    result = np.full((len(lon), len(constituents)), np.nan + 0.0j, dtype=complex)
    col = {str(c).upper(): i for i, c in enumerate(constituents)}

    missing = []
    for c in constituents:
        cu = str(c).upper()
        f = file_map.get(cu)
        if f is None:
            missing.append(c)
            continue
        ds = FES.open_fes_dataset(f, format="netcdf", group="z")
        var = next(iter(ds.data_vars))
        da = ds[var]
        units = str(da.attrs.get("units", "")).lower()
        if units in ("cm", "centimeter", "centimeters"):
            da = da / 100.0
        elif units in ("mm", "millimeter", "millimeters"):
            da = da / 1000.0
        qlon = lon.copy()
        if float(da["x"].max()) > 180.0:
            qlon = np.where(qlon < 0, qlon + 360.0, qlon)
        interp = da.interp(
            x=xr.DataArray(qlon, dims="node"),
            y=xr.DataArray(lat, dims="node"),
        )
        result[:, col[cu]] = interp.values

    if missing:
        print(f"WARNING: no FES load-tide file found for constituents: {missing}")
    return result
