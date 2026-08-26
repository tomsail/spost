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
    df = pytides_to_df(pytides_get_coefs(ts))
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


def tidal_arguments(
    start_date: str,
    constituents: list[str],
    corrections: str = "FES",
) -> tuple[np.ndarray, np.ndarray]:
    """Equilibrium argument (V₀+u) and nodal factor f at a reference epoch.

    Uses pyTMD's astronomical arguments so the result is consistent with the
    convention used to synthesise FES/pyTMD harmonic constants, and with
    SCHISM's own ``tear`` (V₀+u) and ``tnf`` (f) that it stores for the
    earth-tidal-potential term.

    pyTMD predicts a tide as ``f * A * cos(theta - G_lag)`` with
    ``theta = radians(G) + u`` (see ``pyTMD.predict.time_series``), where
    ``G`` is the astronomical argument and ``u`` the nodal angle. Hence the
    full equilibrium argument returned here is ``(V₀+u) = G + degrees(u)``.

    Parameters
    ----------
    start_date : str
        Reference epoch (ISO-8601, e.g. ``"2020-01-01"`` or
        ``"2020-01-01T00:00:00"``). For SCHISM this is the run start (t=0).
    constituents : list[str]
        Constituent names (case-insensitive, e.g. ``["M2", "S2", ...]``).
    corrections : str, default "FES"
        Nodal-correction convention passed to
        :func:`pyTMD.constituents.arguments` (``"FES"``, ``"OTIS"`` or
        ``"GOT"``). Use ``"FES"`` when the harmonic constants come from a FES
        atlas.

    Returns
    -------
    Vu : np.ndarray
        Equilibrium argument ``V₀+u`` in **degrees**, shape ``(n_constituents,)``.
    f : np.ndarray
        Nodal modulation factor (dimensionless), shape ``(n_constituents,)``.
    """
    import datetime as _dt

    import pyTMD.constituents
    import timescale.time

    dt = _dt.datetime.fromisoformat(str(start_date))
    ts = timescale.time.Timescale().from_calendar(
        dt.year, dt.month, dt.day, dt.hour, dt.minute,
        dt.second + dt.microsecond / 1e6,
    )
    mjd = np.atleast_1d(ts.MJD)
    cons = [str(c).lower() for c in constituents]
    pu, pf, G = pyTMD.constituents.arguments(mjd, cons, corrections=corrections)
    vu = (G[0] + np.degrees(pu[0]))  # V0 + u, degrees
    f = pf[0]
    return vu, f


# ---------------------------------------------------------------------------
# Spherical-harmonic self-attraction & loading (SAL)
# ---------------------------------------------------------------------------
# Physical constants (kg/m3).
RHO_WATER = 1026.0
RHO_EARTH = 5517.0

# Load Love numbers h'_n, k'_n (PREM, elastic). This short table (n=2,3,4) is
# the demo set; degrees beyond the table hold the highest tabulated value.
# For production accuracy, replace/extend with a full table (Wang et al. 2012,
# or Delft3D's LOAD_LOVE_NUMBERS_H/K in timespace_data_tables.f90).
LOAD_LOVE: dict[int, tuple[float, float]] = {
    2: (-1.001, -0.3075),
    3: (-1.052, -0.195),
    4: (-1.058, -0.132),
}


def nodes_to_dh_grid(
    values: np.ndarray,
    node_lon: np.ndarray,
    node_lat: np.ndarray,
    n: int = 360,
    fill: float = 0.0,
    seam_pad: float = 5.0,
) -> np.ndarray:
    """Interpolate a scattered REAL field onto a Driscoll-Healy grid.

    Spherical-harmonic analysis operates on a regular grid, but SCHISM tidal
    constants live on the unstructured mesh nodes. This resamples the nodal
    field onto a DH grid of shape ``(n, 2n)`` using linear (Delaunay)
    interpolation.

    DH convention (matches :class:`pyshtools.SHGrid`): ``lat_i = 90 - 180*i/n``
    (i.e. 90 -> just above -90) and ``lon_j = 360*j/(2n)`` (0 -> just below
    360). Cells with no mesh coverage (land, or outside a regional mesh) are set
    to ``fill`` (0 for SAL, so land contributes nothing to the expansion).

    Parameters
    ----------
    values
        Real nodal field, shape ``(n_nodes,)``.
    node_lon, node_lat
        Node coordinates in degrees, shape ``(n_nodes,)``. ``node_lon`` may be
        in either ``[-180, 180]`` or ``[0, 360]``.
    n
        Number of DH latitude bands (must be even). ``2n`` longitudes.
    fill
        Value for grid cells outside the mesh footprint.
    seam_pad
        Nodes within this many degrees of the 0/360 longitude seam are
        duplicated on the far side so linear interpolation does not leave a
        gap along the meridian. Keeps the duplicated-point count small on
        large global meshes.
    """
    from scipy.interpolate import griddata

    if n % 2:
        raise ValueError(f"DH grid needs an even number of latitudes, got n={n}")

    lat_dh = 90.0 - 180.0 * np.arange(n) / n          # (n,)
    lon_dh = 360.0 * np.arange(2 * n) / (2 * n)        # (2n,)
    lon2d, lat2d = np.meshgrid(lon_dh, lat_dh)         # (n, 2n)

    lon = np.mod(np.asarray(node_lon, dtype=float), 360.0)
    lat = np.asarray(node_lat, dtype=float)
    vals = np.asarray(values, dtype=float)

    # Duplicate only the near-seam nodes on the opposite side of 0/360.
    left = lon <= seam_pad
    right = lon >= 360.0 - seam_pad
    pts_lon = np.concatenate([lon, lon[left] + 360.0, lon[right] - 360.0])
    pts_lat = np.concatenate([lat, lat[left], lat[right]])
    pts_val = np.concatenate([vals, vals[left], vals[right]])

    grid = griddata(
        (pts_lon, pts_lat), pts_val, (lon2d, lat2d),
        method="linear", fill_value=fill,
    )
    return np.nan_to_num(grid, nan=fill)


def sal_convolution_real(
    eta_grid: np.ndarray,
    love: dict[int, tuple[float, float]] = LOAD_LOVE,
    lat_out: np.ndarray | None = None,
    lon_out: np.ndarray | None = None,
    mode: str = "sal",
    nmin: int = 2,
    rho_w: float = RHO_WATER,
    rho_e: float = RHO_EARTH,
) -> np.ndarray:
    """Spherical-harmonic convolution of a REAL field (metres).

    Expands ``eta_grid`` (a 2-D Driscoll-Healy grid) into spherical harmonics,
    scales each degree ``n`` by the load-tide response, then synthesises the
    result. With ``mode='sal'`` the per-degree weight is
    ``(3*rho_w/rho_e) * (1 + k'_n - h'_n) / (2n+1)`` (the self-attraction &
    loading forcing, in phase with the ocean tide); with ``mode='load'`` it is
    the crustal-loading displacement weight ``h'_n``.

    Parameters
    ----------
    eta_grid
        Real DH grid, shape ``(nlat, nlon)`` (e.g. from :func:`nodes_to_dh_grid`).
    love
        Mapping ``degree -> (h'_n, k'_n)``. Degrees above the highest key reuse
        that highest entry.
    lat_out, lon_out
        If given (degrees, any broadcastable shape), synthesise the result at
        those points (e.g. the SCHISM nodes) instead of returning a grid.
    mode
        ``'sal'`` or ``'load'``.
    nmin
        Minimum degree kept (degrees ``< nmin`` are zeroed; the degree-0/1
        terms are not physically meaningful for SAL).
    """
    import pyshtools as pysh

    coeffs = pysh.SHGrid.from_array(np.asarray(eta_grid, dtype=float), grid="DH").expand()
    n_last = max(love)
    c = coeffs.coeffs.copy()
    for n in range(coeffs.lmax + 1):
        if n >= nmin:
            hp, kp = love.get(n, love[n_last])   # hold highest degree beyond table
            fac = (1.0 + kp - hp) if mode == "sal" else hp
            c[:, n, :] *= (3.0 * rho_w / rho_e) * fac / (2.0 * n + 1.0)
        else:
            c[:, n, :] = 0.0
    out = pysh.SHCoeffs.from_array(
        c, normalization=coeffs.normalization, csphase=coeffs.csphase
    )
    if lat_out is not None:
        return out.expand(
            lat=np.asarray(lat_out, dtype=float),
            lon=np.asarray(lon_out, dtype=float),
        )
    return out.expand(grid="DH2").data


def sal_from_constituents(
    z: np.ndarray,
    node_lon: np.ndarray,
    node_lat: np.ndarray,
    love: dict[int, tuple[float, float]] = LOAD_LOVE,
    n: int = 360,
    mode: str = "sal",
) -> np.ndarray:
    """Complex tidal constants on the mesh -> complex SAL constants on the mesh.

    Pipeline: scattered nodes -> DH grid (real & imag separately) -> SH
    convolution -> synthesis back at the same nodes. The operation is linear,
    so the same amplitude/phase convention as the input (``z = A * exp(-i*G)``,
    Greenwich phase lag ``G``) carries through to the output.

    Parameters
    ----------
    z
        Complex tidal constants, shape ``(n_nodes,)`` (single constituent) or
        ``(n_nodes, n_constituents)``.
    node_lon, node_lat
        Node coordinates in degrees, shape ``(n_nodes,)``.
    love
        Load Love-number table (see :data:`LOAD_LOVE`).
    n
        DH grid resolution (latitude bands).
    mode
        ``'sal'`` (default) or ``'load'``.

    Returns
    -------
    numpy.ndarray
        Complex SAL constants with the same shape as ``z``.
    """
    z = np.asarray(z)
    if z.ndim == 1:
        cols = z[:, None]
    else:
        cols = z
    out = np.empty_like(cols, dtype=complex)
    for k in range(cols.shape[1]):
        col = cols[:, k]
        re = nodes_to_dh_grid(col.real, node_lon, node_lat, n=n)
        im = nodes_to_dh_grid(col.imag, node_lon, node_lat, n=n)
        s_re = sal_convolution_real(re, love, lat_out=node_lat, lon_out=node_lon, mode=mode)
        s_im = sal_convolution_real(im, love, lat_out=node_lat, lon_out=node_lon, mode=mode)
        out[:, k] = s_re + 1j * s_im
    return out.reshape(z.shape)
