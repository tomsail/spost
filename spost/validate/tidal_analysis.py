"""Full-mesh tidal harmonic decomposition.

For every node in the SCHISM mesh, fit ``pytides2`` to the elevation
timeseries and reduce to a target list of constituents. Joblib parallelizes
across chunks of nodes; thread counts are pinned to 1 inside workers to avoid
thread contention.
"""

from __future__ import annotations

import datetime
import logging
import os
import pathlib
from collections.abc import Sequence

from . import _paths

logger = logging.getLogger(__name__)


# Default constituent list (as specified in the spec, §4.4 "FULL").
SEMI_DIURNAL = ["M2", "S2", "N2", "K2", "2N2", "L2", "T2", "R2", "NU2", "MU2", "EPS2", "LAMBDA2"]
DIURNAL = ["K1", "O1", "P1", "Q1", "J1", "S1"]
LONG_PERIOD = ["MF", "MM", "MSF", "SA", "SSA", "MSQM", "MTM"]
HIGHER_HARMONICS = ["M4", "MS4", "M6", "MN4", "N4", "S4", "M8", "M3", "MKS2"]
FULL_CONSTITUENTS: list[str] = SEMI_DIURNAL + DIURNAL + LONG_PERIOD + HIGHER_HARMONICS


def _coerce_dt(value: datetime.datetime | str) -> datetime.datetime:
    if isinstance(value, datetime.datetime):
        return value
    return datetime.datetime.fromisoformat(value)


def _pin_threads() -> None:
    """Disable BLAS-level threading inside joblib workers."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def _decompose_node(
    times,
    values,
    constituents: Sequence[str],
):
    """Run pytides2 on a single node's timeseries and reduce to target constituents."""
    import numpy as np
    import pandas as pd

    try:
        from pytides2.tide import Tide
    except ImportError as exc:
        raise ImportError(
            "pytides2 is required for tidal-analysis. "
            "Install with: pip install spost[validate]"
        ) from exc

    series = pd.Series(values, index=pd.DatetimeIndex(times)).dropna()
    if len(series) < 24:
        return np.full((len(constituents), 2), np.nan)

    tide = Tide.decompose(series.to_numpy(), series.index.to_pydatetime())
    out = np.full((len(constituents), 2), np.nan, dtype=float)
    for i, name in enumerate(constituents):
        for model in tide.model:
            const = model["constituent"]
            const_name = getattr(const, "name", str(const)).upper()
            if const_name == name.upper():
                out[i, 0] = float(model["amplitude"])
                out[i, 1] = float(model["phase"])
                break
    return out


def _decompose_chunk(
    chunk_indices,
    times,
    elevations,
    constituents: Sequence[str],
):
    import numpy as np

    _pin_threads()
    out = np.full((len(chunk_indices), len(constituents), 2), np.nan, dtype=float)
    for j, idx in enumerate(chunk_indices):
        out[j] = _decompose_node(times, elevations[:, idx], constituents)
    return chunk_indices, out


def tidal_analysis(
    start: datetime.datetime | str,
    end: datetime.datetime | str,
    *,
    run: str | None = None,
    zarr_path: pathlib.Path | None = None,
    output: pathlib.Path | None = None,
    constituents: Sequence[str] = tuple(FULL_CONSTITUENTS),
    chunk_size: int = 100,
    n_jobs: int = -1,
    resample_minutes: int = 60,
) -> pathlib.Path:
    """Decompose elevation across the full mesh into harmonic constituents.

    Returns the path to the output netCDF
    (``{run}_tides.nc`` by default).
    """
    import numpy as np
    import xarray as xr

    _pin_threads()

    try:
        from joblib import Parallel, delayed
    except ImportError as exc:
        raise ImportError(
            "joblib is required for tidal-analysis. "
            "Install with: pip install spost[validate]"
        ) from exc

    start_dt = _coerce_dt(start)
    end_dt = _coerce_dt(end)

    zarr = _paths.run_zarr(run, zarr_path)
    if not zarr.exists():
        raise FileNotFoundError(f"Zarr store not found: {zarr}")
    if output is None:
        output = pathlib.Path(f"./{run}_tides.nc") if run else pathlib.Path("./tides.nc")

    ds = xr.open_zarr(zarr, chunks={})
    if "elevation" in ds:
        elev = ds["elevation"]
    elif "elev" in ds:
        elev = ds["elev"]
    else:
        raise KeyError("zarr store does not contain an 'elevation' or 'elev' variable")

    elev = elev.sel(time=slice(start_dt, end_dt))
    if resample_minutes:
        elev = elev.resample(time=f"{resample_minutes}min").mean()

    elev = elev.load()
    times = elev["time"].values
    values = elev.values  # (time, nNodes)
    n_nodes = values.shape[1]

    chunks = [list(range(i, min(i + chunk_size, n_nodes))) for i in range(0, n_nodes, chunk_size)]
    logger.info(
        "tidal-analysis: %d nodes, %d chunks of <=%d, n_jobs=%s",
        n_nodes,
        len(chunks),
        chunk_size,
        n_jobs,
    )

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_decompose_chunk)(chunk, times, values, list(constituents)) for chunk in chunks
    )

    out_array = np.full((n_nodes, len(constituents), 2), np.nan, dtype=float)
    for chunk_indices, chunk_out in results:
        out_array[chunk_indices] = chunk_out

    da = xr.DataArray(
        out_array,
        dims=("nSCHISM_hgrid_node", "constituent", "metric"),
        coords={
            "constituent": list(constituents),
            "metric": ["amplitude", "phase"],
        },
        name="tides",
        attrs={
            "long_name": "Tidal harmonic constituents",
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "resample_minutes": resample_minutes,
        },
    )

    out_ds = da.to_dataset()
    if "SCHISM_hgrid_node_x" in ds:
        out_ds["SCHISM_hgrid_node_x"] = ds["SCHISM_hgrid_node_x"]
    if "SCHISM_hgrid_node_y" in ds:
        out_ds["SCHISM_hgrid_node_y"] = ds["SCHISM_hgrid_node_y"]
    out_ds.to_netcdf(output)
    logger.info("tidal-analysis: wrote %s", output)
    return output
