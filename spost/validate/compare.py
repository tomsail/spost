"""Skill metric computation by aligning model and observed timeseries.

Per station we:

1. Load the spost model timeseries (parquet/netCDF/zarr).
2. Load the cleaned observations from ``fetch-obs`` output.
3. Clip to ``[start + spinup_days, end]`` and align on a common time axis.
4. Hand the aligned series to ``seastats`` for the metric suite.

Outputs:

- ``{output_dir}/metrics.parquet`` — one row per station, columns are metrics.
- ``{output_dir}/per_station/{code}_comparison.nc`` — aligned model vs obs.
"""

from __future__ import annotations

import datetime
import logging
import pathlib
from collections.abc import Sequence

from . import _paths, _state

logger = logging.getLogger(__name__)

_PER_STATION_DIR = "per_station"
_METRICS_FILE = "metrics.parquet"
_OBS_DIR = "obs_ts"


def _coerce_dt(value: datetime.datetime | str) -> datetime.datetime:
    if isinstance(value, datetime.datetime):
        return value
    return datetime.datetime.fromisoformat(value)


def _load_model_series(path: pathlib.Path, variable: str):
    """Read a station model timeseries and return a ``pd.Series`` indexed by time."""
    import pandas as pd
    import xarray as xr

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        if variable in df.columns:
            s = df[variable]
        else:
            s = df.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        return s.rename(variable)

    if path.suffix in {".nc", ".zarr"}:
        ds = xr.open_dataset(path) if path.suffix == ".nc" else xr.open_zarr(path)
        if variable in ds:
            da = ds[variable]
        else:
            # Best-effort: pick the first data variable.
            da = ds[next(iter(ds.data_vars))]
        return da.to_pandas().rename(variable)

    raise ValueError(f"Unsupported model station file: {path}")


def _load_obs_series(path: pathlib.Path, variable: str):
    import xarray as xr

    ds = xr.open_dataset(path)
    if variable in ds:
        da = ds[variable]
    else:
        da = ds[next(iter(ds.data_vars))]
    return da.to_pandas().rename(variable)


def _align(model_s, obs_s, freq: str):
    import pandas as pd

    if freq:
        model_s = model_s.resample(freq).mean()
        obs_s = obs_s.resample(freq).mean()
    df = pd.concat({"model": model_s, "obs": obs_s}, axis=1).dropna()
    return df


def _seastats_metrics(model_s, obs_s) -> dict:
    """Delegate the metric suite to seastats with a graceful fallback."""
    try:
        import seastats
    except ImportError:
        return _basic_metrics(model_s, obs_s)

    for fn_name in ("get_stats", "compute_stats", "stats"):
        if hasattr(seastats, fn_name):
            try:
                result = getattr(seastats, fn_name)(model_s, obs_s)
                if hasattr(result, "to_dict"):
                    return dict(result.to_dict())
                if isinstance(result, dict):
                    return result
            except TypeError:
                # Some signatures expect (obs, model); try the swap.
                try:
                    result = getattr(seastats, fn_name)(obs_s, model_s)
                    if hasattr(result, "to_dict"):
                        return dict(result.to_dict())
                    if isinstance(result, dict):
                        return result
                except Exception as exc:
                    logger.debug("seastats.%s failed: %s", fn_name, exc)
    return _basic_metrics(model_s, obs_s)


def _basic_metrics(model_s, obs_s) -> dict:
    import numpy as np

    m = model_s.to_numpy()
    o = obs_s.to_numpy()
    n = len(m)
    if n == 0:
        return {"n": 0}
    bias = float((m - o).mean())
    rmse = float(np.sqrt(((m - o) ** 2).mean()))
    mae = float(np.abs(m - o).mean())
    if o.std() > 0 and m.std() > 0:
        corr = float(np.corrcoef(m, o)[0, 1])
    else:
        corr = float("nan")
    return {"n": n, "bias": bias, "rmse": rmse, "mae": mae, "corr": corr}


def _model_file_for_code(station_dir: pathlib.Path, code: str) -> pathlib.Path | None:
    """Find the spost station file matching an IOC code."""
    code_norm = code.lower().replace(" ", "_")
    for path in station_dir.iterdir():
        stem = path.stem.lower()
        # Strip provider prefix (``ioc-``) if present.
        stripped = stem.split("-", 1)[1] if "-" in stem else stem
        if stripped == code_norm or stem == code or stripped.replace("_", " ") == code:
            return path
    return None


def compare(
    start: datetime.datetime | str,
    end: datetime.datetime | str,
    *,
    run: str | None = None,
    station_data_path: pathlib.Path | None = None,
    obs_dir: pathlib.Path | None = None,
    output_dir: pathlib.Path | None = None,
    spinup_days: int = 0,
    resample: str = "1h",
    variables: Sequence[str] = ("elev",),
) -> pathlib.Path:
    """Compute skill metrics for every station with both model and obs data.

    Returns the path to the ``metrics.parquet`` file.
    """
    import pandas as pd

    start_dt = _coerce_dt(start)
    end_dt = _coerce_dt(end)
    if spinup_days:
        start_dt = start_dt + datetime.timedelta(days=spinup_days)

    out_dir = _paths.run_validation_dir(run, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_d = obs_dir if obs_dir is not None else out_dir / _OBS_DIR
    per_station = out_dir / _PER_STATION_DIR
    per_station.mkdir(parents=True, exist_ok=True)

    primary_var = variables[0]
    station_dir = _paths.run_station_data(run, station_data_path, variable=primary_var)

    rows: list[dict] = []
    for obs_file in sorted(obs_d.glob("*.nc")):
        code = obs_file.stem
        model_file = _model_file_for_code(station_dir, code)
        if model_file is None:
            logger.info("No model file for %s — skipping", code)
            continue
        try:
            model_s = _load_model_series(model_file, primary_var)
            obs_s = _load_obs_series(obs_file, primary_var)
        except Exception as exc:
            logger.warning("Could not load %s: %s", code, exc)
            continue

        model_s = model_s.loc[start_dt:end_dt]
        obs_s = obs_s.loc[start_dt:end_dt]
        aligned = _align(model_s, obs_s, resample)
        if aligned.empty:
            logger.info("Empty overlap for %s — skipping", code)
            continue

        metrics = _seastats_metrics(aligned["model"], aligned["obs"])
        metrics_row = {"station": code, **metrics}
        rows.append(metrics_row)

        # Save per-station aligned dataset for use by the report.
        comparison = aligned.to_xarray()
        comparison.attrs["station"] = code
        comparison.to_netcdf(per_station / f"{code}_comparison.nc")

    metrics_df = pd.DataFrame(rows).set_index("station") if rows else pd.DataFrame()
    metrics_path = out_dir / _METRICS_FILE
    metrics_df.to_parquet(metrics_path)

    state = _state.load(out_dir)
    state.start = start_dt.isoformat()
    state.end = end_dt.isoformat()
    _state.save(out_dir, state)

    logger.info("compare: wrote metrics for %d stations to %s", len(rows), metrics_path)
    return metrics_path
