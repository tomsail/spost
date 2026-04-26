"""Observation retrieval and cleaning.

Wraps ``searvey`` (raw IOC fetching) and ``ioc_cleanup`` (per-station
transformations) and writes one cleaned netCDF per station to
``{output_dir}/obs_ts/{station_code}.nc``.
"""

from __future__ import annotations

import datetime
import logging
import pathlib
from collections.abc import Iterable, Sequence

from . import _paths, _state, cache

logger = logging.getLogger(__name__)

_OBS_DIRNAME = "obs_ts"


def _coerce_dt(value: datetime.datetime | str) -> datetime.datetime:
    if isinstance(value, datetime.datetime):
        return value
    return datetime.datetime.fromisoformat(value)


def _model_station_codes(station_data_path: pathlib.Path) -> set[str]:
    """Extract IOC station codes from spost station output filenames.

    The convention is ``{provider}-{provider_id}.parquet`` (or ``.nc``).
    Only files with provider ``ioc`` (case-insensitive) are kept.
    """
    if not station_data_path.exists():
        logger.warning("Model station directory %s does not exist", station_data_path)
        return set()

    codes: set[str] = set()
    for path in station_data_path.iterdir():
        if path.suffix.lower() not in {".parquet", ".nc", ".zarr"}:
            continue
        stem = path.stem
        provider, _, provider_id = stem.partition("-")
        if not provider_id:
            # Fall back: treat the entire stem as a code.
            codes.add(stem.lower())
            continue
        if provider.lower() == "ioc":
            codes.add(provider_id.lower().replace("_", " ").strip())
    return codes


def _load_meta(meta_parquet: pathlib.Path | None):
    """Return an ``ioc_cleanup`` meta DataFrame.

    Falls through to ``ioc_cleanup.get_meta()`` when no path is given.
    """
    import pandas as pd

    if meta_parquet is not None:
        return pd.read_parquet(meta_parquet)
    try:
        import ioc_cleanup
    except ImportError as exc:
        raise ImportError(
            "ioc_cleanup is required for fetch-obs. "
            "Install with: pip install spost[validate]"
        ) from exc
    meta = ioc_cleanup.get_meta()
    if hasattr(meta, "to_frame") or hasattr(meta, "columns"):
        return meta
    return pd.read_parquet(meta)


def _transformations_dir(transformations_dir: pathlib.Path | None) -> pathlib.Path:
    if transformations_dir is not None:
        return transformations_dir
    try:
        import ioc_cleanup
    except ImportError as exc:
        raise ImportError(
            "ioc_cleanup is required for fetch-obs. "
            "Install with: pip install spost[validate]"
        ) from exc
    return pathlib.Path(ioc_cleanup.get_transformations_dir())


def _fetch_one(
    code: str,
    start: datetime.datetime,
    end: datetime.datetime,
    no_cache: bool,
):
    """Fetch raw IOC data for one station, using the local cache when possible."""
    import xarray as xr

    cache_file = cache.cache_path(code, start, end)
    if not no_cache and cache_file.exists():
        try:
            return xr.open_dataset(cache_file)
        except Exception:
            cache_file.unlink(missing_ok=True)

    try:
        import searvey
    except ImportError as exc:
        raise ImportError(
            "searvey is required for fetch-obs. "
            "Install with: pip install spost[validate]"
        ) from exc

    df = searvey.fetch_ioc_station(code, start, end)
    if df is None or len(df) == 0:
        return None
    ds = df.to_xarray() if hasattr(df, "to_xarray") else xr.Dataset.from_dataframe(df)
    if not no_cache:
        ds.to_netcdf(cache_file)
        cache.write_manifest(cache_file, code, start, end)
    return ds


def _apply_transformation(ds, code: str, transformations_dir: pathlib.Path):
    """Apply the per-station ioc_cleanup recipe. Returns ``None`` when missing."""
    import ioc_cleanup

    json_path = transformations_dir / f"{code}.json"
    if not json_path.exists():
        return None
    if hasattr(ioc_cleanup, "transform"):
        return ioc_cleanup.transform(ds, transformation=json_path)
    if hasattr(ioc_cleanup, "apply_transformation"):
        return ioc_cleanup.apply_transformation(ds, json_path)
    raise RuntimeError(
        f"ioc_cleanup does not expose a transform/apply_transformation entry point "
        f"(version {getattr(ioc_cleanup, '__version__', '?')})"
    )


def _resample(ds, freq: str):
    if freq is None:
        return ds
    if "time" not in ds.dims and "time" not in ds.coords:
        return ds
    return ds.resample(time=freq).mean()


def fetch_obs(
    start: datetime.datetime | str,
    end: datetime.datetime | str,
    *,
    run: str | None = None,
    station_data_path: pathlib.Path | None = None,
    meta_parquet: pathlib.Path | None = None,
    transformations_dir: pathlib.Path | None = None,
    output_dir: pathlib.Path | None = None,
    resample: str = "1h",
    no_cache: bool = False,
    force: bool = False,
    stations: Sequence[str] | None = None,
) -> pathlib.Path:
    """Fetch and clean IOC observations for the given window.

    Returns the output directory containing the per-station netCDFs.
    """
    start_dt = _coerce_dt(start)
    end_dt = _coerce_dt(end)

    out_dir = _paths.run_validation_dir(run, output_dir)
    obs_dir = out_dir / _OBS_DIRNAME
    obs_dir.mkdir(parents=True, exist_ok=True)

    state = _state.load(out_dir)
    prev_end = _state.previous_window_end(state) if not force else None

    station_dir = _paths.run_station_data(run, station_data_path)
    model_codes = _model_station_codes(station_dir)
    if not model_codes:
        logger.warning(
            "No model stations found under %s — observations will still be fetched "
            "but no intersection filtering can be applied.",
            station_dir,
        )

    meta = _load_meta(meta_parquet)
    trans_dir = _transformations_dir(transformations_dir)

    if "ioc_code" in getattr(meta, "columns", []):
        meta_codes = {str(c).lower() for c in meta["ioc_code"].tolist()}
    elif "code" in getattr(meta, "columns", []):
        meta_codes = {str(c).lower() for c in meta["code"].tolist()}
    else:
        try:
            meta_codes = {str(c).lower() for c in meta.index.tolist()}
        except Exception:
            meta_codes = set()

    candidate = sorted(meta_codes & model_codes) if model_codes else sorted(meta_codes)
    if stations:
        wanted = {s.lower() for s in stations}
        candidate = [c for c in candidate if c in wanted]

    processed: list[str] = []
    skipped: list[str] = []

    for code in candidate:
        out_file = obs_dir / f"{code}.nc"

        # Incremental: fetch only the new tail when we already have the file.
        fetch_start = start_dt
        if out_file.exists() and prev_end is not None and not force:
            fetch_start = prev_end

        try:
            ds = _fetch_one(code, fetch_start, end_dt, no_cache=no_cache)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", code, exc)
            skipped.append(code)
            continue
        if ds is None:
            skipped.append(code)
            continue

        ds_clean = _apply_transformation(ds, code, trans_dir)
        if ds_clean is None:
            logger.info("No transformation for %s — skipping", code)
            skipped.append(code)
            continue

        ds_clean = _resample(ds_clean, resample)

        if out_file.exists() and prev_end is not None and not force:
            try:
                import xarray as xr

                existing = xr.open_dataset(out_file)
                ds_clean = xr.concat(
                    [existing.sel(time=slice(None, prev_end)), ds_clean],
                    dim="time",
                ).drop_duplicates("time")
                existing.close()
            except Exception as exc:
                logger.warning("Could not append to %s; rewriting (%s)", out_file, exc)

        # netCDF4 cannot overwrite while a handle is open; write to a temp first.
        tmp = out_file.with_suffix(".nc.tmp")
        ds_clean.to_netcdf(tmp)
        tmp.replace(out_file)
        processed.append(code)

    state.start = start_dt.isoformat()
    state.end = end_dt.isoformat()
    state.stations_processed = sorted(set(state.stations_processed) | set(processed))
    state.stations_skipped = sorted(set(skipped))
    _state.save(out_dir, state)

    if skipped:
        logger.info(
            "fetch-obs: processed %d stations, skipped %d (no transform / no data)",
            len(processed),
            len(skipped),
        )
    return obs_dir


def list_obs_files(obs_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    if not obs_dir.exists():
        return []
    return sorted(p for p in obs_dir.iterdir() if p.suffix == ".nc")
