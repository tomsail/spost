"""Observation retrieval and cleaning.

Wraps ``searvey`` (raw IOC fetching) and ``ioc_cleanup`` (per-station
transformations) and writes one cleaned netCDF per station to
``{output_dir}/obs_ts/{station_code}.nc``.
"""
from __future__ import annotations

import datetime
import logging
import pathlib
from collections.abc import Iterable
from collections.abc import Sequence

from . import _paths
from . import cache

logger = logging.getLogger(__name__)


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


def _fetch_one(
    code: str,
    start: datetime.datetime,
    end: datetime.datetime,
):
    """Fetch raw IOC data for one station, using the local cache when possible."""
    import pandas as pd

    cache_file = cache.cache_path(code, start, end)
    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
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
        return pd.DataFrame()  # empty but valid

    return df


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


def fetch_obs(
    start: datetime.datetime | str,
    end: datetime.datetime | str,
    *,
    run: str | None = None,
    station_data_path: pathlib.Path | None = None,
    meta_parquet: pathlib.Path | None = None,
    transformations_dir: pathlib.Path | None = None,
    force: bool = False,
    stations: Sequence[str] | None = None,
) -> pathlib.Path:
    """Fetch and clean IOC observations for the given window.

    Returns the output directory containing the per-station parquet files.
    """
    start_dt = _coerce_dt(start)
    end_dt = _coerce_dt(end)

    print(start_dt, end_dt)

    print(station_data_path)

    station_dir = _paths.get_station_data(station_data_path)
    model_codes = _model_station_codes(station_dir)
    if not model_codes:
        logger.warning(
            "No model stations found under %s - observations will still be fetched "
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
            ds = _fetch_one(code, fetch_start, end_dt)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", code, exc)
            skipped.append(code)
            continue
        if ds is None:
            skipped.append(code)
            continue

        ds_clean = _apply_transformation(ds, code, trans_dir)
        if ds_clean is None:
            logger.info("No transformation for %s - skipping", code)
            skipped.append(code)
            continue


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
