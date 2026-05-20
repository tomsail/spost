"""Cyclopts subcommands for the validation pipeline.

Apps are exposed as standalone ``cyclopts.App`` instances so the parent CLI can
register them at the top level (matching the spec: ``spost validate``,
``spost fetch-obs``, ``spost compare``, ``spost report``,
``spost tidal-analysis``).
"""
from __future__ import annotations

import datetime
import os
import pathlib
import platform
from typing import Annotated

import platformdirs
from cyclopts import Parameter
from cyclopts.types import ResolvedDirectory

_HOSTNAME = platform.node()

if "meluxina" in _HOSTNAME:
    _GROUP_ID = os.getgroups()[-1]
    _PROJECT_DIR = pathlib.Path(f"/project/home/p{_GROUP_ID}")
    _SCRATCH_DIR = pathlib.Path(f"/project/scratch/p{_GROUP_ID}")
    _default_cache_dir = _SCRATCH_DIR / "cache"
    _default_obs_dir = _PROJECT_DIR / "01_obs/ioc_cleanup/"
else:
    _default_cache_dir = platformdirs.user_cache_path()
    _default_obs_dir = pathlib.Path(os.environ.get("IOC_CLEANUP_DIR", "ioc_cleanup"))

_DEFAULT_CACHE_DIR = _default_cache_dir
_DEFAULT_OBS_DIR = _default_obs_dir


def fetch_obs(
    *,
    transformations_dir: pathlib.Path,
    raw_data_folder: ResolvedDirectory = _DEFAULT_OBS_DIR / "raw",
    output_dir: ResolvedDirectory = _DEFAULT_OBS_DIR / "clean",
    overwrite: bool = False,
):
    """
    Fetch IOC observations and apply per-station transformations.

    Parameters
    ----------
    transformations_dir
        Optional path to directory containing per-station transformation JSON files.
        If not provided, will attempt to load from the default transformations directory
    raw_data_folder
        Optional path to directory for caching raw IOC data. Defaults to platform specific paths.
        If not provided, will attempt to load from the default raw data directory.
        Raw data is cached in `_DEFAULT_OBS_DIR/raw`.
    output_dir
        Optional path to directory for storing cleaned IOC. Defaults to platform specific paths.
        If not provided, will attempt to load from the default clean data directory.
        Cleaned data is stored in `_DEFAULT_OBS_DIR/clean`.
    overwrite
        If True, ignore any existing cached observations and re-fetch from IOC. Default False.
    """

    from spost.validate import fetch_obs as _fetch_obs


    _fetch_obs(
        transformations_dir=transformations_dir,
        raw_data_folder=raw_data_folder,
        output_dir=output_dir,
        overwrite=overwrite,
    )


def compare(
    *,
    run: str | None = None,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    station_data_path: pathlib.Path | None = None,
    obs_dir: pathlib.Path | None = None,
    output_dir: pathlib.Path | None = None,
    spinup_days: int = 0,
    variables: Annotated[list[str], Parameter(consume_multiple=True)] = ["elev"],
):
    """Align model and obs and compute seastats skill metrics."""

    from spost.validate import compare as _compare

    _compare(
        start=start,
        end=end,
        run=run,
        station_data_path=station_data_path,
        obs_dir=obs_dir,
        output_dir=output_dir,
        spinup_days=spinup_days,
        variables=tuple(variables),
    )


def tidal(
    *,
    run: str | None = None,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    zarr_path: pathlib.Path | None = None,
    output: pathlib.Path | None = None,
    constituents: Annotated[list[str], Parameter(consume_multiple=True)] | None = None,
    chunk_size: int = 100,
    n_jobs: int = -1,
    resample_minutes: int = 60,
):
    """Decompose elevation across the full mesh into tidal constituents."""

    from spost.validate import tidal_analysis as _tidal
    from spost.validate.tidal_analysis import FULL_CONSTITUENTS

    _tidal(
        start=start,
        end=end,
        run=run,
        zarr_path=zarr_path,
        output=output,
        constituents=tuple(constituents) if constituents else tuple(FULL_CONSTITUENTS),
        chunk_size=chunk_size,
        n_jobs=n_jobs,
        resample_minutes=resample_minutes,
    )


def report(
    *,
    run: str | None = None,
    output_dir: pathlib.Path | None = None,
    metrics_parquet: pathlib.Path | None = None,
    station_data_path: pathlib.Path | None = None,
    obs_dir: pathlib.Path | None = None,
    tides_nc: pathlib.Path | None = None,
    format: str = "html",
    reference_metrics: pathlib.Path | None = None,
    name: str | None = None,
    include_timeseries: bool = True,
    include_scatter: bool = True,
    include_taylor: bool = True,
    include_tidal_maps: bool = True,
    include_map: bool = True,
    include_summary: bool = True,
):
    """Render the validation report (HTML, PDF, or both)."""
    from spost.validate import report as _report

    _report(
        run=run,
        output_dir=output_dir,
        metrics_parquet=metrics_parquet,
        station_data_path=station_data_path,
        obs_dir=obs_dir,
        tides_nc=tides_nc,
        format=format,
        reference_metrics=reference_metrics,
        name=name,
        include_timeseries_plots=include_timeseries,
        include_scatter_plots=include_scatter,
        include_taylor_diagram=include_taylor,
        include_tidal_maps=include_tidal_maps,
        include_map=include_map,
        include_summary_table=include_summary,
    )


def validate(
    *,
    run: str | None = None,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    clean_data_folder: pathlib.Path | None = None,
    meta_parquet: pathlib.Path | None = None,
    transformations_dir: pathlib.Path | None = None,
    output_dir: pathlib.Path | None = None,
    variables: Annotated[list[str], Parameter(consume_multiple=True)] = ["elev"],
    spinup_days: int = 0,
    report_format: str = "html",
    reference_metrics: pathlib.Path | None = None,
    name: str | None = None,
    overwrite: bool = False,
):
    """Run the full validation station pipeline (fetch-obs -> compare -> report).

    Equivalent to running ``fetch-obs``, ``compare`` and ``report`` in
    sequence. ``tidal-analysis`` is intentionally excluded - it's a heavier
    standalone computation.
    """

    from spost.validate import compare as _compare
    from spost.validate import fetch_obs as _fetch_obs
    from spost.validate import report as _report

    _fetch_obs(
        start=start,
        end=end,
        clean_data_folder=clean_data_folder,
        transformations_dir=transformations_dir,
        overwrite=overwrite,
    )
    _compare(
        start=start,
        end=end,
        run=run,
        station_data_path=station_data_path,
        output_dir=output_dir,
        spinup_days=spinup_days,
        variables=tuple(variables),
    )
    _report(
        run=run,
        output_dir=output_dir,
        station_data_path=station_data_path,
        format=report_format,
        reference_metrics=reference_metrics,
        name=name,
    )
