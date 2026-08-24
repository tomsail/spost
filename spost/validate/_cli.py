"""Cyclopts subcommands for the validation pipeline.

Command callables are named ``<stage>_cmd`` and registered under the ``skill``
app with explicit hyphenated names (``fetch-obs``, ``compare``, ``tidal``,
``report``). Each body lazily imports its implementation from
:mod:`spost.validate` so the heavy ``validate`` extra is only required when a
command is actually executed.
"""
from __future__ import annotations

import datetime
import os
import pathlib
import platform
from typing import Annotated

import platformdirs
from cyclopts import Parameter

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


def fetch_obs_cmd(
    *,
    run: str | None = None,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    meta_parquet: pathlib.Path | None = None,
    transformations_dir: pathlib.Path | None = None,
    clean_data_folder: pathlib.Path | None = None,
    output_path: pathlib.Path | None = None,
    no_cache: Annotated[bool, Parameter(negative=())] = False,
    force: Annotated[bool, Parameter(negative=())] = False,
):
    """Download IOC observations and apply ``ioc_cleanup`` transformations.

    Parameters
    ----------
    run
        Run identifier used for default path resolution.
    start, end
        Observation window (ISO-8601).
    meta_parquet
        Station metadata parquet. Defaults to ``ioc_cleanup.get_meta()``.
    transformations_dir
        Directory of cleanup transformations. Defaults to
        ``ioc_cleanup.get_transformations_dir()``.
    clean_data_folder
        Output directory for cleaned observation parquet files.
    output_path
        Base validation output directory.
    no_cache
        Bypass the on-disk IOC raw-data cache.
    force
        Ignore prior state and reprocess everything.
    """
    from spost.validate import fetch_obs

    fetch_obs(
        run=run,
        start=start,
        end=end,
        meta_parquet=meta_parquet,
        transformations_dir=transformations_dir,
        clean_data_folder=clean_data_folder,
        output_path=output_path,
        no_cache=no_cache,
        force=force,
    )


def compare_cmd(
    *,
    run: str | None = None,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    zarr_path: pathlib.Path | None = None,
    station_data_path: pathlib.Path | None = None,
    obs_dir: pathlib.Path | None = None,
    output_path: pathlib.Path | None = None,
    variables: Annotated[list[str], Parameter(consume_multiple=True, negative=())] = ["elev"],
    spinup_days: int = 0,
    overwrite: Annotated[bool, Parameter(negative=())] = False,
):
    """Align model and obs and compute ``seastats`` skill metrics.

    Parameters
    ----------
    run
        Run identifier used for default path resolution.
    start, end
        Comparison window (ISO-8601).
    zarr_path
        Model zarr store. Defaults to ``./{run}.zarr``.
    station_data_path
        Directory of extracted station parquet files.
    obs_dir
        Directory of cleaned observation parquet files.
    output_path
        Directory for comparison outputs (aligned time-series, metrics parquet).
    variables
        Variable names to compare (e.g. ``elev``).
    spinup_days
        Number of leading days to drop before scoring.
    overwrite
        Re-run comparison even if outputs exist.
    """
    from spost.validate import compare

    compare(
        run=run,
        start=start,
        end=end,
        zarr_path=zarr_path,
        station_data_path=station_data_path,
        obs_dir=obs_dir,
        output_path=output_path,
        variables=tuple(variables),
        spinup_days=spinup_days,
        overwrite=overwrite,
    )


def tidal_cmd(
    *,
    run: str | None = None,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    zarr_path: pathlib.Path | None = None,
    output: pathlib.Path | None = None,
    constituents: Annotated[list[str], Parameter(consume_multiple=True, negative=())] | None = None,
    chunk_size: int = 100,
    n_jobs: int = -1,
    resample_minutes: int = 60,
):
    """Decompose elevation across the full mesh into tidal constituents.

    Parameters
    ----------
    run
        Run identifier used for default path resolution.
    start, end
        Analysis window (ISO-8601).
    zarr_path
        Model zarr store. Defaults to ``./{run}.zarr``.
    output
        Output netCDF/zarr path for tidal coefficients.
    constituents
        Constituents to solve for. Defaults to the full constituent set.
    chunk_size
        Number of nodes per joblib chunk.
    n_jobs
        Parallel worker count (``-1`` uses all cores).
    resample_minutes
        Resampling interval before harmonic analysis.
    """
    from spost.validate import tidal_analysis

    tidal_analysis(
        run=run,
        start=start,
        end=end,
        zarr_path=zarr_path,
        output=output,
        constituents=tuple(constituents) if constituents else None,
        chunk_size=chunk_size,
        n_jobs=n_jobs,
        resample_minutes=resample_minutes,
    )


def report_cmd(
    *,
    run: str | None = None,
    output_path: pathlib.Path | None = None,
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
    """Render the validation report (HTML, PDF, or both).

    Parameters
    ----------
    run
        Run identifier used for default path resolution.
    output_path
        Base validation output directory.
    metrics_parquet
        Metrics parquet produced by ``compare``.
    station_data_path
        Directory of extracted station parquet files.
    obs_dir
        Directory of cleaned observation parquet files.
    tides_nc
        Optional tidal-maps netCDF to embed in the report.
    format
        Report format: ``html``, ``pdf``, or ``both``.
    reference_metrics
        Optional reference metrics parquet for comparison.
    name
        Optional report/run display name.
    include_timeseries, include_scatter, include_taylor, include_tidal_maps,
    include_map, include_summary
        Toggle individual report sections (disable with ``--no-include-*``).
    """
    from spost.validate import report

    report(
        run=run,
        output_path=output_path,
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
    zarr_path: pathlib.Path | None = None,
    clean_data_folder: pathlib.Path | None = None,
    meta_parquet: pathlib.Path | None = None,
    transformations_dir: pathlib.Path | None = None,
    station_data_path: pathlib.Path | None = None,
    output_path: pathlib.Path | None = None,
    variables: Annotated[list[str], Parameter(consume_multiple=True, negative=())] = ["elev"],
    spinup_days: int = 0,
    report_format: str = "html",
    reference_metrics: pathlib.Path | None = None,
    name: str | None = None,
    force: Annotated[bool, Parameter(negative=())] = False,
):
    """Run the full validation pipeline (fetch-obs -> compare -> report).

    ``tidal`` is intentionally excluded - it's a heavier standalone computation.

    Parameters
    ----------
    run
        Run identifier used for default path resolution.
    start, end
        Validation window (ISO-8601).
    zarr_path
        Model zarr store. Defaults to ``./{run}.zarr``.
    clean_data_folder
        Output directory for cleaned observation parquet files.
    meta_parquet
        Station metadata parquet. Defaults to ``ioc_cleanup.get_meta()``.
    transformations_dir
        Directory of cleanup transformations.
    station_data_path
        Directory of extracted station parquet files.
    output_path
        Base validation output directory.
    variables
        Variable names to validate (e.g. ``elev``).
    spinup_days
        Number of leading days to drop before scoring.
    report_format
        Report format: ``html``, ``pdf``, or ``both``.
    reference_metrics
        Optional reference metrics parquet for comparison.
    name
        Optional report/run display name.
    force
        Ignore prior state and reprocess everything.
    """
    fetch_obs_cmd(
        run=run,
        start=start,
        end=end,
        meta_parquet=meta_parquet,
        transformations_dir=transformations_dir,
        clean_data_folder=clean_data_folder,
        output_path=output_path,
        force=force,
    )
    compare_cmd(
        run=run,
        start=start,
        end=end,
        zarr_path=zarr_path,
        station_data_path=station_data_path,
        output_path=output_path,
        variables=variables,
        spinup_days=spinup_days,
        overwrite=force,
    )
    report_cmd(
        run=run,
        output_path=output_path,
        station_data_path=station_data_path,
        format=report_format,
        reference_metrics=reference_metrics,
        name=name,
    )
