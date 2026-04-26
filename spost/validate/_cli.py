"""Cyclopts subcommands for the validation pipeline.

Apps are exposed as standalone ``cyclopts.App`` instances so the parent CLI can
register them at the top level (matching the spec: ``spost validate``,
``spost fetch-obs``, ``spost compare``, ``spost report``,
``spost tidal-analysis``).
"""

from __future__ import annotations

import datetime
import pathlib
from typing import Annotated

import cyclopts


validate_app = cyclopts.App(
    name="validate",
    help="Run the full validation pipeline (fetch-obs → compare → report).",
)
fetch_obs_app = cyclopts.App(
    name="fetch-obs",
    help="Fetch IOC observations and apply ioc_cleanup transformations.",
)
compare_app = cyclopts.App(
    name="compare",
    help="Compute skill metrics by aligning model output to cleaned observations.",
)
report_app = cyclopts.App(
    name="report",
    help="Render the HTML/PDF validation report.",
)
tidal_app = cyclopts.App(
    name="tidal-analysis",
    help="Full-mesh tidal harmonic decomposition (pytides2 + joblib).",
)


_Subcommands = (validate_app, fetch_obs_app, compare_app, report_app, tidal_app)


def _split_csv(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [v.strip() for v in value if v]
    return [v.strip() for v in value.split(",") if v.strip()]


@validate_app.default
def validate(
    *,
    run: str | None = None,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    station_data_path: pathlib.Path | None = None,
    meta_parquet: pathlib.Path | None = None,
    transformations_dir: pathlib.Path | None = None,
    output_dir: pathlib.Path | None = None,
    variables: Annotated[list[str], cyclopts.Parameter(consume_multiple=True)] = ["elev"],
    resample: str = "1h",
    spinup_days: int = 0,
    report_format: str = "html",
    reference_metrics: pathlib.Path | None = None,
    name: str | None = None,
    force: bool = False,
):
    """Run the full validation pipeline.

    Equivalent to running ``fetch-obs``, ``compare`` and ``report`` in
    sequence. ``tidal-analysis`` is intentionally excluded — it's a heavier
    standalone computation.
    """
    if run is None or start is None or end is None:
        validate_app.help_print()
        raise SystemExit(0)

    from spost.validate import compare as _compare
    from spost.validate import fetch_obs as _fetch_obs
    from spost.validate import report as _report

    _fetch_obs(
        start=start,
        end=end,
        run=run,
        station_data_path=station_data_path,
        meta_parquet=meta_parquet,
        transformations_dir=transformations_dir,
        output_dir=output_dir,
        resample=resample,
        force=force,
    )
    _compare(
        start=start,
        end=end,
        run=run,
        station_data_path=station_data_path,
        output_dir=output_dir,
        spinup_days=spinup_days,
        resample=resample,
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


@fetch_obs_app.default
def fetch_obs_cmd(
    *,
    run: str | None = None,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    station_data_path: pathlib.Path | None = None,
    meta_parquet: pathlib.Path | None = None,
    transformations_dir: pathlib.Path | None = None,
    output_dir: pathlib.Path | None = None,
    resample: str = "1h",
    no_cache: bool = False,
    force: bool = False,
):
    """Fetch IOC observations and apply per-station transformations."""
    if start is None or end is None:
        fetch_obs_app.help_print()
        raise SystemExit(0)

    from spost.validate import fetch_obs as _fetch_obs

    _fetch_obs(
        start=start,
        end=end,
        run=run,
        station_data_path=station_data_path,
        meta_parquet=meta_parquet,
        transformations_dir=transformations_dir,
        output_dir=output_dir,
        resample=resample,
        no_cache=no_cache,
        force=force,
    )


@compare_app.default
def compare_cmd(
    *,
    run: str | None = None,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    station_data_path: pathlib.Path | None = None,
    obs_dir: pathlib.Path | None = None,
    output_dir: pathlib.Path | None = None,
    spinup_days: int = 0,
    resample: str = "1h",
    variables: Annotated[list[str], cyclopts.Parameter(consume_multiple=True)] = ["elev"],
):
    """Align model and obs and compute seastats skill metrics."""
    if start is None or end is None:
        compare_app.help_print()
        raise SystemExit(0)

    from spost.validate import compare as _compare

    _compare(
        start=start,
        end=end,
        run=run,
        station_data_path=station_data_path,
        obs_dir=obs_dir,
        output_dir=output_dir,
        spinup_days=spinup_days,
        resample=resample,
        variables=tuple(variables),
    )


@tidal_app.default
def tidal_cmd(
    *,
    run: str | None = None,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    zarr_path: pathlib.Path | None = None,
    output: pathlib.Path | None = None,
    constituents: Annotated[list[str], cyclopts.Parameter(consume_multiple=True)] | None = None,
    chunk_size: int = 100,
    n_jobs: int = -1,
    resample_minutes: int = 60,
):
    """Decompose elevation across the full mesh into tidal constituents."""
    if start is None or end is None or (run is None and zarr_path is None):
        tidal_app.help_print()
        raise SystemExit(0)

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


@report_app.default
def report_cmd(
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
