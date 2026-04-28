"""CLI argument-parsing tests.

Follows the cyclopts unit-testing pattern: ``app.parse_args(tokens)`` returns
``(callable, BoundArguments, ignored)`` so we can assert on the resolved
function and its bound arguments without actually executing the command body.
"""

from __future__ import annotations

import datetime
import pathlib

import pytest

from spost._cli import app


# ---------------------------------------------------------------------------
# top-level shape
# ---------------------------------------------------------------------------


def test_top_level_groups_present():
    """Top level stays at three groups."""
    commands = set(app)
    assert "extract" in commands
    assert "plot" in commands
    assert "skill" in commands


def test_skill_hosts_validation_pipeline():
    """All five validation stages live under ``skill``."""
    skill = app["skill"]
    children = set(skill)
    assert {"validate", "fetch-obs", "compare", "report", "tidal"} <= children


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


def test_extract_to_zarr_parses(tmp_path: pathlib.Path):
    in_dir = tmp_path / "schism_run"
    in_dir.mkdir()
    out_path = tmp_path / "out.zarr"

    cmd, bound, _ = app.parse_args(
        [
            "extract",
            "to-zarr",
            "--input-path",
            str(in_dir),
            "--output",
            str(out_path),
            "--variables",
            "elevation",
            "--workers",
            "2",
            "--clevel",
            "5",
            "--overwrite",
            "--exclude-last",
            "1",
        ]
    )

    assert cmd.__name__ == "to_zarr"
    kw = bound.arguments
    assert kw["input_path"] == in_dir
    assert kw["output"] == out_path
    assert kw["variables"] == ["elevation"]
    assert kw["workers"] == 2
    assert kw["clevel"] == 5
    assert kw["overwrite"] is True
    assert kw["exclude_last"] == 1


def test_extract_clip_parses_named_region(tmp_path: pathlib.Path):
    src = tmp_path / "in.zarr"
    src.mkdir()
    cmd, bound, _ = app.parse_args(
        [
            "extract",
            "clip",
            "--input-path",
            str(src),
            "--bbox",
            "med",
        ]
    )
    assert cmd.__name__ == "clip"
    # The bbox converter resolves named regions to a 4-tuple.
    assert bound.arguments["bbox"] == (-5.5, 30.0, 42.0, 47.5)


def test_extract_stations_parses(tmp_path: pathlib.Path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    out_a.mkdir()
    out_b.mkdir()

    cmd, bound, _ = app.parse_args(
        [
            "extract",
            "stations",
            str(out_a),
            str(out_b),
            "--output-path",
            str(tmp_path / "stations"),
            "--staout-indices",
            "1",
            "2",
        ]
    )
    assert cmd.__name__ == "stations"
    args = bound.args  # *outputs_dirs is positional / variadic
    assert tuple(args) == (out_a, out_b)
    assert bound.kwargs["staout_indices"] == [1, 2]


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def test_plot_to_pngs_parses(tmp_path: pathlib.Path):
    src = tmp_path / "x.zarr"
    src.mkdir()
    cmd, bound, _ = app.parse_args(
        [
            "plot",
            "to-pngs",
            "--input-path",
            str(src),
            "--variable",
            "elevation",
            "--width",
            "640",
            "--height",
            "360",
        ]
    )
    assert cmd.__name__ == "to_pngs"
    assert bound.arguments["variable"] == "elevation"
    assert bound.arguments["width"] == 640
    assert bound.arguments["height"] == 360


# ---------------------------------------------------------------------------
# skill (validation pipeline)
# ---------------------------------------------------------------------------


def test_skill_validate_parses_window_and_run():
    cmd, bound, _ = app.parse_args(
        [
            "skill",
            "validate",
            "--run",
            "100",
            "--start",
            "2020-01-01",
            "--end",
            "2020-12-31",
            "--spinup-days",
            "5",
        ]
    )
    assert cmd.__name__ == "validate"
    bound.apply_defaults()
    assert bound.arguments["run"] == "100"
    assert bound.arguments["start"] == datetime.datetime(2020, 1, 1)
    assert bound.arguments["end"] == datetime.datetime(2020, 12, 31)
    assert bound.arguments["spinup_days"] == 5
    # Defaults flow through.
    assert bound.arguments["resample"] == "1h"
    assert bound.arguments["report_format"] == "html"
    assert bound.arguments["force"] is False


def test_skill_fetch_obs_parses_with_overrides(tmp_path: pathlib.Path):
    meta = tmp_path / "meta.parquet"
    meta.touch()
    trans = tmp_path / "transformations"
    trans.mkdir()

    cmd, bound, _ = app.parse_args(
        [
            "skill",
            "fetch-obs",
            "--start",
            "2020-01-01",
            "--end",
            "2020-02-01",
            "--meta-parquet",
            str(meta),
            "--transformations-dir",
            str(trans),
            "--no-cache",
        ]
    )
    assert cmd.__name__ == "fetch_obs_cmd"
    assert bound.arguments["meta_parquet"] == meta
    assert bound.arguments["transformations_dir"] == trans
    assert bound.arguments["no_cache"] is True


def test_skill_compare_parses_variables_list():
    cmd, bound, _ = app.parse_args(
        [
            "skill",
            "compare",
            "--run",
            "100",
            "--start",
            "2020-01-01",
            "--end",
            "2020-06-01",
            "--variables",
            "elev",
            "--spinup-days",
            "3",
        ]
    )
    assert cmd.__name__ == "compare_cmd"
    assert bound.arguments["variables"] == ["elev"]
    assert bound.arguments["spinup_days"] == 3


def test_skill_tidal_parses_constituents_and_chunking():
    cmd, bound, _ = app.parse_args(
        [
            "skill",
            "tidal",
            "--run",
            "100",
            "--start",
            "2020-01-01",
            "--end",
            "2021-01-01",
            "--constituents",
            "M2",
            "S2",
            "K1",
            "--chunk-size",
            "200",
            "--n-jobs",
            "8",
            "--resample-minutes",
            "30",
        ]
    )
    assert cmd.__name__ == "tidal_cmd"
    assert bound.arguments["constituents"] == ["M2", "S2", "K1"]
    assert bound.arguments["chunk_size"] == 200
    assert bound.arguments["n_jobs"] == 8
    assert bound.arguments["resample_minutes"] == 30


def test_skill_report_parses_include_flags(tmp_path: pathlib.Path):
    cmd, bound, _ = app.parse_args(
        [
            "skill",
            "report",
            "--run",
            "100",
            "--format",
            "html",
            "--no-include-tidal-maps",
            "--no-include-map",
        ]
    )
    assert cmd.__name__ == "report_cmd"
    bound.apply_defaults()
    assert bound.arguments["include_tidal_maps"] is False
    assert bound.arguments["include_map"] is False
    # Other includes default to True.
    assert bound.arguments["include_timeseries"] is True
    assert bound.arguments["include_summary"] is True


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


def test_unknown_command_is_rejected():
    """Cyclopts surfaces unknown commands as a CycloptsError."""
    import cyclopts

    with pytest.raises(cyclopts.CycloptsError):
        app.parse_args(["definitely-not-a-command"], exit_on_error=False, print_error=False)


def test_to_zarr_rejects_missing_input(tmp_path: pathlib.Path):
    """``ExistingDirectory`` validator rejects nonexistent paths."""
    import cyclopts

    missing = tmp_path / "does-not-exist"
    with pytest.raises(cyclopts.CycloptsError):
        app.parse_args(
            ["extract", "to-zarr", "--input-path", str(missing)],
            exit_on_error=False,
            print_error=False,
        )
