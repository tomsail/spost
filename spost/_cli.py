import pathlib
from typing import Annotated

import cyclopts
from cyclopts.types import ExistingDirectory
from cyclopts.types import ExistingPath

from spost._constants import parse_bbox
from spost._literals import RegionName
from spost.validate._cli import (
    compare_app,
    fetch_obs_app,
    report_app,
    tidal_app,
    validate_app,
)

app = cyclopts.App(name="spost", help="Post processing tools for SCHISM output")
extract_app = cyclopts.App(name="extract", help="Convert/Extract SCHISM files to readable outputs")
plot_app = cyclopts.App(name="plot", help="Produce graphs from SCHISM outputs")
skill_app = cyclopts.App(
    name="skill",
    help="Validation pipeline: fetch obs, compare, tidal harmonics, report.",
)
app.command(extract_app)
app.command(plot_app)
app.command(skill_app)
skill_app.command(validate_app)
skill_app.command(fetch_obs_app)
skill_app.command(compare_app)
skill_app.command(report_app)
skill_app.command(tidal_app)

_CONFIG_AWARE_APPS = (app, skill_app, validate_app, fetch_obs_app, compare_app, report_app, tidal_app)


@extract_app.command
def to_zarr(
    *,
    input_path: ExistingDirectory | None = None,
    output: pathlib.Path | None = None,
    variables: Annotated[list[str], cyclopts.Parameter(consume_multiple=True)] = ["all"],
    workers: int = 12,
    clevel: int = 3,
    overwrite: bool = False,
    exclude_last: int = 0,
):
    """Convert SCHISM output to Zarr format.

    Supported variables:
     * elevation
     * depth_average_velocity_x
     * depth_average_velocity_y
     * salinity
     * temperature.

    Parameters
    ----------
    input_path
        Path to SCHISM run directory.
    output
        Output zarr path. Defaults to {input_path}/{input_path.stem}.zarr.
    variables
        Which variables to include.
    workers
        Number of parallel workers.
    clevel
        Compression level (1-9).
    overwrite
        Overwrite existing store.
    exclude_last
        Drop the last N files of each pattern (out2d_*.nc, salinity_*.nc, ...)
        from the natsorted glob. Useful when the most recent SCHISM segment is
        still being written and the trailing file is truncated/corrupted.
    """
    if input_path is None:
        extract_app["to-zarr"].help_print()
        raise SystemExit(0)

    from spost._to_zarr import to_zarr as _to_zarr
    from spost._to_zarr import VARIABLE_SPECS

    if variables != ["all"]:
        invalid = set(variables) - VARIABLE_SPECS.keys()
        if invalid:
            raise ValueError(f"Invalid variable(s): {invalid}. Valid options: {list(VARIABLE_SPECS)}")
    if output is None:
        output = input_path / f"{input_path.stem}.zarr"

    _to_zarr(
        base_path=input_path,
        store_path=output,
        variables=variables,
        workers=workers,
        clevel=clevel,
        overwrite=overwrite,
        exclude_last=exclude_last,
    )


@extract_app.command
def clip(
    *,
    input_path: pathlib.Path | None = None,
    output_path: pathlib.Path | None = None,
    overwrite: bool = False,
    bbox: Annotated[RegionName, cyclopts.Parameter(converter=parse_bbox)] = "world",
) -> None:
    """
    Clip a SCHISM zarr store to a bounding box and write a new store.

    Parameters
    ----------
    input_path
        Path to the source zarr store.
    output_path
        Path for the clipped zarr store.
    overwrite
        Overwrite existing output store.
    bbox
        Bounding box as (`lon_min`, `lat_min`, `lon_max`, `lat_max`) or a region string.
        Attention! If `lon_min` starts negative you need to use a = sign, eg: --bbox`=`"-4 0 10 30"
    """
    if input_path is None:
        extract_app["clip"].help_print()
        raise SystemExit(0)

    from spost._clip import clip_zarr

    clip_zarr(
        input_path=input_path,
        output_path=output_path or (input_path.parent / f"{input_path.stem}_clipped.zarr"),
        bbox=bbox,
        overwrite=overwrite,
    )


@plot_app.command
def to_pngs(
    *,
    input_path: ExistingPath | None = None,
    variable: str | None = None,
    output_path: pathlib.Path | None = None,
    width: int = 1920,
    height: int = 1080,
    cmap: str = "coolwarm",
    overwrite: bool = True,
    clip: Annotated[RegionName, cyclopts.Parameter(converter=parse_bbox)] = None,
):
    """Render a variable from a zarr store to PNG frames.

    Parameters
    ----------
    input_path
        Path to the zarr store.
    variable
        Variable name to render (as stored in zarr).
    output_path
        Directory for output PNGs. Defaults to ./{variable}_pngs/.
    variable
        Variable name to render.
    width
        Image width in pixels.
    height
        Image height in pixels.
    cmap
        Colorcet colormap name (e.g. 'coolwarm', 'fire', 'rainbow')
    overwrite
        Re-render PNGs even if they already exist.
    clip
        Clip the rendering to bbox as (`lon_min`, `lat_min`, `lon_max`, `lat_max`) or a region string.
        Attention! If `lon_min` starts negative you need to use a = sign, eg: --clip`=`"-4 0 10 30"
    """
    if input_path is None or variable is None:
        plot_app["to-pngs"].help_print()
        raise SystemExit(0)
    if output_path is None:
        output_path = pathlib.Path(f"./{variable}_pngs")

    from spost._render import _to_pngs

    _to_pngs(
        input_path=input_path,
        variable=variable,
        output_path=output_path,
        width=width,
        height=height,
        cmap=cmap,
        overwrite=overwrite,
        clip=clip,
    )


@plot_app.command
def to_mp4(
    *,
    input_path: ExistingPath | None = None,
    variable: str | None = None,
    output_path: pathlib.Path | None = None,
    width: int = 1920,
    height: int = 1080,
    framerate: int = 48,
    cmap: str = "coolwarm",
    png_dir: pathlib.Path | None = None,
    overwrite: bool = False,
    clip: Annotated[RegionName, cyclopts.Parameter(converter=parse_bbox)] = None,
):
    """Render a variable from a zarr store to an MP4 video.

    Generates PNGs first, then stitches them with ffmpeg.

    Parameters
    ----------
    input_path
        Path to the zarr store.
    variable
        Variable name to render.
    output_path
        Output MP4 file path. Defaults to ./{variable}.mp4.
    width
        Image width in pixels.
    height
        Image height in pixels.
    framerate
        Video framerate (fps).
    cmap
        Colorcet colormap name.
    png_dir
        Directory for intermediate PNGs. Defaults to a hidden dir next to output.
    overwrite
        Re-render PNGs and MP4 even if they already exist.
    clip
        Clip the rendering to bbox as (`lon_min`, `lat_min`, `lon_max`, `lat_max`) or a region string.
        Attention! If `lon_min` starts negative you need to use a = sign, eg: --clip`=`"-4 0 10 30"
    """
    if input_path is None or variable is None:
        plot_app["to-mp4"].help_print()
        raise SystemExit(0)
    if output_path is None:
        output_path = pathlib.Path(f"./{variable}.mp4")

    from spost._render import _to_mp4

    _to_mp4(
        input_path=input_path,
        variable=variable,
        output_path=output_path,
        width=width,
        height=height,
        framerate=framerate,
        cmap=cmap,
        overwrite=overwrite,
        png_dir=png_dir,
        clip=clip,
    )


@extract_app.command
def stations(
    *outputs_dirs: pathlib.Path,
    output_path: pathlib.Path = pathlib.Path("./stations"),
    staout_indices: Annotated[list[int], cyclopts.Parameter(consume_multiple=True)] | None = None,
):
    """Extract station time series to parquet files.

    Reads station.in and staout_* files from one or more SCHISM outputs/
    directories (for hotstart segments) and writes per-station parquet files,
    organized by variable.

    Parameters
    ----------
    output_dir
        One or more SCHISM outputs/ directories.
    output_path
        Base directory for parquet output
    staout_indices
        Which staout indices to process (1-9). Defaults to auto-detect.
    """
    if len(outputs_dirs) == 0:
        extract_app["stations"].help_print()
        raise SystemExit(0)
    for d in outputs_dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"Directory not found: {d}")
    if output_path is None:
        output_path = pathlib.Path("./stations")

    from spost._stations import _extract_stations

    _extract_stations(
        outputs_dirs=outputs_dirs,
        output_path=output_path,
        staout_indices=staout_indices,
    )


@app.meta.default
def _meta(
    *tokens: Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    config: pathlib.Path | None = None,
    verbose: bool = False,
):
    """Top-level entry point.

    ``--config PATH`` attaches a TOML defaults file (see hydrogen-style
    ``cyclopts.config.Toml``); when omitted, parents of the cwd are searched
    for a ``pyproject.toml`` with a ``[spost.<command>]`` section.
    """
    import logging

    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")

    config_path = config
    if config_path is None:
        # Auto-discover pyproject.toml in cwd ancestors.
        cwd = pathlib.Path.cwd()
        for parent in [cwd, *cwd.parents]:
            candidate = parent / "pyproject.toml"
            if candidate.exists():
                config_path = candidate
                break

    if config_path is not None and config_path.exists():
        try:
            config_handler = cyclopts.config.Toml(
                config_path,
                root_keys=["spost"],
                use_commands_as_keys=True,
                allow_unknown=True,
                search_parents=False,
            )
            for sub in _CONFIG_AWARE_APPS:
                sub.config = config_handler
        except Exception:
            # A malformed or partial config should never break the CLI.
            pass

    app(tokens)


def main():
    app.meta()
