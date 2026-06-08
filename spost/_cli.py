import pathlib
from typing import Annotated

import cyclopts
from cyclopts.types import ExistingDirectory
from cyclopts.types import ExistingPath

from spost._region import BboxArg
from spost._region import resolve_region
from spost._region import WktArg


def to_zarr(
    *,
    input_path: ExistingDirectory | None = None,
    output: pathlib.Path | None = None,
    variables: Annotated[list[str], cyclopts.Parameter(consume_multiple=True, negative=())] = ["all"],
    workers: int = 12,
    clevel: int = 3,
    overwrite: Annotated[bool, cyclopts.Parameter(negative=())] = False,
    exclude_last: int = 0,
):
    """
    Convert SCHISM output to Zarr format.

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


def clip(
    *,
    input_path: pathlib.Path | None = None,
    output_path: pathlib.Path | None = None,
    overwrite: Annotated[bool, cyclopts.Parameter(negative=())] = False,
    bbox: BboxArg = None,
    wkt: WktArg = None,
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
        Explicit bounding box, four floats: lon_min lat_min lon_max lat_max.
        Mutually exclusive with --wkt.
    wkt
        WKT polygon file, or the name of a bundled region under
        spost/regions/ (e.g. ``--wkt med``). Mutually exclusive with --bbox.
    """
    from spost._clip import clip_zarr

    region = resolve_region(bbox, wkt)

    clip_zarr(
        input_path=input_path,
        output_path=output_path or (input_path.parent / f"{input_path.stem}_clipped.zarr"),
        region=region,
        overwrite=overwrite,
    )

PLOT_GROUP = cyclopts.Group("Plotting Options")

def to_pngs(
    *,
    input_path: ExistingPath,
    variable: str,
    output_path: pathlib.Path | None = None,
    overwrite: Annotated[bool, cyclopts.Parameter(negative=())] = False,
    width: Annotated[int, cyclopts.Parameter(group=PLOT_GROUP)] = 1920,
    height: Annotated[int, cyclopts.Parameter(group=PLOT_GROUP)] = 1080,
    cmap: Annotated[str, cyclopts.Parameter(group=PLOT_GROUP)] = "coolwarm",
    show_mesh: Annotated[bool, cyclopts.Parameter(group=PLOT_GROUP, negative=())] = False,
    depth_shading: Annotated[bool, cyclopts.Parameter(group=PLOT_GROUP, negative=())] = False,
    bbox: Annotated[BboxArg, cyclopts.Parameter(negative=())] = None,
    wkt: WktArg = None,
    workers: int = 4,
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
    width
        Image width in pixels.
    height
        Image height in pixels.
    cmap
        Colorcet colormap name (e.g. 'coolwarm', 'fire', 'rainbow').
        Append '_r' to reverse the colormap (e.g. 'fire_r', 'coolwarm_r').
    overwrite
        Re-render PNGs even if they already exist.
    bbox
        Explicit bounding box, four floats: lon_min lat_min lon_max lat_max.
        Mutually exclusive with --wkt.
    wkt
        WKT polygon file, or the name of a bundled region (e.g. ``--wkt med``).
        Mutually exclusive with --bbox.
    show_mesh
        If True, overlays the mesh edges on top of the variable rendering.
    depth_shading
        If True, overlays a hillshade shading based on the depth variable
    workers
        Parallel worker count. Use ``--workers 1`` to run sequentially in
        the parent process - recommended on HPC login nodes where loky
        workers may be killed by cgroup memory limits and the failure
        otherwise surfaces only as a UserWarning.
    """
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
        region=resolve_region(bbox, wkt),
        show_mesh=show_mesh,
        depth_shading=depth_shading,
        workers=workers,
    )


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
    overwrite: Annotated[bool, cyclopts.Parameter(negative=())] = False,
    bbox: Annotated[BboxArg, cyclopts.Parameter(negative=())] = None,
    wkt: WktArg = None,
    workers: int = 4,
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
    bbox
        Explicit bounding box: lon_min lat_min lon_max lat_max.
        Mutually exclusive with --wkt.
    wkt
        WKT polygon file, or the name of a bundled region (e.g. ``--wkt med``).
        Mutually exclusive with --bbox.
    """
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
        region=resolve_region(bbox, wkt),
        workers=workers,
    )


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
