"""Render SCHISM zarr output to PNGs and MP4 videos."""
from __future__ import annotations

import functools
import os
import pathlib
import shlex
import subprocess

import colorcet
import datashader
import datashader.transfer_functions as tf
import multifutures as mf
import numpy as np
import pandas as pd
import PIL.ImageDraw
import PIL.ImageFont
import shapely
import tqdm.auto
import xarray as xr

from spost._utils import build_edge_df
from spost._utils import build_simplices
from spost._utils import is_overlapping
from spost._utils import open_zarr_store


def _load_font(size: int = 20) -> PIL.ImageFont.FreeTypeFont | PIL.ImageFont.ImageFont:
    for name in ("Ubuntu-M.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return PIL.ImageFont.truetype(name, size)
        except OSError:
            continue
    return PIL.ImageFont.load_default()


def render_pngs(
    ds: xr.Dataset,
    variable: str,
    output_path: pathlib.Path,
    *,
    width: int = 1920,
    height: int = 1080,
    cmap: list | None = None,
    show_mesh: bool = False,
    overwrite: bool = True,
    font_size: int = 20,
    workers: int = 12,
) -> list[pathlib.Path]:
    """
    Render each timestep of a variable to a PNG using datashader trimesh.

    Parameters
    ----------
    ds
        xarray Dataset with mesh coordinates (SCHISM_hgrid_node_x/y,
        SCHISM_hgrid_face_nodes) and the target variable with a time dimension.
    variable
        Name of the variable to render.
    output_path
        Directory to write PNGs into.
    width, height
        Image dimensions in pixels.
    cmap
        Colorcet colormap list. Defaults per variable.
    overwrite
        If True, overwrites PNGs that already exist.
    show_mesh
        If True, overlays the mesh edges on top of the variable rendering.
    font_size
        Font size for the timestamp overlay.
    workers
        Parallel worker count. ``workers <= 1`` runs sequentially in-process
        (no subprocesses), which is easier to debug on resource-constrained
        machines such as HPC login nodes where loky workers can be killed by
        cgroup memory limits and produce a silent failure.

    Returns
    -------
    List of paths to the generated PNG files.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    font = _load_font(font_size)

    coords_df = ds[["SCHISM_hgrid_node_x", "SCHISM_hgrid_node_y"]].to_dataframe().reset_index(drop=True)
    simplices_df = build_simplices(ds)
    if show_mesh:
        edge_df = build_edge_df(coords_df,simplices_df)
    else:
        edge_df = None

    canvas = datashader.Canvas(plot_width=width, plot_height=height)
    da = ds[variable]
    has_time = "time" in da.dims
    timesteps = da.time.values if has_time else [None]

    def render_timestep(
        i: int,
        ts: np.datetime64 | None,
        *,
        canvas: datashader.Canvas,
        coords_df: pd.DataFrame,
        simplices_df: pd.DataFrame,
        edge_df: pd.DataFrame | None,
        da: xr.DataArray,
        variable: str,
        output_path: pathlib.Path,
        overwrite: bool,
        cmap: list,
        font: PIL.ImageFont.FreeTypeFont,
    ) -> None:
        png_file = output_path / f"{i:06d}.png"
        if not overwrite and png_file.exists():
            return

        step_values = (da.sel(time=ts) if ts is not None else da).values
        vertices_df = coords_df.assign(**{variable: step_values})
        agg = canvas.trimesh(vertices=vertices_df, simplices=simplices_df, agg=datashader.reductions.mean(variable))
        img = tf.shade(agg, cmap=cmap, how="eq_hist")

        if edge_df is not None:
            wireframe = tf.shade(canvas.line(edge_df, x="x", y="y", agg=datashader.reductions.any()), cmap=["black"])
            img = tf.stack(img, wireframe)

        pil_img = img.to_pil()
        draw = PIL.ImageDraw.Draw(pil_img)
        if ts is not None:
            draw.text((20, 20), pd.to_datetime(ts).isoformat(), fill="white", font=font)
        pil_img.save(png_file)

    _render = functools.partial(
        render_timestep,
        canvas=canvas,
        coords_df=coords_df,
        simplices_df=simplices_df,
        edge_df=edge_df,
        da=da,
        variable=variable,
        output_path=output_path,
        overwrite=overwrite,
        cmap=cmap,
        font=font,
    )

    jobs = [
        {"i": i, "ts": ts}
        for i, ts in enumerate(timesteps)
        if overwrite or not (output_path / f"{i:06d}.png").exists()
    ]

    if workers <= 1:
        for kw in tqdm.auto.tqdm(jobs, desc="rendering"):
            _render(**kw)
        return

    mf.multiprocess(
        func=_render,
        func_kwargs=jobs,
        max_workers=workers,
        check=True,
    )

def pngs_to_mp4(
    png_dir: pathlib.Path,
    output_file: pathlib.Path,
    *,
    framerate: int = 48,
    glob_pattern: str = "%06d.png",
) -> pathlib.Path:
    """
    Stitch PNGs into an MP4 using ffmpeg.

    Parameters
    ----------
    png_dir
        Directory containing sequentially numbered PNGs.
    output_file
        Path for the output MP4 file.
    framerate
        Frames per second.
    glob_pattern
        ffmpeg input pattern (must match the PNG naming scheme).

    Returns
    -------
    Path to the generated MP4.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"ffmpeg -y -framerate {framerate} "
        f"-i {glob_pattern} "
        f"-c:v libx264 -pix_fmt yuv420p "
        f"{output_file}"
    )
    subprocess.run(shlex.split(cmd), check=True, cwd=png_dir)
    return output_file


def _to_pngs(
    input_path: pathlib.Path,
    variable: str,
    output_path: pathlib.Path,
    *,
    width: int = 1920,
    height: int = 1080,
    cmap: str | None = None,
    overwrite: bool = False,
    show_mesh: bool = False,
    region=None,
    workers: int = 4,
) -> list[pathlib.Path]:
    ds = open_zarr_store(input_path)
    if region is not None:
        from spost._clip import clip_ds

        ds = clip_ds(ds, region)
    cmap_list = None
    if cmap is not None:
        name = cmap.removesuffix("_r")
        reverse = name != cmap
        cmap_list = getattr(colorcet, name, None)
        if cmap_list is None:
            raise ValueError(f"Unknown colorcet colormap: {cmap}")
        if reverse:
            cmap_list = cmap_list[::-1]
    return render_pngs(
        ds,
        variable,
        output_path,
        width=width,
        height=height,
        cmap=cmap_list,
        overwrite=overwrite,
        show_mesh=show_mesh,
        workers=workers,
    )


def _to_mp4(
    input_path: pathlib.Path,
    variable: str,
    output_path: pathlib.Path,
    *,
    width: int = 1920,
    height: int = 1080,
    framerate: int = 48,
    cmap: str | None = None,
    overwrite: bool = False,
    png_dir: pathlib.Path | None = None,
    region=None,
    workers: int = 4,
) -> pathlib.Path:
    if png_dir is None:
        png_dir = output_path.parent / f".pngs_{variable}"
    _to_pngs(
        input_path,
        variable,
        png_dir,
        width=width,
        height=height,
        cmap=cmap,
        overwrite=overwrite,
        region=region,
        workers=workers,
    )
    return pngs_to_mp4(png_dir, output_path, framerate=framerate)
