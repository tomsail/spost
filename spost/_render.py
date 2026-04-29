"""Render SCHISM zarr output to PNGs and MP4 videos."""

from __future__ import annotations

import os
import pathlib
import shlex
import subprocess

import functools
import colorcet
import datashader
import datashader.transfer_functions as tf
import multifutures as mf
import numpy as np
import pandas as pd
import shapely
import PIL.ImageDraw
import PIL.ImageFont
import tqdm.auto
import xarray as xr

from spost._utils import build_simplices
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
    canvas = datashader.Canvas(plot_width=width, plot_height=height)
    da = ds[variable]

    dict_list = []
    for i, ts in enumerate(da.time.values):
        png_file = output_path / f"{i:06d}.png"
        if overwrite or not png_file.exists():
            dict_list.append(
                {
                    "simplices_df": simplices_df,
                    "variable": variable,
                    "ts": ts,
                }
            )

    def render_timestep(
        canvas: datashader.Canvas,
        simplices_df: pd.DataFrame,
        da: xr.Dataset,
        variable: str,
        ts: np.datetime64,
        output_path: pathlib.Path,
        overwrite: bool,
        cmap: list,
    ) -> None:
        i = np.where(da.time.values == ts)[0][0]
        png_file = output_path / f"{i:06d}.png"

        if overwrite or not png_file.exists():
            step_values = da.sel(time=ts).values
            vertices_df = coords_df.assign(**{variable: step_values})
            agg = canvas.trimesh(
                vertices=vertices_df,
                simplices=simplices_df,
                agg=datashader.reductions.mean(variable),
            )
            img = tf.shade(agg, cmap=cmap, how="eq_hist")
            pil_img = img.to_pil()
            # Timestamp overlay
            draw = PIL.ImageDraw.Draw(pil_img)
            timestamp_str = pd.to_datetime(ts).isoformat()
            draw.text((20, 20), timestamp_str, fill="white", font=font)
            # save
            pil_img.save(png_file)

    if not dict_list:
        return

    if workers <= 1:
        for kw in tqdm.auto.tqdm(dict_list, desc="rendering"):
            render_timestep(**kw)
        return

    mf.multiprocess(
        func=functools.partial(
            render_timestep,
            canvas=canvas,
            output_path=output_path,
            da=da,
            overwrite=overwrite,
            cmap=cmap
        ),
        func_kwargs=dict_list,
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
    clip: tuple = None,
    workers: int = 4,
) -> list[pathlib.Path]:
    ds = open_zarr_store(input_path)
    if clip is not None:
        from spost._clip import clip_ds

        ds = clip_ds(ds, clip)
    cmap_list = None
    if cmap is not None:
        cmap_list = getattr(colorcet, cmap, None)
        if cmap_list is None:
            raise ValueError(f"Unknown colorcet colormap: {cmap}")
    return render_pngs(
        ds,
        variable,
        output_path,
        width=width,
        height=height,
        cmap=cmap_list,
        overwrite=overwrite,
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
    clip: tuple = None,
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
        clip=clip,
        workers=workers,
    )
    return pngs_to_mp4(png_dir, output_path, framerate=framerate)
