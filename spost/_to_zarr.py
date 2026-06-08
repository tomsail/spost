import functools
import pathlib
import typing as ty
from collections.abc import Sequence

import multifutures as mf
import natsort
import numpy as np
import xarray as xr
import zarr.codecs

from ._utils import open_schism_output
from ._utils import sanitize_attrs


STATIC_VARIABLES = [
    "SCHISM_hgrid_edge_nodes",
    "SCHISM_hgrid_edge_x",
    "SCHISM_hgrid_edge_y",
    "SCHISM_hgrid_face_nodes",
    "SCHISM_hgrid_face_x",
    "SCHISM_hgrid_face_y",
    "SCHISM_hgrid_node_x",
    "SCHISM_hgrid_node_y",
    "crs",
    "depth",
    "time",
]


class VariableSpec(ty.TypedDict):
    nc_variable: str
    zarr_variable: str
    pattern: str


VARIABLE_SPECS: dict[str, VariableSpec] = {
    "elevation": {
        "nc_variable": "elevation",
        "zarr_variable": "elevation",
        "pattern": "out2d_*.nc",
    },
    "depth_average_velocity_x": {
        "nc_variable": "depthAverageVelX",
        "zarr_variable": "depth_average_velocity_x",
        "pattern": "out2d_*.nc",
    },
    "depth_average_velocity_y": {
        "nc_variable": "depthAverageVelY",
        "zarr_variable": "depth_average_velocity_y",
        "pattern": "out2d_*.nc",
    },
    "salinity": {
        "nc_variable": "salinity",
        "zarr_variable": "sss",
        "pattern": "salinity_*.nc",
    },
    "temperature": {
        "nc_variable": "temperature",
        "zarr_variable": "sst",
        "pattern": "temperature_*.nc",
    },
}


def get_compressor(clevel: int = 3) -> zarr.codecs.BloscCodec:
    return zarr.codecs.BloscCodec(cname="zstd", clevel=clevel, shuffle="bitshuffle", blocksize=0)


def initialize_store(
    base_path: pathlib.Path,
    store_path: pathlib.Path,
    overwrite: bool = False,
    exclude_last: int = 0,
):
    group = zarr.create_group(store=store_path, overwrite=overwrite, zarr_format=3)
    ds = open_schism_output(base_path, "out2d_*.nc", exclude_last=exclude_last)
    for var in STATIC_VARIABLES:
        da = ds[var]
        group.create_array(
            name=var,
            data=da.values,
            dimension_names=da.dims,
            attributes=sanitize_attrs(da.attrs),
            chunks=da.shape,
            overwrite=True,
            fill_value=None,
        )
    # SCHISM_hgrid datatype is bytes which is not supported by zarr, so we need to change it via `.astype()`
    var = "SCHISM_hgrid"
    da = ds[var]
    group.create_array(
        name=var,
        data=da.values.astype(np.bool_),
        dimension_names=da.dims,
        attributes=sanitize_attrs(da.attrs),
        chunks=da.shape,
        overwrite=True,
        fill_value=None,
    )


def create_2D_array(
    base_path: pathlib.Path,
    store_path: pathlib.Path,
    nc_variable: str,
    pattern: str,
    zarr_variable: str,
    clevel: int = 3,
    node_chunk: int = 500,
    node_shard: int = 50000,
    exclude_last: int = 0,
):
    group = zarr.open_group(store=store_path)
    ds = open_schism_output(base_path, pattern, exclude_last=exclude_last)
    da = ds[nc_variable]
    # If 3D variable, convert to 2D by selecting the top layer
    if "nSCHISM_vgrid_layers" in da.dims:
        da = da.isel(nSCHISM_vgrid_layers=-1)
    group.create_array(
        name=zarr_variable,
        shape=da.shape,
        dtype=da.dtype,
        dimension_names=da.dims,
        attributes=sanitize_attrs(da.attrs),
        chunks=(len(da.time), node_chunk),
        shards=(len(da.time), node_shard),
        overwrite=True,
        fill_value=None,
        compressors=(get_compressor(clevel),),
    )


def process_spatial_chunk(
    store_path: pathlib.Path,
    zarr_variable: str,
    da: xr.DataArray,
    node_start: int,
    node_end: int,
):
    group = zarr.open_group(store_path)
    array = group[zarr_variable]
    array[:, node_start:node_end] = da.isel(nSCHISM_hgrid_node=slice(node_start, node_end)).values


def populate_array(
    base_path: pathlib.Path,
    store_path: pathlib.Path,
    nc_variable: str,
    zarr_variable: str,
    pattern: str,
    node_chunk: int = 500,
    workers: int = 12,
    exclude_last: int = 0,
):
    ds = open_schism_output(base_path, pattern, exclude_last=exclude_last)
    da = ds[nc_variable]
    # If 3D variable, select top layer
    if "nSCHISM_vgrid_layers" in da.dims:
        da = da.isel(nSCHISM_vgrid_layers=-1)
    n_nodes = len(da.nSCHISM_hgrid_node)
    chunk_ranges = [(i, min(i + node_chunk, n_nodes)) for i in range(0, n_nodes, node_chunk)]
    _ = mf.multiprocess(
        func=functools.partial(process_spatial_chunk, store_path=store_path, zarr_variable=zarr_variable, da=da),
        func_kwargs=[dict(node_start=start, node_end=end) for start, end in chunk_ranges],
        max_workers=workers,
        include_kwargs=False,
        check=True,
    )


def to_zarr(
    base_path: pathlib.Path,
    store_path: pathlib.Path,
    variables: Sequence[str],
    workers: int = 12,
    clevel: int = 3,
    overwrite: bool = False,
    exclude_last: int = 0,
):
    initialize_store(base_path, store_path, overwrite=overwrite, exclude_last=exclude_last)
    if variables == ["all"]:
        variables = list(VARIABLE_SPECS.keys())
    for var in variables:
        spec = VARIABLE_SPECS[var]
        create_2D_array(base_path, store_path, clevel=clevel, exclude_last=exclude_last, **spec)
    zarr.consolidate_metadata(store_path)
    for var in variables:
        spec = VARIABLE_SPECS[var]
        populate_array(base_path, store_path, workers=workers, exclude_last=exclude_last, **spec)
