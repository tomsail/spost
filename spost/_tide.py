#!/usr/bin/env python3
import os
import pathlib

def compute_tidemap(
    *,
    input_path: pathlib.Path,
    output: pathlib.Path,
    chunk_size: int = 100,
    overwrite: bool = False,
    fes: pathlib.Path | None = None,
    tpxo: pathlib.Path | None = None,
):
    """
    Compute tidal constituent maps from SCHISM elevation output.

    Parameters
    ----------
    rundir : Path
        Directory containing SCHISM run output (searches for date subdirs).
    output : Path
        Output NetCDF file path for tidal coefficients.
    chunk_size : int
        Number of nodes per joblib chunk.
    overwrite : bool
        Overwrite existing output file if it exists.
    fes : Path, optional
        Root directory of a pyTMD-compatible FES model store. When given,
        the newest auto-detected FES release under `rundir` is interpolated
        onto the mesh nodes
    tpxo : Path, optional
        Root directory of a pyTMD-compatible TPXO model store. When given,
        the newest auto-detected TPXO release under `rundir` is interpolated
        onto the mesh nodes
    """

    # CRITICAL: Set these BEFORE any numpy/scipy imports
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    import zarr

    import xarray as xr

    from ._utils import detide
    from ._utils import FULL
    from ._utils import interpolate_tide_model

    from ._to_zarr import get_compressor
    from ._to_zarr import sanitize_attrs

    # newest-to-oldest
    FES_MODEL_CANDIDATES = [
        "FES2022_extrapolated", "FES2022",
        "FES2014_extrapolated", "FES2014",
        "FES2012",
    ]
    TPXO_MODEL_CANDIDATES = [
        "TPXO10-atlas-v2-nc", "TPXO10-atlas-v2",
        "TPXO9-atlas-v5-nc", "TPXO9-atlas-v5",
        "TPXO9-atlas-v4-nc", "TPXO9-atlas-v4",
        "TPXO9-atlas-v3-nc", "TPXO9-atlas-v3",
        "TPXO9-atlas-v2-nc", "TPXO9-atlas-v2",
        "TPXO9-atlas-nc", "TPXO9-atlas",
        "TPXO8-atlas-nc", "TPXO8-atlas",
        "TPXO7.2",
    ]
    NODE_CHUNK = 1000
    NODE_SHARD = 1000000
    CLEVEL = 3

    if os.path.exists(output) and not overwrite:
        raise ValueError(f"Output file {output} already exists. Add `--overwrite` flag")

    data = xr.open_zarr(input_path)
    S = data['elevation']

    # Multithreaded part
    result = detide(S, chunk_size=chunk_size)

    coords = {
        "nSCHISM_hgrid_node": data.nSCHISM_hgrid_node,
        "constituent": FULL,
        "lon": data.SCHISM_hgrid_node_x,
        "lat": data.SCHISM_hgrid_node_y,
    }
    coef_da = xr.DataArray(
        result,
        dims=("nSCHISM_hgrid_node", "constituent"),
        coords=coords,
        name="model",
    )
    coef_ds = coef_da.to_dataset()

    # Add FES or TPXO
    lons = data.SCHISM_hgrid_node_x.values
    lats = data.SCHISM_hgrid_node_y.values
    for directory, candidates, var_name in (
        (fes, FES_MODEL_CANDIDATES, "fes"),
        (tpxo, TPXO_MODEL_CANDIDATES, "tpxo"),
    ):
        if directory is None:
            continue
        print(f"Detecting tide model under {directory} ...")
        model_name, model_result = interpolate_tide_model(directory, lons, lats, candidates, constituents=FULL)
        print(f"Tide interpolated onto mesh nodes.")
        coef_ds[var_name] = xr.DataArray(
            model_result,
            dims=("nSCHISM_hgrid_node", "constituent"),
            coords=coords,
            attrs={f"{var_name} model": model_name},
        )

    # export
    group = zarr.create_group(store=output, overwrite=overwrite, zarr_format=3)
    for var in ["model", "fes", "tpxo"]:
        if var not in coef_ds:
            continue
        da = coef_ds[var]
        group.create_array(
            name=var,
            shape=da.shape,
            dtype=da.dtype,
            dimension_names=da.dims,
            attributes=sanitize_attrs(da.attrs),
            chunks=(NODE_CHUNK, len(da.constituent)),
            shards=(NODE_SHARD, len(da.constituent)),
            overwrite=True,
            fill_value=None,
            compressors=(get_compressor(CLEVEL),),
        )
