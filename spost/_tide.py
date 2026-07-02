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

    import numpy as np
    import xarray as xr

    from ._utils import detide
    from ._utils import FULL
    from ._utils import interpolate_tide_model

    from ._to_zarr import get_compressor

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
    NODE_CHUNK = 1_000
    NODE_SHARD = 10_000_000
    CLEVEL = 3

    if os.path.exists(output) and not overwrite:
        raise ValueError(f"Output file {output} already exists. Add `--overwrite` flag")

    data = xr.open_zarr(input_path)
    S = data['elevation']

    # Multithreaded part
    result = detide(S, chunk_size=chunk_size)

    coords = {
        "nSCHISM_hgrid_node": data.nSCHISM_hgrid_node,
        "constituent": np.array(FULL, dtype=object),
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

    tidal_models = []
    for directory, candidates in (
        (fes, FES_MODEL_CANDIDATES),
        (tpxo, TPXO_MODEL_CANDIDATES),
    ):
        if directory is None:
            continue
        print(f"Detecting tide model under {directory} ...")
        model_name, model_result = interpolate_tide_model(directory, lons, lats, candidates, constituents=FULL)
        print(f"Tide interpolated onto mesh nodes.")
        coef_ds[model_name] = xr.DataArray(
            model_result,
            dims=("nSCHISM_hgrid_node", "constituent"),
            coords=coords,
        )
        tidal_models.append(model_name)

    coef_vars = ["model", *tidal_models]
    encoding = {
        var: {
            "compressors": (get_compressor(CLEVEL),),
            "chunks": (NODE_CHUNK, coef_ds.sizes["constituent"]),
            "shards": (NODE_SHARD, coef_ds.sizes["constituent"]),
        }
        for var in coef_vars
    }

    if "SCHISM_hgrid_face_nodes" in data:
        faces = data["SCHISM_hgrid_face_nodes"]
        coef_ds["SCHISM_hgrid_face_nodes"] = faces
        coef_ds["SCHISM_hgrid_face_nodes"].encoding.clear()
        n_faces = coef_ds.sizes[faces.dims[0]]
        n_face_nodes = coef_ds.sizes[faces.dims[1]]
        encoding["SCHISM_hgrid_face_nodes"] = {
            "compressors": (get_compressor(CLEVEL),),
            "chunks": (min(NODE_CHUNK, n_faces) or 1, n_face_nodes),
        }

    coef_ds.to_zarr(
        output,
        mode="w",
        zarr_format=3,
        encoding=encoding,
        consolidated=True,
    )
