#!/usr/bin/env python3
import os
import pathlib

def compute_tidemap(
    *,
    input_path: pathlib.Path,
    output: pathlib.Path,
    chunk_size: int = 100,
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
    """

    # CRITICAL: Set these BEFORE any numpy/scipy imports
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    import xarray as xr
    from ._utils import detide, FULL, METRICS

    data = xr.open_zarr(input_path)
    S = data['elevation']

    result = detide(S, chunk_size=chunk_size)

    print(f"Results shape: {result.shape}")
    coef_da = xr.DataArray(
        result,
        dims=("nSCHISM_hgrid_node", "constituent", "metric"),
        coords={
            "nSCHISM_hgrid_node": data.nSCHISM_hgrid_node,
            "constituent": FULL,
            "metric": METRICS,
            "lon": data.SCHISM_hgrid_node_x,
            "lat": data.SCHISM_hgrid_node_y,
        },
        name="tidal_coefs",
    )
    coef_ds = coef_da.to_dataset(dim="metric")
    coef_ds.to_netcdf(output)
