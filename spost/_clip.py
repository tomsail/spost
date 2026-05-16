import logging
import pathlib

import numpy as np
import shapely
import xarray as xr

logger = logging.getLogger(__name__)


def resolve_region(
    region: shapely.Geometry | tuple[float, float, float, float] | None = None,
) -> shapely.Polygon:
    """Normalize the region argument to a shapely polygon.

    Accepts an already-resolved shapely geometry, a 4-tuple bbox, or ``None``
    (which yields the full world).
    """
    if region is None:
        return shapely.box(-180, -90, 180, 90)
    if isinstance(region, shapely.Geometry):
        return region
    return shapely.box(*region)


def crop(x: np.ndarray, y: np.ndarray, tri: np.ndarray, bbox: tuple):
    import numpy_indexed as npi

    nodes_mask = np.where(
        True & (y >= bbox.bounds[1]) & (y <= bbox.bounds[3]) & (x >= bbox.bounds[0]) & (x <= bbox.bounds[2]),
    )[0]
    tri_mask = np.where(np.isin(tri, nodes_mask).all(axis=1))[0]
    remapped_nodes = np.arange(len(nodes_mask))
    remapped_triface_nodes = np.c_[
        npi.remap(tri[tri_mask][:, 0], nodes_mask, remapped_nodes),
        npi.remap(tri[tri_mask][:, 1], nodes_mask, remapped_nodes),
        npi.remap(tri[tri_mask][:, 2], nodes_mask, remapped_nodes),
    ]
    return nodes_mask, tri_mask, remapped_triface_nodes


def clip_ds(ds: xr.Dataset, region: shapely.Polygon | tuple | None = None) -> xr.Dataset:
    region = resolve_region(region)
    x = ds["SCHISM_hgrid_node_x"].values
    y = ds["SCHISM_hgrid_node_y"].values
    tri = ds["SCHISM_hgrid_face_nodes"].values[:, :3].astype(int) - 1  # convert to 0-based
    nodes_mask, tri_mask, new_tri = crop(x, y, tri, region)
    if len(nodes_mask) == 0 or len(new_tri) == 0:
        raise ValueError("No nodes found inside the region.")

    n_orig = len(x)
    n_clipped = len(nodes_mask)
    logger.info(f"Clipping to region bounds: {region.bounds}")
    logger.info(f"Nodes: {n_orig} → {n_clipped}")
    logger.info(f"Faces: {len(tri)} → {len(new_tri)}")
    logger.info(f"Faces: {new_tri}")
    ds_clipped = xr.Dataset(
        coords={"time": ds["time"]},
        data_vars={
            "SCHISM_hgrid_node_x": ("nSCHISM_hgrid_node", x[nodes_mask]),
            "SCHISM_hgrid_node_y": ("nSCHISM_hgrid_node", y[nodes_mask]),
            "SCHISM_hgrid_face_nodes": (
                ("nSCHISM_hgrid_face", "nSCHISM_hgrid_face_nodes"),
                new_tri + 1,
            ),
        },
    )
    for var in ds.data_vars:
        if "nSCHISM_hgrid_node" in ds[var].dims:
            ds_clipped[var] = ds[var].isel(nSCHISM_hgrid_node=nodes_mask)
        else:
            logger.warning(f"Variable '{var}' does not depend on nodes, skipping copy..")
    return ds_clipped


def clip_zarr(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    region: shapely.Polygon | tuple | None = None,
    overwrite: bool = False,
) -> None:
    """
    Clip a SCHISM zarr store to a region and write a new store.

    Parameters
    ----------
    input_path
        Path to the source zarr store.
    output_path
        Path for the clipped zarr store.
    region
        Bounding box as (lon_min, lat_min, lon_max, lat_max), a shapely
        polygon, or ``None`` for the full world.
    overwrite
        Overwrite existing output store.
    """
    ds = xr.open_zarr(input_path, chunks={})

    ds_clipped = clip_ds(ds, region)

    if overwrite and output_path.exists():
        import shutil

        shutil.rmtree(output_path)

    ds_clipped.to_zarr(output_path, mode="w")
    logger.info(f"Clipped store written to {output_path}")
