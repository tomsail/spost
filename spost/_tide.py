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

    from ._utils import harmonic_analysis
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

    # Multiprocessing part
    result = harmonic_analysis(S, chunk_size=chunk_size)

    lons = data.SCHISM_hgrid_node_x.values
    lats = data.SCHISM_hgrid_node_y.values

    coords = {
        "nSCHISM_hgrid_node": data.nSCHISM_hgrid_node,
        "constituent": np.array(FULL, dtype=object),
        "SCHISM_hgrid_node_x": ("nSCHISM_hgrid_node", lons),
        "SCHISM_hgrid_node_y": ("nSCHISM_hgrid_node", lats),
    }
    coef_da = xr.DataArray(
        result,
        dims=("nSCHISM_hgrid_node", "constituent"),
        coords=coords,
        name="model",
    )
    coef_ds = coef_da.to_dataset()

    # Add FES or TPXO
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
        coef_ds["SCHISM_hgrid_face_nodes"] = data["SCHISM_hgrid_face_nodes"]

    coef_ds.to_zarr(
        output,
        mode="w",
        zarr_format=3,
        encoding=encoding,
        consolidated=True,
    )


def compute_sal(
    *,
    input_path: pathlib.Path,
    fes: pathlib.Path,
    output_dir: pathlib.Path = pathlib.Path("."),
    constituents: list[str] | None = None,
    overwrite: bool = False,
):
    """Generate SCHISM self-attraction & loading (SAL) gr3 files from FES load tide.

    The mesh (node coordinates and triangular connectivity) is read from the
    same SCHISM zarr store used by ``compute_tidemap`` (``SCHISM_hgrid_node_x/y``
    and ``SCHISM_hgrid_face_nodes``), so no separate ``hgrid.gr3`` parsing is
    needed. Writes one ``loadtide_<C>.gr3`` per constituent, each node line
    holding the load-tide amplitude (metres) and Greenwich phase (degrees).
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import numpy as np
    import xarray as xr

    from ._utils import build_faces
    from ._utils import interpolate_load_tide
    from ._utils import SAL

    constituents = list(constituents) if constituents else list(SAL)

    input_path = pathlib.Path(input_path)
    fes = pathlib.Path(fes)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fail early if any output exists and we are not overwriting.
    if not overwrite:
        existing = [output_dir / f"loadtide_{c}.gr3" for c in constituents]
        clash = [p for p in existing if p.exists()]
        if clash:
            raise ValueError(
                f"Output file(s) already exist: {[str(p) for p in clash]}. "
                "Pass overwrite=True (or --overwrite) to replace them."
            )

    data = xr.open_zarr(input_path)
    if "SCHISM_hgrid_face_nodes" not in data:
        raise ValueError(
            f"{input_path} has no 'SCHISM_hgrid_face_nodes'; cannot write mesh "
            "connectivity for the gr3 files."
        )
    x = data.SCHISM_hgrid_node_x.values
    y = data.SCHISM_hgrid_node_y.values
    elements = build_faces(data)  # (ne, 3), 0-based
    nn, ne = len(x), elements.shape[0]

    print(f"Reading FES load tide for {constituents} from {fes} ...")
    z = interpolate_load_tide(fes, x, y, constituents)  # (nn, nc) complex, metres

    # pyTMD FES convention: z = A * exp(-i * G) -> A = |z|, G = -arg(z)
    amp = np.abs(z)
    phase = np.angle(z,deg=True) % 360.0

    # Nodes outside the FES domain -> no load contribution.
    nan_mask = np.isnan(z)
    print("number of NaNs",nan_mask.sum())
    amp[nan_mask] = 0.0
    phase[nan_mask] = 0.0
    n_filled = int(nan_mask.any(axis=1).sum())
    if n_filled:
        print(f"Filled {n_filled}/{nn} nodes outside FES coverage with amplitude 0.")

    written = []
    for ic, c in enumerate(constituents):
        outname = output_dir / f"loadtide_{c}.gr3"
        with outname.open("w") as f:
            f.write(f"{c.lower()}\n")
            f.write(f"{ne} {nn}\n")
            for i in range(nn):
                f.write(
                    f"{i + 1} {x[i]:.6f} {y[i]:.6f} {amp[i, ic]:.6f} {phase[i, ic]:.6f}\n"
                )
            for j in range(ne):
                n1, n2, n3 = (elements[j] + 1)
                f.write(f"{j + 1} 3 {n1} {n2} {n3}\n")
        written.append(outname)
        print(f"Written {outname}")

    return written
