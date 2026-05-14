"""Path resolution helpers for the validation subcommands.

Resolution order for any path argument is:

1. Explicit CLI argument (highest priority).
2. TOML config value (handled by ``cyclopts.config.Toml``).
3. Programmatic default (this module).
"""
from __future__ import annotations

import pathlib


def run_root(run: str | None) -> pathlib.Path:
    """Directory holding the per-segment SCHISM outputs (``./{run}/``)."""
    if run is None:
        raise ValueError("--run is required to resolve default paths")
    return pathlib.Path(f"./{run}")


def run_zarr(run: str | None, zarr_path: pathlib.Path | None = None) -> pathlib.Path:
    """Resolve the zarr store for a given run."""
    if zarr_path is not None:
        return zarr_path
    if run is None:
        raise ValueError("--run is required to resolve --zarr-path")
    return pathlib.Path(f"./{run}.zarr")


def get_station_data(
    station_data_path: pathlib.Path | None = None,
) -> pathlib.Path:
    """Resolve the spost station-data directory.
    """
    import ioc_cleanup as C
    ioc = C.get_meta()
    stats = C.calc_statistics(ioc, stations_dir=C.TRANSFORMATIONS_DIR, pattern="*.json")
    print(stats)

    return station_data_path

def default_meta_parquet(meta_parquet: pathlib.Path | None = None) -> pathlib.Path | None:
    """Return the user-supplied path or the bundled ``ioc_cleanup`` meta."""
    if meta_parquet is not None:
        return meta_parquet
    try:
        import ioc_cleanup
    except ImportError:
        return None
    if hasattr(ioc_cleanup, "get_meta"):
        meta = ioc_cleanup.get_meta()
        if isinstance(meta, (str, pathlib.Path)):
            return pathlib.Path(meta)
        # Some versions return a DataFrame; let the caller deal with that.
    return None


def default_transformations_dir(transformations_dir: pathlib.Path | None = None) -> pathlib.Path | None:
    if transformations_dir is not None:
        return transformations_dir
    try:
        import ioc_cleanup
    except ImportError:
        return None
    if hasattr(ioc_cleanup, "get_transformations_dir"):
        return pathlib.Path(ioc_cleanup.get_transformations_dir())
    return None
