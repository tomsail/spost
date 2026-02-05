import pathlib
from typing import Annotated

import cyclopts
from cyclopts.types import ExistingDirectory

from spost._to_zarr import to_zarr as _to_zarr, VARIABLE_SPECS

app = cyclopts.App(name="spost", help="Post processing tools for SCHISM output")


@app.command
def to_zarr(
    *,
    input_path: ExistingDirectory | None = None,
    output: pathlib.Path | None = None,
    variables: Annotated[list[str], cyclopts.Parameter(consume_multiple=True)] = ["all"],
    workers: int = 12,
    clevel: int = 3,
    overwrite: bool = False,
):
    """Convert SCHISM output to Zarr format.

    Supported variables:
        - elevation
        - depth_average_velocity_x
        - depth_average_velocity_y
        - salinity
        - temperature.

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
    """
    if input_path is None:
        app["to-zarr"].help_print()
        raise SystemExit(0)
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
    )


def main():
    app()
