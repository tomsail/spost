"""Mutually-exclusive ``--bbox`` / ``--wkt`` region arguments.

Replaces the previous ``RegionName`` literal + ``parse_bbox`` machinery with
two explicit arguments:

* ``--bbox lon_min lat_min lon_max lat_max`` — an explicit axis-aligned box.
* ``--wkt PATH_OR_NAME`` — a WKT polygon. Either a path to a ``.wkt`` file on
  disk, or the bare name (with or without the extension) of one of the
  predefined regions bundled under ``spost/regions/``.

The two are wired into a ``cyclopts.Group`` with
``validators.MutuallyExclusive()`` so the CLI rejects supplying both at once.
"""

from __future__ import annotations

import pathlib
from typing import Annotated

import cyclopts
from cyclopts import Group, Parameter, validators
import shapely


_REGIONS_DIR = pathlib.Path(__file__).parent / "regions"


REGION_GROUP = Group("Region", validator=validators.MutuallyExclusive())


BboxArg = Annotated[
    tuple[float, float, float, float] | None,
    Parameter(
        group=REGION_GROUP,
        help=(
            "Bounding box as four floats: lon_min lat_min lon_max lat_max. "
            "If lon_min is negative use `--bbox=...` to avoid argparse "
            'eating the sign, eg --bbox="-4 0 10 30".'
        ),
    ),
]

WktArg = Annotated[
    pathlib.Path | None,
    Parameter(
        group=REGION_GROUP,
        help=(
            "Path to a WKT polygon file, or the name of a bundled region "
            "(see spost/regions/*.wkt)."
        ),
    ),
]


def list_bundled_regions() -> list[str]:
    """Names of the predefined regions shipped with the package."""
    return sorted(p.stem for p in _REGIONS_DIR.glob("*.wkt"))


def _resolve_wkt_path(value: pathlib.Path) -> pathlib.Path:
    """Resolve ``value`` to an existing WKT file.

    Tries the literal path first; otherwise interprets the value as a
    bundled region name (with or without the ``.wkt`` extension).
    """
    if value.exists():
        return value
    name = pathlib.Path(value).with_suffix(".wkt").name
    candidate = _REGIONS_DIR / name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"WKT region not found: {value!r}. "
        f"Available bundled regions: {list_bundled_regions()}"
    )


def resolve_region(
    bbox: tuple[float, float, float, float] | None,
    wkt: pathlib.Path | None,
) -> shapely.Polygon | None:
    """Return a shapely polygon for the active region, or ``None`` for the world.

    The mutual-exclusion is enforced by the cyclopts group at parse time;
    this function still checks defensively so Python callers can't bypass it.
    """
    if bbox is not None and wkt is not None:
        raise cyclopts.ValidationError("--bbox and --wkt are mutually exclusive")
    if bbox is not None:
        return shapely.box(*bbox)
    if wkt is not None:
        path = _resolve_wkt_path(wkt)
        return shapely.from_wkt(path.read_text().strip())
    return None
