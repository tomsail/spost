"""Unit tests for the bbox/wkt region machinery."""

from __future__ import annotations

import pathlib

import cyclopts
import pytest
import shapely

from spost._region import list_bundled_regions, resolve_region


# Names that used to live in REGIONS — every one must still be reachable
# via the WKT fallback so existing scripts keep working with --wkt <name>.
_EXPECTED_REGIONS = {
    "chile",
    "caribbean",
    "west_indies",
    "north_sea",
    "channel",
    "gascogne",
    "japan",
    "west_us",
    "southeast_us",
    "east_us",
    "northwest_atlantic",
    "indian",
    "europe",
    "med",
    "world",
}


def test_all_legacy_regions_are_bundled():
    assert _EXPECTED_REGIONS <= set(list_bundled_regions())


def test_resolve_region_none_returns_none():
    assert resolve_region(None, None) is None


def test_resolve_region_from_bbox():
    poly = resolve_region((-5.5, 30.0, 42.0, 47.5), None)
    assert isinstance(poly, shapely.Polygon)
    assert poly.bounds == (-5.5, 30.0, 42.0, 47.5)


def test_resolve_region_from_bundled_name():
    poly = resolve_region(None, pathlib.Path("med"))
    assert isinstance(poly, shapely.Polygon)
    # ``med`` used to be (-5.5, 30.0, 42.0, 47.5) in REGIONS — make sure the
    # bundled WKT preserves those bounds.
    assert poly.bounds == (-5.5, 30.0, 42.0, 47.5)


def test_resolve_region_from_explicit_wkt_file(tmp_path: pathlib.Path):
    wkt_file = tmp_path / "custom.wkt"
    wkt_file.write_text("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")
    poly = resolve_region(None, wkt_file)
    assert poly.bounds == (0.0, 0.0, 1.0, 1.0)


def test_resolve_region_rejects_both():
    with pytest.raises(cyclopts.ValidationError):
        resolve_region((-5, 30, 42, 47), pathlib.Path("med"))


def test_resolve_region_unknown_name():
    with pytest.raises(FileNotFoundError):
        resolve_region(None, pathlib.Path("not_a_real_region"))
