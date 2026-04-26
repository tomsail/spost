"""Local cache for raw IOC observations.

Cache key: ``(station_code, start, end)``. Layout::

    ~/.cache/spost/ioc/{station_code}/{start}__{end}.nc

A small JSON manifest sits next to each file so we can detect partial overlaps
and extend the cache without re-downloading.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib


def _default_cache_dir() -> pathlib.Path:
    base = os.environ.get("SPOST_CACHE_DIR")
    if base:
        return pathlib.Path(base) / "ioc"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return pathlib.Path(xdg) / "spost" / "ioc"
    return pathlib.Path.home() / ".cache" / "spost" / "ioc"


def cache_dir() -> pathlib.Path:
    path = _default_cache_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _key(station_code: str, start: datetime.datetime, end: datetime.datetime) -> str:
    raw = f"{station_code}|{start.isoformat()}|{end.isoformat()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def cache_path(
    station_code: str,
    start: datetime.datetime,
    end: datetime.datetime,
) -> pathlib.Path:
    """Return the cache file path for a given (station, window) tuple."""
    key = _key(station_code, start, end)
    station_dir = cache_dir() / station_code
    station_dir.mkdir(parents=True, exist_ok=True)
    return station_dir / f"{key}.nc"


def manifest_path(cache_file: pathlib.Path) -> pathlib.Path:
    return cache_file.with_suffix(".json")


def read_manifest(cache_file: pathlib.Path) -> dict | None:
    mf = manifest_path(cache_file)
    if not mf.exists():
        return None
    try:
        return json.loads(mf.read_text())
    except json.JSONDecodeError:
        return None


def write_manifest(
    cache_file: pathlib.Path,
    station_code: str,
    start: datetime.datetime,
    end: datetime.datetime,
) -> None:
    mf = manifest_path(cache_file)
    payload = {
        "station_code": station_code,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    mf.write_text(json.dumps(payload, indent=2))
