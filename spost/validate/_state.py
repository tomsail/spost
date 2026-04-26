"""Incremental-state tracker for the validation pipeline.

Each run's validation directory holds a ``state.json`` describing what has
already been processed. ``fetch-obs``/``compare``/``report`` consult this file
to skip work that's already been done, and update it when new data lands.
"""

from __future__ import annotations

import datetime
import json
import pathlib
from dataclasses import asdict, dataclass, field


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _spost_version() -> str:
    try:
        from importlib.metadata import version

        return version("spost")
    except Exception:
        return "unknown"


@dataclass
class ValidationState:
    spost_version: str = field(default_factory=_spost_version)
    last_updated: str = field(default_factory=_utc_now)
    start: str | None = None
    end: str | None = None
    stations_processed: list[str] = field(default_factory=list)
    stations_skipped: list[str] = field(default_factory=list)
    segments_included: list[str] = field(default_factory=list)

    def touch(self) -> None:
        self.last_updated = _utc_now()
        self.spost_version = _spost_version()


def state_path(output_dir: pathlib.Path) -> pathlib.Path:
    return output_dir / "state.json"


def load(output_dir: pathlib.Path) -> ValidationState:
    path = state_path(output_dir)
    if not path.exists():
        return ValidationState()
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return ValidationState()
    return ValidationState(
        spost_version=data.get("spost_version", _spost_version()),
        last_updated=data.get("last_updated", _utc_now()),
        start=data.get("start"),
        end=data.get("end"),
        stations_processed=list(data.get("stations_processed", [])),
        stations_skipped=list(data.get("stations_skipped", [])),
        segments_included=list(data.get("segments_included", [])),
    )


def save(output_dir: pathlib.Path, state: ValidationState) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state.touch()
    state_path(output_dir).write_text(json.dumps(asdict(state), indent=2))


def previous_window_end(state: ValidationState) -> datetime.datetime | None:
    if state.end is None:
        return None
    try:
        return datetime.datetime.fromisoformat(state.end)
    except ValueError:
        return None
