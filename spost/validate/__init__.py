"""Skill validation pipeline for spost.

Three composable stages plus a heavier full-mesh tidal decomposition:

- :func:`fetch_obs` — download IOC observations, apply ``ioc_cleanup``
  transformations, write per-station netCDFs.
- :func:`compare` — align model and obs on a common grid, hand to
  ``seastats`` for skill metrics.
- :func:`tidal_analysis` — decompose elevation across the full mesh into
  harmonic constituents (separate from ``compare``; runs independently).
- :func:`report` — render an HTML/PDF report from the previous outputs.

Each function is callable from Python with the same arguments the CLI exposes.
"""

from .compare import compare
from .fetch_obs import fetch_obs
from .report import report
from .tidal_analysis import tidal_analysis

__all__ = ["compare", "fetch_obs", "report", "tidal_analysis"]
