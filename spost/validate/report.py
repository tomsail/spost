"""Validation report generation.

Reads ``metrics.parquet`` plus the per-station comparison netCDFs (and
optionally the full-mesh tidal output) and renders a self-contained HTML or
PDF report via Jinja2.
"""
from __future__ import annotations

import base64
import datetime
import io
import logging
import pathlib
from collections.abc import Sequence

from . import _paths

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = pathlib.Path(__file__).parent / "templates"
_OBS_DIR = "obs_ts"
_PER_STATION_DIR = "per_station"
_METRICS_FILE = "metrics.parquet"


def _spost_version() -> str:
    try:
        from importlib.metadata import version

        return version("spost")
    except Exception:
        return "unknown"


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _plot_timeseries(code: str, comparison_path: pathlib.Path) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray as xr

    try:
        ds = xr.open_dataset(comparison_path)
    except Exception as exc:
        logger.warning("Skipping timeseries for %s: %s", code, exc)
        return None
    fig, ax = plt.subplots(figsize=(9, 3))
    ds["model"].plot(ax=ax, label="model", lw=1.0)
    ds["obs"].plot(ax=ax, label="obs", lw=1.0, alpha=0.8)
    ax.set_title(f"{code} - timeseries")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_scatter(code: str, comparison_path: pathlib.Path) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    try:
        ds = xr.open_dataset(comparison_path)
    except Exception as exc:
        logger.warning("Skipping scatter for %s: %s", code, exc)
        return None
    m = ds["model"].to_pandas().to_numpy()
    o = ds["obs"].to_pandas().to_numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(o, m, s=4, alpha=0.4)
    if len(o):
        lo = float(min(np.nanmin(o), np.nanmin(m)))
        hi = float(max(np.nanmax(o), np.nanmax(m)))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel("obs")
    ax.set_ylabel("model")
    ax.set_title(f"{code} - scatter")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_taylor(metrics_df) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if metrics_df is None or metrics_df.empty:
        return None
    if "corr" not in metrics_df.columns or "rmse" not in metrics_df.columns:
        return None
    fig, ax = plt.subplots(figsize=(5, 4), subplot_kw={"projection": "polar"})
    corr = metrics_df["corr"].to_numpy()
    rmse = metrics_df["rmse"].to_numpy()
    theta = np.arccos(np.clip(corr, -1.0, 1.0))
    ax.scatter(theta, rmse, s=20)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_title("Taylor diagram (RMSE vs correlation)")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_tidal_maps(tides_nc: pathlib.Path, constituents: Sequence[str]) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray as xr

    plots: list[dict] = []
    try:
        ds = xr.open_dataset(tides_nc)
    except Exception as exc:
        logger.warning("Could not open tides nc %s: %s", tides_nc, exc)
        return plots
    if "tides" not in ds:
        return plots
    if "SCHISM_hgrid_node_x" not in ds or "SCHISM_hgrid_node_y" not in ds:
        return plots
    x = ds["SCHISM_hgrid_node_x"].values
    y = ds["SCHISM_hgrid_node_y"].values
    available = [str(c) for c in ds["constituent"].values]
    for name in constituents:
        if name not in available:
            continue
        amp = ds["tides"].sel(constituent=name, metric="amplitude").values
        fig, ax = plt.subplots(figsize=(7, 4))
        sc = ax.scatter(x, y, c=amp, s=2, cmap="viridis")
        plt.colorbar(sc, ax=ax, label=f"{name} amplitude (m)")
        ax.set_title(f"{name} amplitude")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        plots.append({"constituent": name, "image": _fig_to_b64(fig)})
    return plots


def _metrics_to_html(metrics_df) -> str:
    if metrics_df is None or metrics_df.empty:
        return "<p>No metrics available.</p>"
    return metrics_df.round(4).to_html(classes="metrics", border=0)


def _reference_to_html(metrics_df, reference_path: pathlib.Path | None) -> str | None:
    if reference_path is None or not reference_path.exists() or metrics_df.empty:
        return None
    import pandas as pd

    ref = pd.read_parquet(reference_path)
    common_cols = [c for c in metrics_df.columns if c in ref.columns]
    if not common_cols:
        return None
    delta = (metrics_df[common_cols] - ref[common_cols]).round(4)
    return delta.to_html(classes="reference", border=0)


def _station_map_html(metrics_df) -> str | None:
    """Best-effort folium map; returns None when folium isn't installed."""
    try:
        import folium
    except ImportError:
        return None
    if metrics_df is None or metrics_df.empty:
        return None
    if "lat" not in metrics_df.columns or "lon" not in metrics_df.columns:
        return None
    fmap = folium.Map(location=[float(metrics_df["lat"].mean()), float(metrics_df["lon"].mean())], zoom_start=2)
    for code, row in metrics_df.iterrows():
        rmse = row.get("rmse", float("nan"))
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=4,
            popup=f"{code}: rmse={rmse:.3f}",
        ).add_to(fmap)
    return fmap._repr_html_()


def report(
    *,
    run: str | None = None,
    output_dir: pathlib.Path | None = None,
    metrics_parquet: pathlib.Path | None = None,
    station_data_path: pathlib.Path | None = None,
    obs_dir: pathlib.Path | None = None,
    tides_nc: pathlib.Path | None = None,
    format: str = "html",
    reference_metrics: pathlib.Path | None = None,
    name: str | None = None,
    include_timeseries_plots: bool = True,
    include_scatter_plots: bool = True,
    include_taylor_diagram: bool = True,
    include_tidal_maps: bool = True,
    include_map: bool = True,
    include_summary_table: bool = True,
    tidal_constituents: Sequence[str] = ("M2", "S2", "K1", "O1"),
) -> pathlib.Path:
    """Generate the validation report. Returns the path to the rendered file."""
    try:
        import jinja2
    except ImportError as exc:
        raise ImportError(
            "jinja2 is required for report. Install with: pip install spost[validate]"
        ) from exc
    import pandas as pd

    out_dir = _paths.run_validation_dir(run, output_dir)
    metrics_path = metrics_parquet or out_dir / _METRICS_FILE
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Cannot find {metrics_path}; run `spost compare` first."
        )
    metrics_df = pd.read_parquet(metrics_path)

    per_station_dir = out_dir / _PER_STATION_DIR

    stations: list[dict] = []
    for code in metrics_df.index:
        comparison_path = per_station_dir / f"{code}_comparison.nc"
        if not comparison_path.exists():
            continue
        ts = _plot_timeseries(code, comparison_path) if include_timeseries_plots else None
        sc = _plot_scatter(code, comparison_path) if include_scatter_plots else None
        stations.append({"code": code, "timeseries": ts, "scatter": sc})

    taylor = _plot_taylor(metrics_df) if include_taylor_diagram else None
    map_html = _station_map_html(metrics_df) if include_map else None
    metrics_table = _metrics_to_html(metrics_df) if include_summary_table else None
    reference_table = _reference_to_html(metrics_df, reference_metrics)
    tidal_maps = (
        _plot_tidal_maps(tides_nc, tidal_constituents)
        if (include_tidal_maps and tides_nc is not None and tides_nc.exists())
        else []
    )

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")

    state_path = out_dir / "state.json"
    if state_path.exists():
        import json

        state = json.loads(state_path.read_text())
    else:
        state = {}

    html = template.render(
        name=name,
        run=run,
        start=state.get("start"),
        end=state.get("end"),
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        spost_version=_spost_version(),
        stations=stations,
        metrics_table=metrics_table,
        reference_table=reference_table,
        taylor_plot=taylor,
        map_html=map_html,
        tidal_maps=tidal_maps,
        include_timeseries=include_timeseries_plots,
        include_scatter=include_scatter_plots,
        include_taylor=include_taylor_diagram,
        include_tidal_maps=include_tidal_maps and bool(tidal_maps),
        include_map=include_map and bool(map_html),
        include_summary=include_summary_table,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "report.html"
    if format in ("html", "both"):
        html_path.write_text(html, encoding="utf-8")
        logger.info("report: wrote %s", html_path)

    if format in ("pdf", "both"):
        try:
            from weasyprint import HTML
        except ImportError as exc:
            raise ImportError(
                "weasyprint is required for PDF reports. "
                "Install with: pip install spost[validate-pdf]"
            ) from exc
        pdf_path = out_dir / "report.pdf"
        HTML(string=html).write_pdf(str(pdf_path))
        logger.info("report: wrote %s", pdf_path)
        if format == "pdf":
            return pdf_path

    return html_path
