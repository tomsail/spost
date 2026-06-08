from __future__ import annotations

import logging
import pathlib
from collections.abc import Sequence

logger = logging.getLogger(__name__)

_METRICS_FILE = "metrics.parquet"


def sim_on_obs(sim, obs):
    """
    Align model series onto obs index via outer merge + linear interpolation.

    Returns
    -------
    sim_aligned, obs_aligned : pd.Series
        Aligned model and obs series, indexed by the obs timestamps. Model values are linearly interpolated to obs timestamps, and any obs timestamps outside the model range are dropped.
    """
    import pandas as pd

    obs = pd.Series(obs, name="obs")
    sim = pd.Series(sim, name="sim")
    df = pd.merge(sim, obs, left_index=True, right_index=True, how="outer")
    df["sim"] = df["sim"].interpolate(method="linear", limit_direction="both")
    df = df.dropna(subset=["obs"])
    obs_ = df["obs"]#.drop_duplicates()
    sim_ = df["sim"]#.drop_duplicates()
    return sim_, obs_


def compare(
    *,
    sim_dir: pathlib.Path,
    obs_dir: pathlib.Path,
    output_path: pathlib.Path | None = None,
    variables: Sequence[str] = ("elev",),
    overwrite: bool = False
) -> pathlib.Path:
    """Align model and obs time-series and compute seastats skill metrics.

    For each obs parquet file in ``obs_dir`` (named ``{station}_{sensor}.parquet``),
    looks for a matching model parquet file under
    ``sim_dir/{primary_var}/*{station}*.parquet``, aligns the two
    series, and computes seastats general + storm metrics.

    Results are written to ``output_path/metrics.parquet``.
    Returns the path to the metrics file.
    """
    import pandas as pd
    import seastats

    if output_path is None:
        output_path = pathlib.Path("comparisons") / f"{sim_dir.name}_{obs_dir.name}"
        if output_path.exists() and not overwrite:
            raise ValueError(f"Output path {output_path} already exists - set overwrite=True to overwrite")

    output_path.mkdir(parents=True, exist_ok=True)

    primary_var = variables[0]
    model_dir = sim_dir / primary_var

    rows: list[dict] = []

    for obs_file in sorted(obs_dir.glob("*.parquet")):
        parts = obs_file.stem.split("_", 1)
        if len(parts) != 2:
            logger.warning("Unexpected obs filename %s - skipping", obs_file.name)
            continue
        station_code, sensor = parts

        model_matches = list(model_dir.glob(f"*{station_code}*.parquet"))
        if not model_matches:
            logger.info("No model file for %s in %s - skipping", station_code, model_dir)
            continue
        model_file = model_matches[0]

        try:
            obs_df = pd.read_parquet(obs_file)
            if obs_df.index.tz is not None:
                obs_df.index = obs_df.index.tz_convert("UTC").tz_localize(None)
            obs_s = obs_df.iloc[:, 0]

            model_df = pd.read_parquet(model_file)
            if model_df.index.tz is not None:
                model_df.index = model_df.index.tz_convert("UTC").tz_localize(None)
            model_s = (
                model_df[primary_var]
                if primary_var in model_df.columns
                else model_df.iloc[:, 0]
            )
        except Exception as exc:
            logger.warning("Could not load data for %s: %s", station_code, exc)
            continue

        model_aligned, obs_aligned = sim_on_obs(model_s, obs_s)
        if model_aligned.empty or obs_aligned.empty:
            logger.info("Empty overlap for %s - skipping", station_code)
            continue

        try:
            normal_stats = seastats.get_stats(model_aligned, obs_aligned, seastats.GENERAL_METRICS_ALL)
        except Exception as exc:
            logger.warning("Could not compute normal stats for %s: %s", station_code, exc)
            continue

        try:
            storm_stats = seastats.get_stats(
                model_aligned,
                obs_aligned,
                seastats.STORM_METRICS,
                quantile=0.95,
            )
        except Exception as exc:
            logger.warning("Storm stats failed for %s: %s", station_code, exc)
            storm_stats = {m: None for m in seastats.STORM_METRICS}

        row: dict = {"station": station_code, "sensor": sensor, **normal_stats, **storm_stats}
        for key in ("lon", "lat"):
            val = obs_df.attrs.get(key)
            if val is not None:
                row[key] = float(val)

        rows.append(row)
        logger.info("Computed metrics for %s", station_code)

    metrics_df = pd.DataFrame(rows).set_index("station") if rows else pd.DataFrame()
    metrics_path = output_path / _METRICS_FILE
    metrics_df.to_parquet(metrics_path)

    logger.info("compare: wrote metrics for %d stations to %s", len(rows), metrics_path)
    return metrics_path
