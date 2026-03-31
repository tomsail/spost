"""Station data extraction from SCHISM output."""

from __future__ import annotations

import datetime
import functools
import pathlib
import typing
from collections.abc import Sequence
from zoneinfo import ZoneInfo

import multifutures as mf
import numpy as np

if typing.TYPE_CHECKING:
    import pandas as pd

UTC = ZoneInfo("UTC")

STAOUT_VARIABLES: dict[int, str] = {
    1: "elev",
    2: "air_pressure",
    3: "windx",
    4: "windy",
    5: "temperature",
    6: "salinity",
    7: "u",
    8: "v",
    9: "w",
}


def parse_station_in(path: pathlib.Path) -> pd.DataFrame:
    """
    Parse a SCHISM station.in file.

    Returns a DataFrame with columns:
        mesh_lon, mesh_lat, mesh_index, lon, lat, distance, depth,
        provider, provider_id, location.
    """
    import pandas as pd

    usecols = (1, 2, 7, 8, 9, 10, 11, 12, 13, 14)
    names = [
        "mesh_lon",
        "mesh_lat",
        "mesh_index",
        "lon",
        "lat",
        "distance",
        "depth",
        "provider",
        "provider_id",
        "location",
    ]
    df = pd.read_csv(
        path,
        skiprows=2,
        header=None,
        sep=r"\t",
        usecols=usecols,
        names=names,
        engine="python",
    )
    return df


def parse_start_date_from_param_nml(path: pathlib.Path) -> datetime.datetime:
    """Extract the simulation start date from a SCHISM param.out.nml file."""
    year = month = day = hour = minute = second = -1
    with path.open("r") as fd:
        for line in fd.readlines():
            if "START_YEAR=" in line:
                year = int(line.split("=")[-1].split(",")[0].strip())
            elif "START_MONTH=" in line:
                month = int(line.split("=")[-1].split(",")[0].strip())
            elif "START_DAY=" in line:
                day = int(line.split("=")[-1].split(",")[0].strip())
            elif "START_HOUR=" in line:
                schism_hours = float(line.split("=")[-1].split(",")[0].strip())
                hour = int(schism_hours)
                minute = int((schism_hours - hour) * 60)
                second = int((schism_hours - hour - minute / 60) * 3600)
            elif "UTC_START=" in line:
                utc_start = float(line.split("=")[-1].split(",")[0].strip())
                if utc_start != 0:
                    raise ValueError("Non-UTC start times are not supported")

    start_date = datetime.datetime(year, month, day, hour, minute, second, tzinfo=UTC)
    return start_date


def parse_staout(outputs_dir: pathlib.Path, staout_index: int) -> pd.DataFrame:
    """Parse a single staout_{index} file into a time-indexed DataFrame."""
    import pandas as pd

    start_date = parse_start_date_from_param_nml(outputs_dir / "param.out.nml")
    path = outputs_dir / f"staout_{staout_index}"
    array = np.loadtxt(path)
    df = pd.DataFrame(array)
    zero_col = df.pop(0)
    timedelta_index = pd.to_timedelta(
        np.arange(zero_col.iloc[0], zero_col.iloc[0] * len(df) + 1, zero_col.iloc[0]),
        unit="s",
    )
    index = start_date + timedelta_index
    index = index.rename("time")
    df = df.set_index(index)
    df.columns -= 1
    return df


def merge_staouts(outputs_dirs: Sequence[pathlib.Path], staout_index: int = 1) -> pd.DataFrame:
    """Merge staout files across multiple output directories (hotstart segments)."""
    import pandas as pd

    results = mf.multiprocess(
        func=functools.partial(parse_staout, staout_index=staout_index),
        func_kwargs=[{"outputs_dir": d} for d in outputs_dirs],
        check=True,
    )
    df = pd.concat([r.result for r in results], ignore_index=False).sort_index()
    return df


def _extract_stations(
    outputs_dirs: Sequence[pathlib.Path],
    output_path: pathlib.Path,
    staout_indices: Sequence[int] | None = None,
) -> None:
    """
    Extract station time series to parquet files.

    For each staout index and each station, writes a parquet file named
    ``{provider}-{provider_id}.parquet`` into a subdirectory of output_path
    named after the variable.

    Parameters
    ----------
    outputs_dirs
        One or more SCHISM outputs/ directories (for hotstart segments).
    output_path
        Base directory for parquet output.
    staout_indices
        Which staout indices to process. Defaults to all known (1-9).
    """
    if staout_indices is None:
        # Auto-detect which staout files exist
        staout_indices = [idx for idx in STAOUT_VARIABLES if (outputs_dirs[0] / f"staout_{idx}").exists()]

    station_data = parse_station_in(outputs_dirs[0] / "station.in")

    for idx in staout_indices:
        var_name = STAOUT_VARIABLES[idx]
        var_dir = output_path / var_name
        var_dir.mkdir(parents=True, exist_ok=True)

        df = merge_staouts(outputs_dirs, staout_index=idx)

        func_kwargs = []
        for station_index, station_meta in station_data.iterrows():
            station_df = df[[station_index]].rename(columns={station_index: var_name})
            station_df.attrs.update(station_meta.to_dict())
            filename = f"{station_meta['provider']}-{station_meta['provider_id'].replace(' ', '_')}.parquet"
            func_kwargs.append({"df": station_df, "target": var_dir / filename})

        _ = mf.multithread(
            func=_to_parquet,
            func_kwargs=func_kwargs,
            check=True,
        )


def _to_parquet(df: pd.DataFrame, target: pathlib.Path) -> None:
    df.to_parquet(target)
