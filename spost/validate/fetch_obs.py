from __future__ import annotations

import logging
import pathlib

logger = logging.getLogger(__name__)


def _gather_codes(folder: pathlib.Path, suffix: str, remove_sensor = True, non_existant_fail=True) -> set[str]:
    """
    Gather IOC codes from a folder by looking for files with a given suffix.
    If `sensor_pattern` is True, expects filenames of the form `{code}_{sensor}.parquet` and extracts just the code. Otherwise, expects filenames of the form `{code}.parquet`.
    """
    codes: list[str] = []
    if not folder.exists():
        if non_existant_fail:
            raise ValueError(f"folder {folder} does not exist")
        else:
            logger.warning(f"folder {folder} does not exist - skipping")
        return codes
    for path in folder.iterdir():
        if path.suffix.lower() != suffix:
            continue
        if remove_sensor:
            stem = path.stem
            station, sensor = stem.split("_")
        else:
            station = path.stem
        codes.append(station)
    return codes


def fetch_obs(
    *,
    transformations_dir: pathlib.Path | str,
    raw_data_folder: pathlib.Path,
    output_dir: pathlib.Path,
    overwrite: bool = False,
) -> pathlib.Path:
    import ioc_cleanup as C
    import pandas as pd

    json_codes = _gather_codes(transformations_dir, '.json')
    json_full_codes = _gather_codes(transformations_dir, '.json', remove_sensor=False)
    raw_codes = _gather_codes(raw_data_folder, '.parquet',non_existant_fail=False)
    clean_codes = _gather_codes(output_dir, '.parquet', remove_sensor=False, non_existant_fail=False)

    raw_candidates = sorted(json_codes & raw_codes) if raw_codes else sorted(json_codes)
    clean_candidates = sorted(json_codes & clean_codes) if clean_codes else sorted(json_codes)

    raw_processed: list[str] = []
    clean_processed: list[str] = []
    skipped: list[str] = []

    for code in raw_candidates:
        for year in range(2020, 2026):
            path = raw_data_folder / str(year) / f"{code}.parquet"
            if not path.exists() or overwrite:
                try:
                    C.download_year_station(code, year, raw_data_folder)
                except Exception as exc:
                    logger.warning(f"Failed to fetch {code} for {year}: {exc}")
                    skipped.append(code)
        raw_processed.append(code)
        print(f"Fetched raw data for {code}")

    for icode, code in enumerate(clean_candidates):
        if code in skipped:
            logger.warning(f"Skipping cleaning for {code} since it failed to fetch")
            continue
        try:
            station, sensor = sorted(json_full_codes)[icode].split("_")
            t = C.load_transformation(station, sensor, transformations_dir)
            ts = C.load_station(station, raw_data_folder)
            ts = C.transform(ts, t)[sensor]
            ts.attrs["cleaning_date"] = pd.Timestamp.now().isoformat()
            ts.to_parquet(output_dir / f"{code}_{sensor}.parquet")
        except Exception as exc:
            logger.warning(f"Failed to clean {code}: {exc}")
            skipped.append(code)
        else:
            clean_processed.append(code)
