import cyclopts

REGIONS = {
    # (lon_min, lat_min, lon_max, lat_max)
    "chile": (-100.0, -56.0, -66.0, -17.0),
    "north_sea": (-5.0, 51.0, 9.0, 62.0),
    "channel": (-6.0, 48.0, 2.0, 52.0),
    "gascogne": (-10.0, 43.0, -1.0, 48.5),
    "japan": (122.0, 24.0, 146.0, 46.0),
    "west_us": (-130.0, 30.0, -115.0, 50.0),
    "southeast_us": (-90.0, 24.0, -75.0, 36.0),
    "east_us": (-80.0, 35.0, -65.0, 47.0),
    "northwest_atlantic": (-100.0, 40.0, -20.0, 70.0),
    "indian": (20.0, -40.0, 120.0, 30.0),
    "europe": (-50.0, 30.0, 42.0, 66.0),
    "med": (-5.5, 30.0, 42.0, 47.5),
    "world": (-180.0, -78.73, 180.0, 90.0),
}


def parse_bbox(type_, token: str) -> tuple[float, float, float, float]:
    value = token[0].value
    if value in REGIONS:
        return REGIONS[value]
    try:
        parts = value.split()
        if len(parts) != 4:
            raise ValueError
        return tuple(float(p) for p in parts)
    except ValueError:
        valid = ", ".join(REGIONS.keys())
        raise cyclopts.ValidationError(
            f"--bbox must be a named region ({valid}) " f"or 4 floats 'xmin ymin xmax ymax', got: {value!r}"
        )
