# spost

Post processing tools for SCHISM output.

## Installation

```bash
git clone https://github.com/seareport/spost.git
cd spost
pip install -e .
```

There are three main subcommands:
 1. Extract: Convert and extract SCHISM files to readable outputs.
 2. Plot: Produce graphs from SCHISM outputs
 3. Skill: Compute skill by comparing against in-situ data or other datasets

The `skill` group hosts the full validation pipeline
(`pip install spost[validate]`):

| Command | Purpose |
| --- | --- |
| `spost skill fetch-obs` | Download IOC observations and apply `ioc_cleanup` transformations. |
| `spost skill compare` | Align model and obs and compute `seastats` skill metrics. |
| `spost skill tidal` | Full-mesh tidal harmonic decomposition (`pytides2` + `joblib`). |
| `spost skill report` | Render an HTML/PDF validation report. |
| `spost skill validate` | Run `fetch-obs` → `compare` → `report` end-to-end. |

## Usage

### Extract

Convert and extract SCHISM files to readable outputs.

#### Clip

```bash
$ spost extract clip
Usage: spost [OPTIONS]

Clip a SCHISM zarr store to a bounding box and write a new store.

╭─ Parameters ─────────────────────────────────────────────────────────────────────────────────────╮
│ --input-path                Path to the source zarr store.                                       │
│ --output-path               Path for the clipped zarr store.                                     │
│ --overwrite --no-overwrite  Overwrite existing output store. [default: False]                    │
│ --bbox                      Bounding box as (lon_min, lat_min, lon_max, lat_max) or a region     │
│                             string. Attention! If lon_min starts negative you need to use a =    │
│                             sign, eg: --bbox="-4 0 10 30" [choices: chile, north_sea, channel,   │
│                             gascogne, japan, southeast_us, west_us, east_us, northwest_atlantic, │
│                             indian, europe, med, world] [default: world]                         │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

#### Stations

```bash
$ spost extract stations
Usage: spost [OPTIONS] [ARGS...]

Extract station time series to parquet files.                                                       

Reads station.in and staout_* files from one or more SCHISM outputs/ directories (for hotstart      
segments) and writes per-station parquet files, organized by variable.

╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────╮
│ OUTPUTS_DIRS                                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Parameters ─────────────────────────────────────────────────────────────────────────────────────╮
│ --output-path             Base directory for parquet output [default: stations]                  │
│ --staout-indices          Which staout indices to process (1-9). Defaults to auto-detect.        │
│   --empty-staout-indices                                                                         │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

#### To-Zarr

```bash
$ spost extract to-zarr
Usage: spost [OPTIONS]

Convert SCHISM output to Zarr format.                                                               

Supported variables:                                                                                

 • elevation                                                                                        
 • depth_average_velocity_x                                                                         
 • depth_average_velocity_y                                                                         
 • salinity                                                                                         
 • temperature.

╭─ Parameters ─────────────────────────────────────────────────────────────────────────────────────╮
│ --input-path                   Path to SCHISM run directory.                                     │
│ --output                       Output zarr path. Defaults to                                     │
│                                {input_path}/{input_path.stem}.zarr.                              │
│ --variables --empty-variables  Which variables to include. [default: ['all']]                    │
│ --workers                      Number of parallel workers. [default: 12]                         │
│ --clevel                       Compression level (1-9). [default: 3]                             │
│ --overwrite --no-overwrite     Overwrite existing store. [default: False]                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Plot

#### To-PNGs

```bash
$ spost plot to-pngs
Usage: spost [OPTIONS]

Render a variable from a zarr store to PNG frames.

╭─ Parameters ─────────────────────────────────────────────────────────────────────────────────────╮
│ --input-path                Path to the zarr store.                                              │
│ --variable                  Variable name to render.                                             │
│ --output-path               Directory for output PNGs. Defaults to ./{variable}_pngs/.           │
│ --width                     Image width in pixels. [default: 1920]                               │
│ --height                    Image height in pixels. [default: 1080]                              │
│ --cmap                      Colorcet colormap name (e.g. 'coolwarm', 'fire', 'rainbow')          │
│                             [default: coolwarm]                                                  │
│ --overwrite --no-overwrite  Re-render PNGs even if they already exist. [default: True]           │
│ --clip                      Clip the rendering to bbox as (lon_min, lat_min, lon_max, lat_max)   │
│                             or a region string. Attention! If lon_min starts negative you need   │
│                             to use a = sign, eg: --clip="-4 0 10 30" [choices: chile, north_sea, │
│                             channel, gascogne, japan, southeast_us, west_us, east_us,            │
│                             northwest_atlantic, indian, europe, med, world]                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

#### To-MP4


```bash
$ spost plot to-mp4
Usage: spost [OPTIONS]

Render a variable from a zarr store to an MP4 video.                                                

Generates PNGs first, then stitches them with ffmpeg.

╭─ Parameters ─────────────────────────────────────────────────────────────────────────────────────╮
│ --input-path                Path to the zarr store.                                              │
│ --variable                  Variable name to render.                                             │
│ --output-path               Output MP4 file path. Defaults to ./{variable}.mp4.                  │
│ --width                     Image width in pixels. [default: 1920]                               │
│ --height                    Image height in pixels. [default: 1080]                              │
│ --framerate                 Video framerate (fps). [default: 48]                                 │
│ --cmap                      Colorcet colormap name. [default: coolwarm]                          │
│ --png-dir                   Directory for intermediate PNGs. Defaults to a hidden dir next to    │
│                             output.                                                              │
│ --overwrite --no-overwrite  Re-render PNGs and MP4 even if they already exist. [default: False]  │
│ --clip                      Clip the rendering to bbox as (lon_min, lat_min, lon_max, lat_max)   │
│                             or a region string. Attention! If lon_min starts negative you need   │
│                             to use a = sign, eg: --clip="-4 0 10 30" [choices: chile, north_sea, │
│                             channel, gascogne, japan, southeast_us, west_us, east_us,            │
│                             northwest_atlantic, indian, europe, med, world]                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Skill

The `skill` group houses the validation pipeline. Python API lives at
`spost.validate`; the CLI is exposed as `spost skill <stage>`. Heavy
dependencies (`searvey`, `ioc_cleanup`, `seastats`, `pytides2`, `joblib`,
`matplotlib`, `jinja2`) are gated behind the `validate` extra:

```bash
pip install spost[validate]
```

Common workflow:

```bash
# 1. Extract station timeseries from SCHISM output.
spost extract stations 100/20200101.00/outputs --output-path 100/stations

# 2. Run the full pipeline (fetch-obs → compare → report) for run 100.
spost skill validate --run 100 --start 2020-01-01 --end 2020-12-31

# Individual stages:
spost skill fetch-obs --run 100 --start 2020-01-01 --end 2020-12-31
spost skill compare   --run 100 --start 2020-01-01 --end 2020-12-31 --spinup-days 5
spost skill report    --run 100 --format html

# Heavy full-mesh tidal decomposition (kept separate from `validate`).
spost skill tidal --run 100 --start 2020-01-01 --end 2021-01-01 \
  --chunk-size 200 --n-jobs 32

# Pull tidal maps into the report.
spost skill report --run 100 --tides-nc 100_tides.nc
```

Default path resolution (overridable via CLI or TOML):

| Argument | Default |
| --- | --- |
| `--zarr-path` | `./{run}.zarr` |
| `--output-dir` | `./{run}.validation/` |
| `--station-data-path` | `./{run}/stations/{variable}` (falls back to `./stations/{variable}`) |
| `--meta-parquet` | `ioc_cleanup.get_meta()` |
| `--transformations-dir` | `ioc_cleanup.get_transformations_dir()` |

Defaults can also live in `pyproject.toml` (auto-discovered, hydrogen-style).
Tables follow the command path:

```toml
[spost.skill.validate]
output_dir = "./validation/"
report_format = "html"
spinup_days = 5

[spost.skill.tidal]
chunk_size = 100
n_jobs = -1
resample_minutes = 60

[spost.skill.report]
format = "html"
include_timeseries = true
include_scatter = true
include_taylor = true
include_tidal_maps = true
include_map = true
include_summary = true
```

A different config file can be supplied with `--config /path/to/file.toml`
(handled by the meta entry point on the main app).

#### Incremental runs

`spost skill validate` writes a `state.json` to the validation output
directory and uses it to skip already-fetched data on subsequent runs.
Pass `--force` to ignore prior state and reprocess everything. The IOC raw
data cache lives at `~/.cache/spost/ioc/` (override via `SPOST_CACHE_DIR`
or `XDG_CACHE_HOME`).

#### Python API

```python
from spost.validate import fetch_obs, compare, tidal_analysis, report

fetch_obs(start="2020-01-01", end="2020-12-31", run="100")
compare(start="2020-01-01", end="2020-12-31", run="100", spinup_days=5)
tidal_analysis(start="2020-01-01", end="2021-01-01", run="100")
report(run="100", tides_nc="100_tides.nc")
```
