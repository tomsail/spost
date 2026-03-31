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

Compute skill scores comparing model output to observations. (Not yet implemented)

```bash
spost skill --help
```

Available subcommands:
- `tide-stations`: Not implemented yet.
- `tides-grid`: Not implemented yet.
