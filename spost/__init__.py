import typing

import cyclopts
from cyclopts import Group
from cyclopts.types import ResolvedExistingFile

from ._cli import clip
from ._cli import stations
from ._cli import to_mp4
from ._cli import to_pngs
from ._cli import to_zarr
from .validate._cli import compare
from .validate._cli import fetch_obs
from .validate._cli import report
from .validate._cli import tidal
from .validate._cli import validate

# Store all subcommand apps in a list for easy config application
_Subcommands: list[cyclopts.App] = []


def register_subcommand(parent_app: cyclopts.App, name: str, help: str) -> cyclopts.App:
    """Helper to register subcommands and track them for config setup."""
    subcommand_app = cyclopts.App(name=name, help=help)
    _ = parent_app.command(subcommand_app)
    _Subcommands.append(subcommand_app)
    return subcommand_app


app = cyclopts.App()
app.meta.group_parameters = Group("Parameters",sort_key=None)

# Child apps
extract_app = register_subcommand(app, "extract", "Convert/Extract SCHISM files to readable outputs")
plot_app = register_subcommand(app, "plot", "Produce graphs from SCHISM outputs")
skill_app = register_subcommand(app, "skill", "Validation pipeline: fetch obs, compare, tidal harmonics, report.")

# Groups in extract
zarr_group = Group("Zarr tools", sort_key=0)
staout_group = Group("Staout tools", sort_key=2)
group_commands=Group("Commands", sort_key=3),

# Groups in plot
render_group = Group("Render tools", sort_key=0)

# Groups in skill
station_group = Group("Discrete Station Comparisons", sort_key=0)
map_group = Group("2D analysis", sort_key=1)
report_group = Group("Reporting", sort_key=2)
auto_group = Group("Auto", sort_key=3)

_ = extract_app.command(to_zarr, group=zarr_group)
_ = extract_app.command(clip, group=zarr_group)
_ = plot_app.command(to_pngs, group=render_group)
_ = plot_app.command(to_mp4, group=render_group)
_ = extract_app.command(stations, group=staout_group)
_ = skill_app.command(fetch_obs, group=station_group)
_ = skill_app.command(compare, group=station_group)
_ = skill_app.command(tidal, group=map_group)
_ = skill_app.command(report, group=report_group)
_ = skill_app.command(validate, group=auto_group)

@app.meta.default
def _meta(
    *tokens: typing.Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    config: ResolvedExistingFile | None = None,
    verbose: bool = False,
):
    """
    Post processing tools for SCHISM output

    Args:
        config: Path to configuration file (default: pyproject.toml)
        verbose: Enable verbose output
    """
    import logging

    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")

    if verbose:
        print(f"Loading configuration from: {config}")


    if config is not None:
        config_handler = cyclopts.config.Toml(
            config,
            root_keys=["spost"],
            use_commands_as_keys=True,
            allow_unknown=True,
            search_parents=False,
        )

        # Apply config to main app
        app.config = config_handler

        # Apply config to all registered subcommands
        for sub in _Subcommands:
            sub.config = config_handler

    if verbose and config is not None:
        print(f"Configuration loaded successfully from {config.absolute()}")
    elif verbose:
        print(f"Config file {config} not found, using defaults")

    app(tokens)
