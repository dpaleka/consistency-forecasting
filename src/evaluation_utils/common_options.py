import click
from pathlib import Path

from common.path_utils import get_src_path, get_data_path

CONFIGS_DIR: Path = get_src_path() / "forecasters/forecaster_configs"
BASE_FORECASTS_OUTPUT_PATH: Path = get_data_path() / "forecasts"


def common_options(f):
    options = [
        click.option(
            "-f",
            "--forecaster_class",
            default="AdvancedForecaster",
            help="Forecaster to use. Can be BasicForecaster, COT_Forecaster, AdvancedForecaster, ConsistentForecaster, RecursiveConsistentForecaster.",
        ),
        click.option(
            "-c",
            "--config_path",
            type=click.Path(),
            default=CONFIGS_DIR / "cheap_gpt4o-mini.yaml",
            help="Path to the configuration file",
        ),
        click.option(
            "-m",
            "--model",
            default=None,
            help="Model to use for BasicForecaster and CoT_Forecaster. Is overridden by the config file in case of AdvancedForecaster.",
        ),
        click.option("-r", "--run", is_flag=True, help="Run the forecaster"),
        click.option(
            "-l",
            "--load_dir",
            required=False,
            type=click.Path(),
            help="Directory to load results from in case run is False. Defaults to most_recent_directory",
        ),
        click.option(
            "-n",
            "--num_lines",
            default=3,
            help="Number of lines to process in each of the files",
        ),
        click.option(
            "--async",
            "is_async",
            is_flag=True,
            default=False,
            help="Await gather the forecaster over all lines in a check",
        ),
        click.option(
            "--output_dir",
            type=click.Path(),
            required=False,
            help=f"Path to the output directory. Will default to timestamped directory in {BASE_FORECASTS_OUTPUT_PATH} otherwise",
        ),
    ]
    for option in reversed(options):
        f = option(f)
    return f
