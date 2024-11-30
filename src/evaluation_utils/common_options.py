import click
from pathlib import Path
from typing import Any
import yaml

from common.path_utils import get_src_path, get_data_path

CONFIGS_DIR: Path = get_src_path() / "forecasters/forecaster_configs"
BASE_FORECASTS_OUTPUT_PATH: Path = get_data_path() / "forecasts"
PREDEFINED_FORECASTER_CLASSES = [
    "BasicForecaster",
    "BasicForecasterWithExamples",
    "BasicForecasterTextBeforeParsing",
    "CoT_Forecaster",
    "CoT_ForecasterWithExamples",
    "CoT_ForecasterTextBeforeParsing",
    "AdvancedForecaster",
    "ConsistentForecaster",
    "RecursiveConsistentForecaster",
    "LoadForecaster",
    "CrowdForecaster",
    "PromptedToCons_Forecaster",
]


def common_options(f):
    options = [
        click.option(
            "-f",
            "--forecaster_class",
            default=None,
            type=click.Choice(PREDEFINED_FORECASTER_CLASSES),
            help="Forecaster to use.",
        ),
        click.option(
            "--load",
            type=str,
            default=None,
            help="Shorthand for using --load_dir and --tuple_dir to the same directory; and for evaluation, for -k to all that are present in the directory.",
        ),
        click.option(
            "-p",
            "--custom_path",
            type=str,
            default=None,
            help="Either: "
            "(1) Path to the custom forecaster Python module (e.g. `src/forecasters/custom_forecaster.py`). Has to contain exactly one Forecaster subclass. "
            "(2) Path to the custom forecaster Python module, then `::`, then the class name (e.g. `src/forecasters/custom_forecaster.py::CustomForecaster1`). "
            "Only used when forecaster_class is None.",
        ),
        click.option(
            "-c",
            "--config_path",
            type=click.Path(),
            default=None,
            help="Path to the configuration file. Can be used for all forecasters. Do not use the --model option when using this.",
        ),
        click.option(
            "-o",
            "--forecaster_options",
            multiple=True,
            help="Additional options for the forecaster in the format key=value. Can be used multiple times. These options will be passed as kwargs when creating the forecaster. Overrides options in config_path. "
            "If an option is a list, it can be passed as a comma-separated string with quotes and square brackets (e.g. `-o key='[val1, val2, val3]'`). "
            "Alternatively, multiple=True works per key, so `-o key=val1 -o key=val2` works as the same as `-o key='[val1, val2]'`.",
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


def parse_forecaster_options(options: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for option in options:
        key, value = option.split("=")
        value = parse_value(value)
        if key in result:
            if isinstance(result[key], list):
                result[key].append(value)
            else:
                result[key] = [result[key], value]
        else:
            result[key] = value
    return result


def parse_value(value: str) -> Any:
    value = value.strip("\"' ")
    if value.startswith("[") and value.endswith("]"):
        return [parse_value(item.strip()) for item in value[1:-1].split(",")]

    # Handle boolean values
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip()


def get_forecaster_config(
    config_path: str | None, forecaster_options: list[str] | None
) -> dict[str, Any]:
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            forecaster_config: dict[str, Any] = yaml.safe_load(f)
    else:
        forecaster_config = {}

    if forecaster_options is not None:
        forecaster_options_dict = parse_forecaster_options(forecaster_options)

        # override config with forecaster_options
        forecaster_config.update(forecaster_options_dict)

    return forecaster_config
