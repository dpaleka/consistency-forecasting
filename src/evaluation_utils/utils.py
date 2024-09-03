import yaml
from pathlib import Path
from datetime import datetime
from typing import Any

from forecasters import (
    Forecaster,
    AdvancedForecaster,
    BasicForecaster,
    COT_Forecaster,
)
import json

from forecasters.consistent_forecaster import ConsistentForecaster
from static_checks.Checker import ParaphraseChecker, NegChecker


def write_to_dirs(
    results: list[dict],
    filename: str,
    dirs_to_write: list[Path],
    overwrite: bool = False,
):
    for dir in dirs_to_write:
        if overwrite:
            with open(dir / filename, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")
        else:
            with open(dir / filename, "a", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")


def load_forecaster(
    forecaster_class: str, config_path: str | None, model: str | None
) -> Forecaster:
    match forecaster_class:
        case "AdvancedForecaster":
            if model is not None:
                raise ValueError(
                    "The 'model' parameter should not be set when using AdvancedForecaster. Model configuration should be done through the config file and the 'config_path' parameter."
                )
            print(f"Using AdvancedForecaster config file: {config_path}")
        case _:
            assert model is not None, "Model must be specified for forecaster class"
            print(f"Using model: {model}")

    match forecaster_class:
        case "BasicForecaster":
            return BasicForecaster()
        case "COT_Forecaster":
            return COT_Forecaster()
        case "ConsistentForecaster":
            return ConsistentForecaster(
                hypocrite=BasicForecaster(),
                instantiation_kwargs={"model": model},
                bq_func_kwargs={"model": model},
            )
        case "RecursiveConsistentForecaster":
            return ConsistentForecaster.recursive(
                depth=4,
                hypocrite=BasicForecaster(),
                checks=[ParaphraseChecker(), NegChecker()],
                instantiation_kwargs={"model": model},
                bq_func_kwargs={"model": model},
            )
        case "AdvancedForecaster":
            assert (
                config_path is not None
            ), "Config path must be provided for AdvancedForecaster"
            with open(config_path, "r", encoding="utf-8") as f:
                config: dict[str, Any] = yaml.safe_load(f)
            return AdvancedForecaster(**config)
        case _:
            raise ValueError(f"Invalid forecaster class: {forecaster_class}")


def validate_load_directory(
    run: bool, load_dir: str | None, most_recent_directory: Path
) -> Path:
    if run:
        assert load_dir is None, "LOAD_DIR must be None if RUN is True"
    else:
        if load_dir is None:
            print(f"LOAD_DIR is None, using {most_recent_directory}")
            load_dir = most_recent_directory
        load_dir = Path(load_dir)
        assert (
            load_dir.exists() and load_dir.is_dir()
        ), "LOAD_DIR must be a valid directory"
    return load_dir


def create_output_directory(
    forecaster: Forecaster, model: str | None, base_path: Path, output_dir: str | None
) -> tuple[Path, Path]:
    most_recent_directory = base_path / f"A_{forecaster.__class__.__name__}_most_recent"
    most_recent_directory.mkdir(parents=True, exist_ok=True)

    if output_dir is None:
        timestamp = datetime.now()
        folder_name = (
            f"{forecaster.__class__.__name__}_{timestamp.strftime('%m-%d-%H-%M')}"
        )
        output_directory = base_path / folder_name
    else:
        output_directory = Path(output_dir)

    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"Directory '{output_directory}' created or already exists.")
    return output_directory, most_recent_directory
