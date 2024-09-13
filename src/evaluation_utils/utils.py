from pathlib import Path
from datetime import datetime

from forecasters import (
    Forecaster,
)
import json


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
    forecaster: Forecaster, base_path: Path, output_dir: str | None
) -> tuple[Path, Path]:
    most_recent_directory = base_path / f"A_{forecaster.__class__.__name__}_most_recent"
    most_recent_directory.mkdir(parents=True, exist_ok=True)

    if output_dir is None:
        timestamp = datetime.now()
        folder_name = (
            # dirty hack: we explicitly don't round the seconds because we want the neighboring runs to write to the same dir
            f"{forecaster.__class__.__name__}_{timestamp.strftime('%m-%d-%H-%M')}"
        )
        output_directory = base_path / folder_name
    else:
        output_directory = Path(output_dir)

    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"Directory '{output_directory}' created or already exists.")
    return output_directory, most_recent_directory
