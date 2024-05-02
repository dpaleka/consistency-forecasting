from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent.resolve()


def get_data_path() -> Path:
    return get_project_root() / "data"
