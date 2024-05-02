from pathlib import Path


def get_src_path() -> Path:
    return Path(__file__).parent.parent.resolve()


def get_data_path() -> Path:
    return get_src_path() / "data"
