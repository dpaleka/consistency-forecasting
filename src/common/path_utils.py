from pathlib import Path


def get_src_path() -> Path:
    return Path(__file__).parent.parent.resolve()


def get_data_path() -> Path:
    return get_src_path() / "data"


def get_root_path() -> Path:
    return get_src_path().parent


def get_scripts_path() -> Path:
    return get_root_path() / "scripts"


def get_logs_path() -> Path:
    return get_root_path() / "logs"
