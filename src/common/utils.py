import json
import aiofiles
from typing import List, Any
import jsonlines
from copy import deepcopy
import hashlib
from pydantic import BaseModel
from common.datatypes import ForecastingQuestion
from datetime import datetime
from pathlib import Path
from typing import Optional


def round_floats(x, precision: int = 3, convert_ints: bool = False) -> Any:
    if isinstance(x, float):
        return round(x, precision)
    if convert_ints and isinstance(x, int):
        return round(float(x), precision)
    if isinstance(x, dict):
        return {k: round_floats(v, precision, convert_ints) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(round_floats(v, precision, convert_ints) for v in x)
    if isinstance(x, list):
        return [round_floats(v, precision, convert_ints) for v in x]
    return x


def stringify_params(*args, **kwargs):
    args_stringified = tuple(json.dumps(arg, sort_keys=True) for arg in args)
    kwargs_stringified = {
        key: json.dumps(value, sort_keys=True) for key, value in kwargs.items()
    }
    return (args_stringified, tuple(sorted(kwargs_stringified.items())))


def json_serializable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def make_json_serializable(value):
    if isinstance(value, dict):
        return {k: make_json_serializable(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [make_json_serializable(v) for v in value]
    elif isinstance(value, tuple):
        return tuple(make_json_serializable(v) for v in value)
    elif not json_serializable(value):
        return str(value)
    return value


def hash_params(*args, **kwargs):
    # Copy the arguments so we don't modify them
    args = deepcopy(args)
    kwargs = deepcopy(kwargs)

    # Make all values JSON serializable
    args = tuple(make_json_serializable(arg) for arg in args)
    kwargs = {key: make_json_serializable(value) for key, value in kwargs.items()}

    # Stringify the arguments
    str_args, str_kwargs = stringify_params(*args, **kwargs)
    return hashlib.md5(str(str_args).encode() + str(str_kwargs).encode()).hexdigest()[
        0:8
    ]


def write_jsonl(path: str, data: List[dict], append: bool = False):
    with jsonlines.open(path, mode="a" if append else "w") as writer:
        for item in data:
            writer.write(item)


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        jsonl_content = f.read()
    return [json.loads(jline) for jline in jsonl_content.splitlines()]


def write_jsonl_from_str(path: str, data: List[str], append: bool = False):
    with open(path, "a" if append else "w") as file:
        for item in data:
            file.write(item + "\n")


async def write_jsonl_async(path: str, data: List[dict], append: bool = True):
    mode = "a" if append else "w"
    async with aiofiles.open(path, mode=mode, encoding="utf-8") as file:
        for item in data:
            json_line = json.dumps(item) + "\n"
            await file.write(json_line)


async def write_jsonl_async_from_str(path: str, data: List[str], append: bool = False):
    mode = "a" if append else "w"
    async with aiofiles.open(path, mode=mode, encoding="utf-8") as file:
        for item in data:
            await file.write(item + "\n")


def shallow_dict(model: BaseModel) -> dict:
    return {
        field_name: (
            getattr(model, field_name)
            if isinstance(getattr(model, field_name), BaseModel)
            else value
        )
        for field_name, value in model
    }


def load_questions(path: str) -> list[ForecastingQuestion]:
    with open(path, "r") as f:
        jsonl_content = f.read()
    return [
        ForecastingQuestion(**json.loads(jline)) for jline in jsonl_content.splitlines()
    ]


def write_questions(questions: list[ForecastingQuestion], path: str):
    with open(path, "w") as f:
        for q in questions:
            f.write(f"{q.model_dump_json()}\n")


def append_question(question: ForecastingQuestion, path: str):
    with open(path, "a") as f:
        f.write(f"{question.model_dump_json()}\n")


def update_recursive(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            update_recursive(source[key], value)
        else:
            source[key] = value
    return source


def normalize_date_format(date: str) -> Optional[datetime]:
    for fmt in (
        "%Y-%m-%d %H:%M:%S",  # 2029-12-31 00:00:00
        "%Y-%m-%dT%H:%M:%S",  # 2029-12-31T00:00:00
        "%Y-%m-%d",  # 2029-12-31
        "%Y-%m-%dT%H:%M:%SZ",  # 2029-12-31T00:00:00Z
        "%d/%m/%Y",  # 31/12/2029
    ):
        try:
            return datetime.strptime(date, fmt)
        except ValueError:
            pass

    print(
        f"\033[1mWARNING:\033[0m Date format invalid and cannot be normalized: {date=}"
    )
    return None


def compare_dicts(dict1, dict2, path=""):
    differences = []
    for key in set(dict1.keys()) | set(dict2.keys()):
        current_path = f"{path}.{key}" if path else key
        if key not in dict1:
            differences.append(f"Key '{current_path}' missing in first dict")
        elif key not in dict2:
            differences.append(f"Key '{current_path}' missing in second dict")
        elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            differences.extend(compare_dicts(dict1[key], dict2[key], current_path))
        elif dict1[key] != dict2[key]:
            if isinstance(dict1[key], str) and isinstance(dict2[key], str):
                try:
                    date1 = normalize_date_format(dict1[key])
                    date2 = normalize_date_format(dict2[key])
                    if date1 == date2:
                        continue
                except ValueError:
                    pass
            differences.append(f"Mismatch for key '{current_path}':")
            differences.append(f"  First dict:  {dict1[key]}")
            differences.append(f"  Second dict: {dict2[key]}")
    return differences


def recombine_filename(filename: Path, suffix: str) -> Path:
    # Remove the current suffix (if any) and add the new one
    current_suffix = filename.suffix
    return filename.with_name(f"{filename.stem}{suffix}").with_suffix(current_suffix)


def shorten_model_name(model_name: str) -> str:
    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name


def delist(item):
    if isinstance(item, list):
        return item[0]
    return item
