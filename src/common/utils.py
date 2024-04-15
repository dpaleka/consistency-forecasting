import json
import aiofiles
from typing import List
import jsonlines
from copy import deepcopy
import hashlib
from pydantic import BaseModel

def format_float(x) -> str:
    if isinstance(x, float) or isinstance(x, int):
        return "{:.3f}".format(x)
    if isinstance(x, dict):
        return {k: format_float(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(format_float(v) for v in x)
    if isinstance(x, list):
        return [format_float(v) for v in x]
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
    with jsonlines.open(path, mode='a' if append else 'w') as writer:
        for item in data:
            writer.write(item)

def write_jsonl_from_str(path: str, data: List[str], append: bool = False):
    with open(path, 'a' if append else 'w') as file:
        for item in data:
            file.write(item + "\n")


async def write_jsonl_async(path: str, data: List[dict], append: bool = True):
    mode = 'a' if append else 'w'
    async with aiofiles.open(path, mode=mode) as file:
        for item in data:
            json_line = json.dumps(item) + "\n"
            await file.write(json_line)
            
async def write_jsonl_async_from_str(path: str, data: List[str], append: bool = False):
    mode = 'a' if append else 'w'
    async with aiofiles.open(path, mode=mode) as file:
        for item in data:
            await file.write(item + "\n")

def shallow_dict(model: BaseModel) -> dict:
    return {
        field_name: (getattr(model, field_name) if isinstance(getattr(model, field_name), BaseModel) else value)
        for field_name, value in model
    }