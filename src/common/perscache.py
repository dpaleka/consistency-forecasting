# Adapted from https://github.com/dpaleka/perscache, which is adapted from https://github.com/leshchenko1979/perscache
"""
MIT License

Copyright (c) 2022 Alexey Leshchenko
Copyright (c) 2024 Daniel Paleka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import datetime as dt
import functools
import hashlib
import inspect
import io
import json
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import cloudpickle
import redis
from beartype import beartype
from beartype.typing import Any, Callable, Iterable, Iterator, Optional, Type, Union
from icontract import require
from pydantic import BaseModel

from .datatypes import (
    ForecastingQuestion,
    ForecastingQuestion_stripped,
    ForecastingQuestions,
    PlainText,
    Prob,
    Prob_cot,
    ValidationResult,
    VerificationResult,
    BodyAndDate,
    ResolutionDate,
    SyntheticRelQuestion,
    SyntheticTagQuestion,
    QuestionGenerationResponse,
    QuestionGenerationResponse3,
)

perscache_supported_models = {
    "PlainText": PlainText,
    "Prob": Prob,
    "Prob_cot": Prob_cot,
    "ForecastingQuestion_stripped": ForecastingQuestion_stripped,
    "ForecastingQuestion": ForecastingQuestion,
    "ForecastingQuestions": ForecastingQuestions,
    "ValidationResult": ValidationResult,
    "VerificationResult": VerificationResult,
    "BodyAndDate": BodyAndDate,
    "ResolutionDate": ResolutionDate,
    "SyntheticRelQuestion": SyntheticRelQuestion,
    "SyntheticTagQuestion": SyntheticTagQuestion,
    "QuestionGenerationResponse": QuestionGenerationResponse,
    "QuestionGenerationResponse3": QuestionGenerationResponse3,
}

# Note: we cannot cache dynamically created BaseModels as in MiniInstantiator.py.
# Use NO_CACHE if you're instantiating tuples directly using instructor.


# Logger stubs
def debug(msg, *args):
    pass


def trace(fn):
    return fn


# Serializers
class Serializer(ABC):
    extension: str = None

    def __repr__(self):
        return f"<{self.__class__.__name__}(extension='{self.extension}')>"

    @abstractmethod
    def dumps(self, data: Any) -> bytes:
        ...

    @abstractmethod
    def loads(self, data: bytes) -> Any:
        ...


@beartype
def make_serializer(
    class_name: str,
    ext: str,
    dumps_fn: Callable[[Any], bytes],
    loads_fn: Callable[[bytes], Any],
) -> Type[Serializer]:
    """Create a serializer class.

    Args:
        class_name (str): The name of the serializer class.
        extension (str): The file extension of the serialized data.
        dumps (callable): The function to serialize data.
                Takes a single argument and returns a bytes object.
        loads (callable): The function to deserialize data.
                Takes a single bytes object as argument and returns an object.
    """
    return type(
        class_name,
        (Serializer,),
        {
            "extension": ext,
            "dumps": lambda _, data: dumps_fn(data),
            "loads": lambda _, data: loads_fn(data),
        },
    )


CloudPickleSerializer = make_serializer(
    "CloudPickleSerializer", "pickle", cloudpickle.dumps, cloudpickle.loads
)
JSONSerializer = make_serializer(
    "JSONSerializer",
    "json",
    lambda data: json.dumps(data).encode("utf-8"),
    lambda data: json.loads(data.decode("utf-8")),
)


class ResponseModelNotRegisteredError(NotImplementedError):
    def __init__(self, model_name):
        self.model_name = model_name
        super().__init__(
            f"Response model not registered in {__file__}: {model_name}.\n"
            "Note that caching dynamically created BaseModels, as in MiniInstantiator.py, is currently not supported.\n"
            "Set NO_CACHE=True to use instructor LLM calls for dynamically created BaseModels."
        )
        raise self


def pydantic_response_dumps(data: Any) -> bytes:
    if (
        isinstance(data, dict)
        and "value" in data
        and isinstance(data["value"], BaseModel)
    ):
        # Serialize the 'value' field which is a Pydantic model
        model_data = {
            "__class__": data["value"].__class__.__name__,
            "data": data["value"].model_dump(mode="json"),
        }
        # Replace the original 'value' with its serialized form
        data = {**data, "value": model_data}

    if (
        isinstance(data, dict)
        and "kwargs" in data
        and "response_model" in data["kwargs"]
    ):
        # Serialize the 'response_model' which is a class type
        # Assumption: it ends with something in known_models
        if data["kwargs"]["response_model"].__name__ in perscache_supported_models:
            data["kwargs"]["response_model"] = data["kwargs"]["response_model"].__name__
        else:
            raise ResponseModelNotRegisteredError(data["kwargs"]["response_model"])

    elif (
        isinstance(data, dict)
        and "bound_args" in data
        and "kwargs" in data["bound_args"]
        and "response_model" in data["bound_args"]["kwargs"]
    ):
        # Serialize the 'response_model' which is a class type
        # Assumption: it ends with something in known_models
        if (
            data["bound_args"]["kwargs"]["response_model"].__name__
            in perscache_supported_models
        ):
            data["bound_args"]["kwargs"]["response_model"] = data["bound_args"][
                "kwargs"
            ]["response_model"].__name__
        else:
            raise ResponseModelNotRegisteredError(
                data["bound_args"]["kwargs"]["response_model"]
            )

    return json.dumps(data).encode("utf-8")


def pydantic_response_loads(
    data: bytes, known_models: dict[str, Type[BaseModel]]
) -> Any:
    data_dict = json.loads(data.decode("utf-8"))
    if (
        "value" in data_dict
        and isinstance(data_dict["value"], dict)
        and "__class__" in data_dict["value"]
    ):
        class_name = data_dict["value"]["__class__"]
        if class_name in known_models:
            # Deserialize the 'value' field using the appropriate Pydantic model
            model_class = known_models[class_name]
            data_dict["value"] = model_class.model_validate(data_dict["value"]["data"])
        else:
            raise ResponseModelNotRegisteredError(class_name)

    if (
        "kwargs" in data_dict
        and "response_model" in data_dict["kwargs"]
        and isinstance(data_dict["kwargs"]["response_model"], str)
    ):
        # Deserialize the 'response_model' which is a class type
        class_name = data_dict["kwargs"]["response_model"]

        if class_name in known_models:
            data_dict["kwargs"]["response_model"] = known_models[class_name]

    elif (
        "bound_args" in data_dict
        and "kwargs" in data_dict["bound_args"]
        and "response_model" in data_dict["bound_args"]["kwargs"]
    ):
        # Deserialize the 'response_model' which is a class type
        class_name = data_dict["bound_args"]["kwargs"]["response_model"]
        if class_name in known_models:
            data_dict["bound_args"]["kwargs"]["response_model"] = known_models[
                class_name
            ]

    return data_dict


JSONPydanticResponseSerializer = make_serializer(
    "JSONPydanticSerializer",
    "json",
    pydantic_response_dumps,
    lambda data: pydantic_response_loads(data, perscache_supported_models),
)

PickleSerializer = make_serializer(
    "PickleSerializer", "pickle", pickle.dumps, pickle.loads
)


class YAMLSerializer(Serializer):
    extension = "yaml"

    def dumps(self, data: Any) -> bytes:
        import yaml

        return yaml.dump(data).encode("utf-8")

    def loads(self, data: bytes) -> Any:
        import yaml

        return yaml.safe_load(data.decode("utf-8"))


class ParquetSerializer(Serializer):
    extension = "parquet"

    @beartype
    def __init__(self, compression: Optional[str] = "brotli"):
        self.compression = compression

    def __repr__(self):
        return f"<ParquetSerializer(extension='parquet', compression='{self.compression}')>"

    def dumps(self, data: Any) -> bytes:
        import pyarrow
        import pyarrow.parquet

        buf = pyarrow.BufferOutputStream()
        pyarrow.parquet.write_table(
            pyarrow.Table.from_pandas(data), buf, compression=self.compression
        )
        buf.flush()
        return buf.getvalue()

    def loads(self, data: bytes) -> Any:
        import pyarrow
        import pyarrow.parquet

        return pyarrow.parquet.read_table(pyarrow.BufferReader(data)).to_pandas()


class CSVSerializer(Serializer):
    extension = "csv"

    def dumps(self, data: Any) -> bytes:
        import pandas as pd

        return pd.DataFrame(data).to_csv().encode("utf-8")

    def loads(self, data: bytes) -> Any:
        import pandas as pd

        return pd.read_csv(io.StringIO(data.decode("utf-8")), index_col=0)


# Storage
class CacheExpired(Exception):
    ...


class Storage(ABC):
    @abstractmethod
    def read(self, path: Union[str, Path], deadline: dt.datetime) -> bytes:
        ...

    @abstractmethod
    def write(self, path: Union[str, Path], data: bytes) -> None:
        ...


class FileStorage(Storage):
    @beartype
    def __init__(
        self,
        location: Optional[Union[str, Path]] = ".cache",
        max_size: Optional[int] = None,
    ):
        self.location = Path(location)
        self.max_size = max_size

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(location={self.location}, max_size={self.max_size})>"

    def read(self, path: Union[str, Path], deadline: dt.datetime) -> bytes:
        final_path = self.location / path

        if deadline is not None and self.mtime(final_path) < deadline:
            raise CacheExpired

        return self.read_file(self.location / path)

    def write(self, path: Union[str, Path], data: bytes) -> None:
        final_path = self.location / path

        self.ensure_path(final_path.parent)

        if self.max_size and self.size(self.location) + len(data) > self.max_size:
            self.delete_least_recently_used(target_size=self.max_size)

        self.write_file(final_path, data)

    def delete_least_recently_used(self, target_size: int) -> None:
        """Removes the least recently used file from the cache.
        The least recently used file is the one with the smallest last access time.

        Args:
            target_size: The target size of the cache.
        """
        files = sorted(self.iterdir(self.location), key=self.atime, reverse=True)

        # find the set of most recently accessed files whose total size
        # is smaller than the target size
        i, size = 0, 0
        while size < target_size and i < len(files):
            size += self.size(files[i])
            i += 1

        # remove remaining files
        for f in files[i - 1 :]:
            self.delete(f)

    def clear(self) -> None:
        """Remove the directory with the cache along with all of its contents
        if it exists, otherwise just silently passes with no exceptions.
        """

        for f in self.iterdir(self.location):
            self.delete(f)
        self.rmdir(self.location)

    @abstractmethod
    def read_file(self, path: Union[str, Path]) -> bytes:
        """Read a file at a relative path inside the cache
        or raise FileNotFoundError if not found.
        """
        ...

    @abstractmethod
    def write_file(self, path: Union[str, Path]) -> bytes:
        """Write a file at a relative path inside the cache
        or raise FileNotFoundError if the cache directory doesn't exist.
        """
        ...

    @abstractmethod
    def ensure_path(self, path: Union[str, Path]) -> None:
        """Create an absolute path if it doesn't exist."""
        ...

    @abstractmethod
    def iterdir(self, path: Union[str, Path]) -> Union[Iterator[Path], list]:
        """Return an iterator through files within a directory indicated by path
        or an empty list if the path doesn't exist.
        """
        ...

    @abstractmethod
    def rmdir(self, path: Union[str, Path]) -> None:
        """Remove a directory. Silently pass if it doesn't exist."""
        ...

    @abstractmethod
    def mtime(self, path: Union[str, Path]) -> dt.datetime:
        """Get file last modified time."""
        ...

    @abstractmethod
    def atime(self, path: Union[str, Path]) -> dt.datetime:
        """Get file last accessed time."""
        ...

    @abstractmethod
    def size(self, path: Union[str, Path]) -> int:
        """Get the size in bytes for a file or a directory indicated by path.
        Zero if the path doesn't exist.
        """
        ...

    @abstractmethod
    def delete(self, path: Union[str, Path]) -> None:
        """Remove file or raise FileNotFoundError if not found."""
        ...


class LocalFileStorage(FileStorage):
    def read_file(self, path: Union[str, Path]) -> bytes:
        return path.read_bytes()

    def write_file(self, path: Union[str, Path], data: bytes) -> None:
        path.write_bytes(data)

    def ensure_path(self, path: Union[str, Path]) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def iterdir(self, path: Union[str, Path]) -> Union[Iterator[Path], list]:
        return path.iterdir() if path.exists() else []

    def rmdir(self, path: Union[str, Path]) -> None:
        if path.exists():
            path.rmdir()

    def mtime(self, path: Union[str, Path]) -> dt.datetime:
        return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)

    def atime(self, path: Union[str, Path]) -> dt.datetime:
        return dt.datetime.fromtimestamp(path.stat().st_atime, tz=dt.timezone.utc)

    def size(self, path: Union[str, Path]) -> int:
        return path.stat().st_size if path.exists() else 0

    def delete(self, path: Union[str, Path]) -> None:
        path.unlink()


REDIS_CONFIG_DEFAULT = {
    "host": "localhost",
    "port": 6379,
    "db": 7,
}

if os.getenv("REDIS_PASSWORD"):
    REDIS_CONFIG_DEFAULT["password"] = os.getenv("REDIS_PASSWORD")


class DbStorage(Storage):
    """A generic database storage class."""

    @abstractmethod
    def read(self, path: Union[str, Path], deadline: dt.datetime) -> bytes:
        """Read data from the database."""

    @abstractmethod
    def write(self, path: Union[str, Path], data: bytes) -> None:
        """Write data to the database."""


class RedisStorage(Storage):
    """A Redis-specific storage class optimized for storing arbitrary JSON objects."""

    @beartype
    def __init__(
        self,
        namespace: Optional[Union[str, Path]] = "cache",
        redis_config: Optional[dict] = None,
    ):
        """
        Initialize the RedisStorage instance.

        :param location: A namespace prefix for keys in Redis to avoid collisions.
        :param redis_config: Configuration dictionary for Redis connection.
        """
        # Convert location to string to ensure compatibility with Redis key naming
        self.namespace = str(namespace)
        assert ":" not in self.namespace, "Namespace cannot contain ':'"
        self.redis_config = redis_config if redis_config else REDIS_CONFIG_DEFAULT
        self.db = self._connect()

    def _connect(self):
        """Establish a connection to the Redis server."""
        return redis.Redis(**self.redis_config)

    def read(self, path: str | Path, deadline: dt.datetime) -> bytes:
        """
                Read a JSON object from Redis.

                :param path: The key associated with the JSON object.
        #        :param deadline: Not used in this implementation, provided for interface compatibility.
                :return: The JSON object as bytes.
                :raises FileNotFoundError: If the key does not exist in Redis.
        """
        key = self._make_key(path)
        data = self.db.get(key)
        if data is None:
            raise FileNotFoundError(f"No data found for key: {key}")

        return data

    def write(self, path: Union[str, Path], data: bytes) -> None:
        """
        Write a JSON object to Redis.

        :param path: The key under which to store the JSON object.
        :param data: The JSON object, serialized into bytes.
        """
        key = self._make_key(path)
        self.db.set(key, data)

    def _make_key(self, path: Union[str, Path]) -> str:
        """
        Generate a Redis key based on the given path and the location prefix.

        :param path: The original key for the JSON object.
        :return: A namespaced key for use in Redis.
        """
        return f"{self.namespace}:{str(path)}"

    def __decode_key(self, key: str) -> str:
        """
        Decode a Redis key into its original form.

        :param key: The Redis key.
        :return: The original key for the JSON object.
        """
        return key.split(":", 1)[1]

    def get_all(self, namespace: str):
        """
        Get all keys and values from a given namespace.

        :param namespace: The namespace to query.
        :return: A dictionary of keys and values.
        """
        keys = self.db.keys(f"{namespace}:*")
        print("len(keys):", len(keys))
        return {
            self.__decode_key(key.decode("utf-8")): self.db.get(key) for key in keys
        }


# Value Wrappers
class ValueWrapper(ABC):
    @abstractmethod
    def wrap(
        self, args: tuple, kwargs: dict, value: Any, fn: Optional[Callable]
    ) -> Any:
        ...

    @abstractmethod
    def unwrap(self, value: Any, fn: Optional[Callable]) -> Any:
        ...

    """
    The hashed key doesn't depend on the value wrapper, nor on the serializer or storage, for that matter.
    Hence  unwrappers should all work in the same way, to be able to read the cached requests using other value wrappers.
    """


class ValueWrapperId(ValueWrapper):
    def __init__(self):
        pass

    def wrap(
        self, args: tuple, kwargs: dict, value: Any, fn: Optional[Callable] = None
    ) -> Any:
        return {"value": value}

    def unwrap(self, wrapped_value: Any, fn: Optional[Callable] = None) -> Any:
        return wrapped_value["value"]


class ValueWrapperDict(ValueWrapper):
    def __init__(self):
        pass

    def wrap(
        self, args: tuple, kwargs: dict, value: Any, fn: Optional[Callable] = None
    ) -> Any:
        return {"args": args, "kwargs": kwargs, "value": value}

    def unwrap(self, wrapped_value: Any, fn: Optional[Callable] = None) -> Any:
        return wrapped_value["value"]


class ValueWrapperDictInspectArgs(ValueWrapper):
    """
    Names the args to be in the same format as kwargs.
    """

    def __init__(self):
        pass

    def wrap(self, args: tuple, kwargs: dict, value: Any, fn: Callable) -> Any:
        arg_dict = inspect.signature(fn).bind(*args, **kwargs).arguments
        return {"bound_args": arg_dict, "value": value}

    def unwrap(self, wrapped_value: Any, fn: Optional[Callable] = None) -> Any:
        return wrapped_value["value"]


# Cache
def hash_it(*data) -> str:
    result = hashlib.md5()  # nosec B303
    for datum in data:
        result.update(cloudpickle.dumps(datum))
    return result.hexdigest()


def is_async(fn):
    return inspect.iscoroutinefunction(fn) and not inspect.isgeneratorfunction(fn)


class Cache:
    @beartype
    def __init__(
        self,
        serializer: Serializer = None,
        storage: Storage = None,
        value_wrapper: ValueWrapper = None,
    ):
        if os.getenv("NO_CACHE"):
            print("\n\033[1m NO_CACHE \033[0m\n")
            pass
        self.serializer = serializer or CloudPickleSerializer()
        self.storage = storage or LocalFileStorage()
        self.value_wrapper = value_wrapper or ValueWrapperId()

    def __repr__(self) -> str:
        return f"<Cache(serializer={self.serializer}, storage={self.storage})>"

    @beartype
    @require(
        lambda ttl: ttl is None or ttl > dt.timedelta(seconds=0),
        "ttl must be positive.",
    )
    def __call__(
        self,
        fn: Optional[Callable] = None,
        *,
        ignore: Optional[Iterable[str]] = None,
        serializer: Optional[Serializer] = None,
        storage: Optional[Storage] = None,
        ttl: Optional[dt.timedelta] = None,
        value_wrapper: Optional[ValueWrapper] = None,
    ):
        if os.getenv("NO_CACHE"):
            return fn
        if isinstance(ignore, str):
            ignore = [ignore]
        wrapper = _CachedFunction(
            self,
            ignore,
            serializer or self.serializer,
            storage or self.storage,
            ttl,
            value_wrapper or self.value_wrapper,
        )
        return wrapper if fn is None else wrapper(fn)

    cache = __call__

    @staticmethod
    @trace
    def _get(
        key: str, serializer: Serializer, storage: Storage, deadline: dt.datetime
    ) -> Any:
        data = storage.read(key, deadline)
        return serializer.loads(data)

    @staticmethod
    @trace
    def _set(key: str, value: Any, serializer: Serializer, storage: Storage) -> None:
        data = serializer.dumps(value)
        storage.write(key, data)

    @staticmethod
    def _get_hash(
        fn: Callable,
        args: tuple,
        kwargs: dict,
        serializer: Serializer,
        ignore: Iterable[str],
    ) -> str:
        arg_dict = inspect.signature(fn).bind(*args, **kwargs).arguments
        if ignore is not None:
            arg_dict = {k: v for k, v in arg_dict.items() if k not in ignore}
        return hash_it(inspect.getsource(fn), type(serializer).__name__, arg_dict)

    def _get_filename(self, fn: Callable, key: str, serializer: Serializer) -> str:
        return f"{fn.__name__}-{key}.{serializer.extension}"


class _CachedFunction:
    @beartype
    def __init__(
        self,
        cache: Cache,
        ignore: Optional[Iterable[str]],
        serializer: Serializer,
        storage: Storage,
        ttl: Optional[dt.timedelta],
        value_wrapper: Optional[ValueWrapper],
    ):
        self.cache = cache
        self.ignore = ignore
        self.serializer = serializer
        self.storage = storage
        self.ttl = ttl
        self.value_wrapper = value_wrapper or ValueWrapperId()

    @require(
        lambda self, fn: self.ignore is None
        or all(x in inspect.signature(fn).parameters for x in self.ignore),
        "Ignored parameters not found in the function signature.",
    )
    def __call__(self, fn: Callable) -> Callable:
        wrapper = self._async_wrapper if is_async(fn) else self._non_async_wrapper
        return functools.update_wrapper(functools.partial(wrapper, fn), fn)

    def _non_async_wrapper(self, fn: Callable, *args, **kwargs):
        debug("Getting cached result for function %s", fn.__name__)
        key = self.cache._get_hash(fn, args, kwargs, self.serializer, self.ignore)
        key = self.cache._get_filename(fn, key, self.serializer)
        try:
            if os.getenv("NO_READ_CACHE"):
                raise FileNotFoundError
            wrapped_value = self.cache._get(
                key, self.serializer, self.storage, self.deadline
            )
            return self.value_wrapper.unwrap(wrapped_value, fn)
        except (FileNotFoundError, CacheExpired) as exception:
            debug("Unable to get cached result for %s: %s", fn.__name__, exception)
            value = fn(*args, **kwargs)
            if os.getenv("NO_WRITE_CACHE"):
                return value
            wrapped_value = self.value_wrapper.wrap(args, kwargs, value, fn)
            self.cache._set(key, wrapped_value, self.serializer, self.storage)
            return value

    async def _async_wrapper(self, fn: Callable, *args, **kwargs):
        debug("Getting cached result for function %s", fn.__name__)
        key = self.cache._get_hash(fn, args, kwargs, self.serializer, self.ignore)
        key = self.cache._get_filename(fn, key, self.serializer)
        try:
            if os.getenv("NO_READ_CACHE"):
                raise FileNotFoundError
            wrapped_value = self.cache._get(
                key, self.serializer, self.storage, self.deadline
            )
            return self.value_wrapper.unwrap(wrapped_value, fn)
        except (FileNotFoundError, CacheExpired) as exception:
            debug("Unable to get cached result for %s: %s", fn.__name__, exception)
            value = await fn(*args, **kwargs)
            if os.getenv("NO_WRITE_CACHE"):
                return value
            wrapped_value = self.value_wrapper.wrap(args, kwargs, value, fn)
            self.cache._set(key, wrapped_value, self.serializer, self.storage)
            return value

    @property
    def deadline(self) -> dt.datetime:
        return dt.datetime.now(dt.timezone.utc) - self.ttl if self.ttl else None
