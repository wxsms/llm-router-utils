"""Lightweight utils extracted from sglang.utils.

Only contains symbols referenced by the retained code paths.
"""
from __future__ import annotations

import importlib
import json
import sys
import traceback
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Tuple, Type, Union

from pydantic import BaseModel

from llm_router_utils.sglang.srt.environ import envs


def convert_json_schema_to_str(json_schema: Union[dict, str, Type[BaseModel]]) -> str:
    """Convert a JSON schema to a string.
    Parameters
    ----------
    json_schema
        The JSON schema.
    Returns
    -------
    str
        The JSON schema converted to a string.
    Raises
    ------
    ValueError
        If the schema is not a dictionary, a string or a Pydantic class.
    """
    if isinstance(json_schema, dict):
        schema_str = json.dumps(json_schema)
    elif isinstance(json_schema, str):
        schema_str = json_schema
    elif issubclass(json_schema, BaseModel):
        schema_str = json.dumps(json_schema.model_json_schema())
    else:
        raise ValueError(
            f"Cannot parse schema {json_schema}. The schema must be either "
            + "a Pydantic class, a dictionary or a string that contains the JSON "
            + "schema specification"
        )
    return schema_str


def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


def is_in_ci() -> bool:
    return envs.SGLANG_IS_IN_CI.get()


class LazyImport:
    """Lazy import to make `import sglang` run faster."""

    def __init__(self, module_name: str, class_name: str):
        self.module_name = module_name
        self.class_name = class_name
        self._module = None

    def _load(self):
        if self._module is None:
            module = importlib.import_module(self.module_name)
            self._module = getattr(module, self.class_name)
        return self._module

    def __getattr__(self, name: str):
        module = self._load()
        return getattr(module, name)

    def __call__(self, *args, **kwargs):
        module = self._load()
        return module(*args, **kwargs)


class TypeBasedDispatcher:
    def __init__(self, mapping: List[Tuple[Type, Callable]]):
        # Use dictionary for fast exact type matching, using OrderedDict(mapping)
        # to maintains registration order
        self._mapping = OrderedDict(mapping)
        # MRO cache for inheritance-based matching
        self._mro_cache = {}
        self._fallback_fn = None

    def add_fallback_fn(self, fallback_fn: Callable):
        self._fallback_fn = fallback_fn

    def __iadd__(self, other: "TypeBasedDispatcher"):
        for ty, fn in other._mapping.items():
            if ty not in self._mapping:
                self._mapping[ty] = fn

        self._mro_cache.clear()
        return self

    def __call__(self, obj: Any):
        obj_type = type(obj)
        # 1. First try exact match(o(1))
        fn = self._mapping.get(obj_type)
        if fn is not None:
            return fn(obj)

        # 2. If exact match fails, check MRO cache
        cached_fn = self._mro_cache.get(obj_type)
        if cached_fn is not None:
            return cached_fn(obj)

        # 3.search in registration order for compatible type(maintains origin behavior)
        for ty, fn in self._mapping.items():
            if isinstance(obj, ty):
                self._mro_cache[obj_type] = fn
                return fn(obj)

        # 4. if no matching type found, cache this result
        self._mro_cache[obj_type] = None

        if self._fallback_fn is not None:
            return self._fallback_fn(obj)
        raise ValueError(f"Invalid object: {obj}")
