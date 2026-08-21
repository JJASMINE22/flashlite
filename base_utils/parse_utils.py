#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/03/09 11:42
# @Author  : XinYi Huang
# @FileName: parse_utils.py
# @Software: PyCharm
import dataclasses
from argparse import ArgumentParser, ArgumentTypeError
from typing import Iterable, Union, Any, Dict, Optional, Callable, get_type_hints


def create_task_enum(root_dir):
    from pathlib import Path

    root_path = Path(root_dir).resolve()
    if not root_path.is_dir():
        raise ValueError(f"root dir {root_dir} not exist")

    subdirs = {d.name: d for d in root_path.iterdir() if d.is_dir()}
    enum_items = {
        name: str(d.resolve()) for name, d in subdirs.items()
    }

    from enum import Enum
    task_enum = Enum("YamadaEnum", enum_items, module=__name__)

    return task_enum


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def make_choice_type_function(choices: list) -> Callable[[str], Any]:
    """
    Creates a mapping function from each choices string representation to the actual value. Used to support multiple
    value types for a single argument.

    Args:
        choices (list): List of choices.

    Returns:
        Callable[[str], Any]: Mapping function from string representation to actual value for each choice.
    """
    str_to_choice = {str(choice): choice for choice in choices}
    return lambda arg: str_to_choice.get(arg, arg)


class CustomArgumentParser(ArgumentParser):
    def __init__(
            self,
            dataclass_types: Union[Iterable[Any], Any]
    ):
        super().__init__()
        if not isinstance(dataclass_types, Iterable):
            dataclass_types = [dataclass_types]
        self.dataclass_types = dataclass_types
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    def _parse_dataclass_field(self, field):
        long_options = [f"--{field.name}"]
        if "_" in field.name:
            long_options.append(f"--{field.name.replace('_', '-')}")

        kwargs = {}
        if isinstance(field.type, str):
            raise RuntimeError(
                "Unresolved type detected, which should have been done with the help of "
                "`typing.get_type_hints` method by default"
            )

        if field.type is bool or field.type == Optional[bool]:
            kwargs["type"] = string_to_bool
            if field.type is bool or (field.default is not None and field.default is not dataclasses.MISSING):
                field.default = False if field.default is dataclasses.MISSING else field.default
                kwargs["default"] = field.default
                kwargs["nargs"] = "?"
                kwargs["const"] = True
        else:
            kwargs["type"] = field.type
            if field.default is not dataclasses.MISSING:
                kwargs["default"] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                kwargs["default"] = field.default_factory()
            else:
                kwargs["required"] = True

        self.add_argument(*long_options, **kwargs)

    def _add_dataclass_arguments(self, dtype: Any):

        try:
            type_hints: Dict[str, type] = get_type_hints(dtype)
        except Exception as e:
            raise e

        for field in dataclasses.fields(dtype):
            if not field.init:
                continue
            field.type = type_hints[field.name]
            self._parse_dataclass_field(field)

    def parse_args_into_dataclasses(self):
        namespace, _ = self.parse_known_args()

        outputs = []

        for dtype in self.dataclass_types:
            keys = [field.name for field in dataclasses.fields(dtype) if field.init]
            dtype_args = {k: v for k, v in vars(namespace).items() if k in keys}

            for k in keys:
                delattr(namespace, k)

            outputs.append(dtype(**dtype_args))

        return *outputs,


if __name__ == '__main__':

    tasks = create_task_enum(r"D:\datasets\yamadarec")
    for task in tasks:
        print(task.name, task.value)