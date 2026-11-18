"""Minimal TOML writer for the ifopt config layout (stdlib ``tomllib`` only reads TOML).

Supports nested tables, arrays of tables, strings, bools, ints, floats (shortest round-trip repr,
inf/nan) and arrays of those.  Output of ``dumps_toml`` read back with ``tomllib.loads`` gives the
same data.
"""
from __future__ import annotations

import json
import math
import re

_BARE = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(k: str) -> str:
    return k if _BARE.match(k) else json.dumps(k)


def _value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if hasattr(v, "item"):                       # numpy scalar
        return _value(v.item())
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


def _is_table_array(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) > 0 and all(isinstance(x, dict) for x in v)


def _emit(table: dict, prefix: str, out: list[str]) -> None:
    for k, v in table.items():
        if not isinstance(v, dict) and not _is_table_array(v):
            out.append(f"{_key(k)} = {_value(v)}")
    for k, v in table.items():
        name = f"{prefix}.{_key(k)}" if prefix else _key(k)
        if isinstance(v, dict):
            out.append("")
            out.append(f"[{name}]")
            _emit(v, name, out)
        elif _is_table_array(v):
            for item in v:
                out.append("")
                out.append(f"[[{name}]]")
                _emit(item, name, out)


def dumps_toml(data: dict) -> str:
    out: list[str] = []
    _emit(data, "", out)
    return "\n".join(out).lstrip("\n") + "\n"
