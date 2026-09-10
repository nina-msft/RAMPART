# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Shared value-domain rules for dataclass trace adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    TypeAlias,
)

from pydantic import (
    BeforeValidator,
    PlainSerializer,
    WithJsonSchema,
)

if TYPE_CHECKING:
    from pydantic import ValidationError, ValidationInfo


def json_value(value: object) -> object:
    """Check and copy JSON values without lossy coercion.

    Returns:
        object: JSON primitives, lists, and string-keyed dictionaries.

    Raises:
        ValueError: If a value is non-finite, cyclic, or outside the JSON domain.
    """
    return _json_value(value=value, path="$", active=set())


def _json_value(*, value: object, path: str, active: set[int]) -> object:
    """Recursively validate the JSON domain, retaining the offending path.

    Returns:
        object: A JSON-safe copy.

    Raises:
        ValueError: If the value cannot be represented faithfully in JSON.
    """
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if not isinstance(value, Mapping | list):
        msg = f"{path}: {type(value).__name__} is outside the finite JSON domain"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] Pydantic wraps ValueError.
    if id(value) in active:
        msg = f"{path}: cyclic JSON value"
        raise ValueError(msg)
    active.add(id(value))
    try:
        if isinstance(value, list):
            return [
                _json_value(value=item, path=f"{path}[{index}]", active=active)
                for index, item in enumerate(value)
            ]
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                msg = f"{path}: JSON object keys must be strings"
                raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] Pydantic wraps ValueError.
            result[key] = _json_value(value=item, path=f"{path}.{key}", active=active)
        return result
    finally:
        active.remove(id(value))


# Standard dataclass construction stays permissive; adapters validate these maps.
JsonMapping: TypeAlias = Annotated[dict[str, Any], BeforeValidator(json_value)]


# Pydantic supplies value/info positionally to BeforeValidator callbacks.
def _iso_datetime(value: object, info: ValidationInfo) -> object:
    """Retain Python ISO datetime support, including naive and subminute offsets.

    Returns:
        object: Parsed datetime strings, or the unchanged value for validation.

    Raises:
        ValueError: If the string is not an ISO datetime.
    """
    if info.mode == "json" and isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:
            msg = "expected an ISO 8601 datetime string"
            raise ValueError(msg) from exc
    return value


IsoDatetime: TypeAlias = Annotated[
    datetime,
    BeforeValidator(_iso_datetime),
    PlainSerializer(datetime.isoformat),
    # Python ISO datetimes include values outside RFC 3339's date-time format.
    WithJsonSchema(
        {"type": "string", "description": "ISO 8601 datetime; UTC offset is optional."}
    ),
]


def validation_message(*, error: ValidationError, path: str) -> str:
    """Render Pydantic errors without including producer data in the message.

    Returns:
        str: Field paths and validation reasons.
    """
    messages: list[str] = []
    for detail in error.errors(include_url=False, include_input=False):
        location = path
        for part in detail["loc"]:
            location += f"[{part}]" if isinstance(part, int) else f".{part}"
        messages.append(f"{location}: {detail['msg']}")
    return "; ".join(messages)
