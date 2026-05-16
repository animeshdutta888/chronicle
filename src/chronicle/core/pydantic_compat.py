from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field
from typing import Any, get_args, get_origin, get_type_hints


try:
    from pydantic import BaseModel, ConfigDict, Field
except ModuleNotFoundError:
    class BaseModel:
        def __init_subclass__(cls, **kwargs: Any) -> None:
            super().__init_subclass__(**kwargs)
            annotations = get_type_hints(cls)
            defaults = {key: getattr(cls, key, MISSING) for key in annotations}
            namespace: dict[str, Any] = {"__annotations__": dict(annotations)}
            for key, annotation in annotations.items():
                default = defaults[key]
                if isinstance(default, _FieldSpec):
                    if default.default_factory is not MISSING:
                        namespace[key] = field(default_factory=default.default_factory)
                    elif default.default is not MISSING:
                        namespace[key] = default.default
                elif default is not MISSING:
                    namespace[key] = default
            for key, value in namespace.items():
                setattr(cls, key, value)
            dataclass(cls)

        def model_dump(self, mode: str | None = None) -> dict[str, Any]:
            return asdict(self)

        @classmethod
        def model_validate(cls, data: dict[str, Any]):
            kwargs: dict[str, Any] = {}
            for key, annotation in get_type_hints(cls).items():
                if key not in data:
                    continue
                kwargs[key] = _coerce_value(annotation, data[key])
            return cls(**kwargs)

    class ConfigDict(dict):
        pass

    class _FieldSpec:
        def __init__(self, default: Any = MISSING, default_factory: Any = MISSING) -> None:
            self.default = default
            self.default_factory = default_factory

    def Field(default: Any = MISSING, default_factory: Any = MISSING):
        return _FieldSpec(default=default, default_factory=default_factory)


def _coerce_value(annotation: Any, value: Any) -> Any:
    origin = get_origin(annotation)
    if origin is list:
        inner = get_args(annotation)[0]
        return [_coerce_value(inner, item) for item in value]
    if origin is dict:
        key_type, value_type = get_args(annotation)
        return {
            _coerce_value(key_type, key): _coerce_value(value_type, nested)
            for key, nested in value.items()
        }
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel) and isinstance(value, dict):
            return annotation.model_validate(value)
        return value
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    for arg in args:
        if isinstance(arg, type) and issubclass(arg, BaseModel) and isinstance(value, dict):
            return arg.model_validate(value)
    return value
