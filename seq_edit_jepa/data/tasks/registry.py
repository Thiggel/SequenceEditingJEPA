from __future__ import annotations

from typing import Any, Callable

from seq_edit_jepa.data.tasks.base import SequenceTask


_TASKS: dict[str, type[SequenceTask]] = {}


def register_task(name: str) -> Callable[[type[SequenceTask]], type[SequenceTask]]:
    def decorator(cls: type[SequenceTask]) -> type[SequenceTask]:
        _TASKS[name] = cls
        return cls

    return decorator


def build_task(config: dict[str, Any]) -> SequenceTask:
    name = str(config.get("name", "lano"))
    if name not in _TASKS:
        raise ValueError(f"Unknown task '{name}'. Available tasks: {sorted(_TASKS)}")
    return _TASKS[name](config)


def available_tasks() -> list[str]:
    return sorted(_TASKS)
