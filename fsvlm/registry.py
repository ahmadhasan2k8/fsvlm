"""Registry pattern for extensible component lookup.

New readers and report generators can be added without touching existing code:
just implement the ABC and register with the decorator.
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


class Registry:
    """Generic registry for named implementations.

    Usage:
        readers = Registry()

        @readers.register("folder")
        class FolderReader(LabelReader): ...

        reader_cls = readers.get("folder")
    """

    def __init__(self) -> None:
        self._items: dict[str, type] = {}

    def register(self, name: str):  # type: ignore[type-arg]
        """Decorator to register a class under a name."""

        def decorator(cls: type) -> type:
            self._items[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type:
        """Look up a registered class by name.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in self._items:
            available = list(self._items.keys())
            raise KeyError(f"Unknown: {name!r}. Available: {available}")
        return self._items[name]

    def list_available(self) -> list[str]:
        """Return all registered names."""
        return list(self._items.keys())

    def all_classes(self) -> list[type]:
        """Return all registered classes."""
        return list(self._items.values())


# Global registries — populated by reader/report module imports
label_readers: Registry = Registry()
report_generators: Registry = Registry()
