"""Event system for agent communication.

Agents emit events that other agents or the UI can subscribe to.
This decouples agents from the presentation layer: the Training Agent
emits TrainingProgressEvent without knowing whether CLI or Gradio listens.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Event dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrainingProgressEvent:
    """Emitted by Training Agent at each logging step."""

    epoch: int
    total_epochs: int
    step: int
    total_steps: int
    loss: float
    learning_rate: float
    elapsed_seconds: float


@dataclass
class TrainingCompleteEvent:
    """Emitted when a single training run finishes."""

    adapter_path: Path
    elapsed_seconds: float
    final_loss: float


@dataclass
class ValidationCompleteEvent:
    """Emitted when validation finishes."""

    num_samples: int
    f1: float
    auroc: float
    accuracy: float


@dataclass
class DataPrepCompleteEvent:
    """Emitted when data preparation finishes."""

    total_images: int
    good_count: int
    defect_count: int
    train_count: int
    val_count: int


@dataclass
class SweepProgressEvent:
    """Emitted between sweep candidates."""

    candidate_index: int
    total_candidates: int
    current_config: dict[str, Any] = field(default_factory=dict)
    best_f1_so_far: float = 0.0


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class EventBus:
    """Simple synchronous publish-subscribe event bus.

    Usage::

        bus = EventBus()
        bus.subscribe(TrainingProgressEvent, my_handler)
        bus.emit(TrainingProgressEvent(epoch=1, ...))
    """

    def __init__(self) -> None:
        self._handlers: dict[type, list[Callable[..., None]]] = defaultdict(list)

    def subscribe(self, event_type: type, handler: Callable[..., None]) -> None:
        """Register a handler for an event type.

        Args:
            event_type: The dataclass type to listen for.
            handler: Callable that receives the event instance.
        """
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: type, handler: Callable[..., None]) -> None:
        """Remove a handler for an event type."""
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def emit(self, event: Any) -> None:
        """Dispatch an event to all subscribed handlers.

        Args:
            event: The event instance to dispatch.
        """
        for handler in self._handlers.get(type(event), []):
            handler(event)

    def clear(self) -> None:
        """Remove all handlers."""
        self._handlers.clear()
