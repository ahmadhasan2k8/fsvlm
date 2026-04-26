"""Tests for fsvlm.events."""

from __future__ import annotations

from fsvlm.events import (
    DataPrepCompleteEvent,
    EventBus,
    SweepProgressEvent,
    TrainingCompleteEvent,
    TrainingProgressEvent,
    ValidationCompleteEvent,
)


def test_event_bus_subscribe_and_emit():
    bus = EventBus()
    received = []

    bus.subscribe(TrainingProgressEvent, received.append)
    event = TrainingProgressEvent(
        epoch=1,
        total_epochs=3,
        step=10,
        total_steps=100,
        loss=0.5,
        learning_rate=2e-4,
        elapsed_seconds=30.0,
    )
    bus.emit(event)

    assert len(received) == 1
    assert received[0].epoch == 1
    assert received[0].loss == 0.5


def test_event_bus_multiple_handlers():
    bus = EventBus()
    results_a: list[TrainingProgressEvent] = []
    results_b: list[TrainingProgressEvent] = []

    bus.subscribe(TrainingProgressEvent, results_a.append)
    bus.subscribe(TrainingProgressEvent, results_b.append)

    event = TrainingProgressEvent(
        epoch=2,
        total_epochs=5,
        step=20,
        total_steps=100,
        loss=0.3,
        learning_rate=1e-4,
        elapsed_seconds=60.0,
    )
    bus.emit(event)

    assert len(results_a) == 1
    assert len(results_b) == 1


def test_event_bus_different_types():
    bus = EventBus()
    training_events: list = []
    validation_events: list = []

    bus.subscribe(TrainingProgressEvent, training_events.append)
    bus.subscribe(ValidationCompleteEvent, validation_events.append)

    bus.emit(
        TrainingProgressEvent(
            epoch=1,
            total_epochs=3,
            step=5,
            total_steps=50,
            loss=1.0,
            learning_rate=2e-4,
            elapsed_seconds=10.0,
        )
    )
    bus.emit(
        ValidationCompleteEvent(
            num_samples=100,
            f1=0.92,
            auroc=0.95,
            accuracy=0.93,
        )
    )

    assert len(training_events) == 1
    assert len(validation_events) == 1


def test_event_bus_unsubscribe():
    bus = EventBus()
    received = []
    handler = received.append

    bus.subscribe(TrainingProgressEvent, handler)
    bus.unsubscribe(TrainingProgressEvent, handler)

    bus.emit(
        TrainingProgressEvent(
            epoch=1,
            total_epochs=1,
            step=1,
            total_steps=1,
            loss=0.1,
            learning_rate=1e-4,
            elapsed_seconds=5.0,
        )
    )

    assert len(received) == 0


def test_event_bus_clear():
    bus = EventBus()
    received = []
    bus.subscribe(TrainingProgressEvent, received.append)
    bus.clear()

    bus.emit(
        TrainingProgressEvent(
            epoch=1,
            total_epochs=1,
            step=1,
            total_steps=1,
            loss=0.1,
            learning_rate=1e-4,
            elapsed_seconds=5.0,
        )
    )

    assert len(received) == 0


def test_event_bus_no_handler():
    """Emitting with no subscribers should not raise."""
    bus = EventBus()
    bus.emit(
        TrainingProgressEvent(
            epoch=1,
            total_epochs=1,
            step=1,
            total_steps=1,
            loss=0.1,
            learning_rate=1e-4,
            elapsed_seconds=5.0,
        )
    )


def test_training_complete_event():
    from pathlib import Path

    e = TrainingCompleteEvent(
        adapter_path=Path("/tmp/adapter"),
        elapsed_seconds=120.0,
        final_loss=0.05,
    )
    assert e.elapsed_seconds == 120.0


def test_data_prep_complete_event():
    e = DataPrepCompleteEvent(
        total_images=100,
        good_count=80,
        defect_count=20,
        train_count=80,
        val_count=20,
    )
    assert e.total_images == 100


def test_sweep_progress_event():
    e = SweepProgressEvent(
        candidate_index=1,
        total_candidates=3,
        current_config={"rank": 32},
        best_f1_so_far=0.85,
    )
    assert e.candidate_index == 1
