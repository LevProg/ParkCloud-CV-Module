"""
Тесты модуля потока событий.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.event_stream import EventBus, EventType


class TestEventBus:

    def test_initial_state(self):
        bus = EventBus(space_ids=["A01", "A02", "B01"])
        assert bus.total_spaces == 3
        assert len(bus.occupied_spaces) == 0
        assert len(bus.free_spaces) == 3

    def test_update_to_occupied(self):
        bus = EventBus(
            space_ids=["A01"],
            hysteresis_seconds=0,  # отключено для тестирования
        )
        event = bus.update("A01", 80.0, 0.9)
        assert event is not None
        assert event.event_type == EventType.SPACE_OCCUPIED
        assert event.space_id == "A01"
        assert bus._is_occupied["A01"] is True

    def test_update_to_freed(self):
        bus = EventBus(
            space_ids=["A01"],
            hysteresis_seconds=0,
        )
        # Сначала занять
        bus.update("A01", 80.0, 0.9)
        assert bus._is_occupied["A01"] is True

        # Затем освободить
        event = bus.update("A01", 10.0, 0.3)
        assert event is not None
        assert event.event_type == EventType.SPACE_FREED

    def test_no_event_in_dead_zone(self):
        bus = EventBus(
            space_ids=["A01"],
            hysteresis_seconds=0,
        )
        # Обновление до 40% — между порогами, нет перехода
        event = bus.update("A01", 40.0, 0.5)
        assert event is None
        assert bus._is_occupied["A01"] is False

    def test_batch_update(self):
        bus = EventBus(
            space_ids=["A01", "A02"],
            hysteresis_seconds=0,
        )
        events = bus.update_batch([
            {"space_id": "A01", "occupancy_pct": 90.0, "confidence": 0.9},
            {"space_id": "A02", "occupancy_pct": 5.0, "confidence": 0.8},
        ])
        assert len(events) == 1  # Только A01 стало занятым
        assert events[0].space_id == "A01"

    def test_summary(self):
        bus = EventBus(
            space_ids=["A01", "A02", "B01"],
            hysteresis_seconds=0,
        )
        bus.update("A01", 90.0, 0.9)
        bus.update("B01", 70.0, 0.8)

        summary = bus.summary
        assert summary["total"] == 3
        assert summary["occupied"] == 2
        assert summary["free"] == 1

    def test_recent_events(self):
        bus = EventBus(
            space_ids=["A01"],
            hysteresis_seconds=0,
        )
        bus.update("A01", 90.0, 0.9)
        bus.update("A01", 10.0, 0.3)

        events = bus.get_recent_events()
        assert len(events) == 2

    def test_snapshot_event(self):
        bus = EventBus(space_ids=["A01", "A02"])
        snapshot = bus.get_snapshot_event()
        assert snapshot.event_type == EventType.FULL_SNAPSHOT
        assert "total" in snapshot.metadata
        assert "spaces" in snapshot.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
