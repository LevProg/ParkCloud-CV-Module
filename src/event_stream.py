"""
Модуль потока событий.

Генерирует структурированные события занятости и распространяет их
через Server-Sent Events (SSE) для потребления в реальном времени.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    SPACE_OCCUPIED = "space_occupied"
    SPACE_FREED = "space_freed"
    OCCUPANCY_UPDATE = "occupancy_update"
    SYSTEM_STATUS = "system_status"
    FULL_SNAPSHOT = "full_snapshot"


@dataclass
class OccupancyEvent:
    """Одно событие занятости."""
    event_type: EventType
    space_id: str
    timestamp: str
    occupancy_pct: float
    confidence: float
    camera_id: Optional[str] = None
    previous_pct: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class EventBus:
    """
    Центральная шина событий: отслеживает состояние и рассылает события.

    - Хранит текущее состояние занятости всех мест.
    - Обнаруживает переходы состояния (свободно ↔ занято) и генерирует события.
    - Поддерживает множество SSE-подписчиков.
    """

    def __init__(
        self,
        space_ids: List[str],
        occupied_threshold: float = 60.0,
        freed_threshold: float = 30.0,
        hysteresis_seconds: float = 3.0,
    ):
        self.occupied_threshold = occupied_threshold
        self.freed_threshold = freed_threshold
        self.hysteresis_seconds = hysteresis_seconds

        # Текущее состояние
        self._state: Dict[str, float] = {sid: 0.0 for sid in space_ids}
        self._confidence: Dict[str, float] = {sid: 0.0 for sid in space_ids}
        self._is_occupied: Dict[str, bool] = {sid: False for sid in space_ids}
        self._last_transition: Dict[str, float] = {sid: 0.0 for sid in space_ids}

        # Подписчики (asyncio-очереди для SSE)
        self._subscribers: Set[asyncio.Queue] = set()

        # История событий (кольцевой буфер)
        self._history: List[OccupancyEvent] = []
        self._max_history = 500

    @property
    def state(self) -> Dict[str, float]:
        return dict(self._state)

    @property
    def occupied_spaces(self) -> List[str]:
        return [sid for sid, occ in self._is_occupied.items() if occ]

    @property
    def free_spaces(self) -> List[str]:
        return [sid for sid, occ in self._is_occupied.items() if not occ]

    @property
    def total_spaces(self) -> int:
        return len(self._state)

    @property
    def summary(self) -> dict:
        occupied = len(self.occupied_spaces)
        total = self.total_spaces
        return {
            "total": total,
            "occupied": occupied,
            "free": total - occupied,
            "occupancy_rate": round(occupied / total * 100, 1) if total > 0 else 0,
            "spaces": {
                sid: {
                    "occupancy_pct": self._state[sid],
                    "confidence": self._confidence[sid],
                    "is_occupied": self._is_occupied[sid],
                }
                for sid in sorted(self._state.keys())
            },
        }

    def update(
        self,
        space_id: str,
        occupancy_pct: float,
        confidence: float,
        camera_id: Optional[str] = None,
    ) -> Optional[OccupancyEvent]:
        """Обновить занятость одного места. Возвращает событие при смене состояния."""
        if space_id not in self._state:
            logger.warning("Unknown space_id: %s", space_id)
            return None

        prev_pct = self._state[space_id]
        self._state[space_id] = occupancy_pct
        self._confidence[space_id] = confidence

        now = time.time()
        event: Optional[OccupancyEvent] = None

        was_occupied = self._is_occupied[space_id]
        time_since = now - self._last_transition[space_id]

        # Гистерезис: требуется порог + временная задержка для смены состояния
        if not was_occupied and occupancy_pct >= self.occupied_threshold:
            if time_since >= self.hysteresis_seconds:
                self._is_occupied[space_id] = True
                self._last_transition[space_id] = now
                event = OccupancyEvent(
                    event_type=EventType.SPACE_OCCUPIED,
                    space_id=space_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    occupancy_pct=occupancy_pct,
                    confidence=confidence,
                    camera_id=camera_id,
                    previous_pct=prev_pct,
                )
        elif was_occupied and occupancy_pct <= self.freed_threshold:
            if time_since >= self.hysteresis_seconds:
                self._is_occupied[space_id] = False
                self._last_transition[space_id] = now
                event = OccupancyEvent(
                    event_type=EventType.SPACE_FREED,
                    space_id=space_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    occupancy_pct=occupancy_pct,
                    confidence=confidence,
                    camera_id=camera_id,
                    previous_pct=prev_pct,
                )

        if event is not None:
            self._push_event(event)

        return event

    def update_batch(
        self,
        updates: List[Dict[str, Any]],
    ) -> List[OccupancyEvent]:
        """Обновить несколько мест за раз. Возвращает список событий смены состояния."""
        events = []
        for u in updates:
            ev = self.update(
                space_id=u["space_id"],
                occupancy_pct=u["occupancy_pct"],
                confidence=u["confidence"],
                camera_id=u.get("camera_id"),
            )
            if ev is not None:
                events.append(ev)
        return events

    def get_snapshot_event(self) -> OccupancyEvent:
        """Создать событие-снимок с текущим состоянием."""
        return OccupancyEvent(
            event_type=EventType.FULL_SNAPSHOT,
            space_id="__all__",
            timestamp=datetime.now(timezone.utc).isoformat(),
            occupancy_pct=0,
            confidence=0,
            metadata=self.summary,
        )

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.add(q)
        logger.info("New SSE subscriber (total: %d)", len(self._subscribers))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)
        logger.info("SSE subscriber removed (total: %d)", len(self._subscribers))

    async def sse_generator(self, q: asyncio.Queue) -> AsyncGenerator[str, None]:
        """Асинхронный генератор, выдающий строки в формате SSE."""
        # Отправляем начальный снимок
        snapshot = self.get_snapshot_event()
        yield f"event: {snapshot.event_type.value}\ndata: {snapshot.to_json()}\n\n"

        try:
            while True:
                event: OccupancyEvent = await q.get()
                yield f"event: {event.event_type.value}\ndata: {event.to_json()}\n\n"
        except asyncio.CancelledError:
            pass

    def _push_event(self, event: OccupancyEvent) -> None:
        """Сохранить событие в историю и разослать подписчикам."""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("SSE subscriber queue full, dropping event")

    def get_recent_events(self, limit: int = 50) -> List[dict]:
        return [e.to_dict() for e in self._history[-limit:]]
