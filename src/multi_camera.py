"""
Модуль фузии нескольких камер.

Когда одно и то же парковочное место наблюдается несколькими камерами,
этот модуль объединяет их показания занятости в единый
комплексный показатель с помощью настраиваемых стратегий фузии.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from .parking_analyzer import SpaceOccupancy

logger = logging.getLogger(__name__)


class FusionStrategy(str, Enum):
    """Доступные стратегии объединения показаний нескольких камер."""
    MAX_CONFIDENCE = "max_confidence"       # Использовать показание с макс. уверенностью детектора
    WEIGHTED_AVERAGE = "weighted_average"   # Взвешенное среднее по уверенности
    MAX_OCCUPANCY = "max_occupancy"         # Использовать максимальный процент занятости
    VOTE = "vote"                           # Голосование (занято, если >50% камер согласны)


@dataclass
class FusedOccupancy:
    """Комплексный результат занятости места по всем камерам."""
    space_id: str
    occupancy_pct: float      # 0.0 – 100.0
    confidence: float         # 0.0 – 1.0
    camera_count: int         # количество камер-источников
    readings: List[SpaceOccupancy]  # показания отдельных камер
    strategy_used: str


class MultiCameraFusion:
    """
    Объединяет показания занятости от нескольких камер по каждому месту.

    Пример:
        fusion = MultiCameraFusion(strategy=FusionStrategy.WEIGHTED_AVERAGE)
        results_cam1 = analyzer_cam1.analyze(detections_cam1)
        results_cam2 = analyzer_cam2.analyze(detections_cam2)
        fused = fusion.fuse([results_cam1, results_cam2])
    """

    def __init__(
        self,
        strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE,
        occupied_threshold: float = 60.0,
    ):
        self.strategy = strategy
        self.occupied_threshold = occupied_threshold

    def fuse(
        self, camera_results: List[List[SpaceOccupancy]]
    ) -> Dict[str, FusedOccupancy]:
        """Объединить результаты занятости от нескольких камер.

        Параметры
        ---------
        camera_results : List[List[SpaceOccupancy]]
            Каждый элемент — список ``SpaceOccupancy`` от одной камеры.

        Возвращает
        ----------
        Dict[str, FusedOccupancy]
            Ключ — space_id.
        """
        # Группировка показаний по space_id
        by_space: Dict[str, List[SpaceOccupancy]] = {}
        for cam_result in camera_results:
            for occ in cam_result:
                by_space.setdefault(occ.space_id, []).append(occ)

        # Фузия по каждому месту
        fused: Dict[str, FusedOccupancy] = {}
        for space_id, readings in by_space.items():
            fused[space_id] = self._fuse_one(space_id, readings)

        return fused

    def _fuse_one(
        self, space_id: str, readings: List[SpaceOccupancy]
    ) -> FusedOccupancy:
        """Применить стратегию фузии к показаниям одного места."""
        if len(readings) == 1:
            r = readings[0]
            return FusedOccupancy(
                space_id=space_id,
                occupancy_pct=r.occupancy_pct,
                confidence=r.confidence,
                camera_count=1,
                readings=readings,
                strategy_used=self.strategy.value,
            )

        if self.strategy == FusionStrategy.MAX_CONFIDENCE:
            return self._fuse_max_confidence(space_id, readings)
        elif self.strategy == FusionStrategy.MAX_OCCUPANCY:
            return self._fuse_max_occupancy(space_id, readings)
        elif self.strategy == FusionStrategy.VOTE:
            return self._fuse_vote(space_id, readings)
        else:  # По умолчанию: взвешенное среднее
            return self._fuse_weighted_average(space_id, readings)

    def _fuse_max_confidence(
        self, space_id: str, readings: List[SpaceOccupancy]
    ) -> FusedOccupancy:
        best = max(readings, key=lambda r: r.confidence)
        return FusedOccupancy(
            space_id=space_id,
            occupancy_pct=best.occupancy_pct,
            confidence=best.confidence,
            camera_count=len(readings),
            readings=readings,
            strategy_used=self.strategy.value,
        )

    def _fuse_max_occupancy(
        self, space_id: str, readings: List[SpaceOccupancy]
    ) -> FusedOccupancy:
        best = max(readings, key=lambda r: r.occupancy_pct)
        return FusedOccupancy(
            space_id=space_id,
            occupancy_pct=best.occupancy_pct,
            confidence=best.confidence,
            camera_count=len(readings),
            readings=readings,
            strategy_used=self.strategy.value,
        )

    def _fuse_weighted_average(
        self, space_id: str, readings: List[SpaceOccupancy]
    ) -> FusedOccupancy:
        total_conf = sum(r.confidence for r in readings)
        if total_conf == 0:
            avg_occ = sum(r.occupancy_pct for r in readings) / len(readings)
            avg_conf = 0.0
        else:
            avg_occ = sum(
                r.occupancy_pct * r.confidence for r in readings
            ) / total_conf
            avg_conf = total_conf / len(readings)

        return FusedOccupancy(
            space_id=space_id,
            occupancy_pct=round(avg_occ, 1),
            confidence=round(avg_conf, 3),
            camera_count=len(readings),
            readings=readings,
            strategy_used=self.strategy.value,
        )

    def _fuse_vote(
        self, space_id: str, readings: List[SpaceOccupancy]
    ) -> FusedOccupancy:
        votes_occupied = sum(
            1 for r in readings if r.occupancy_pct >= self.occupied_threshold
        )
        is_occupied = votes_occupied > len(readings) / 2

        if is_occupied:
            occ_readings = [r for r in readings if r.occupancy_pct >= self.occupied_threshold]
            avg_occ = sum(r.occupancy_pct for r in occ_readings) / len(occ_readings)
            avg_conf = sum(r.confidence for r in occ_readings) / len(occ_readings)
        else:
            free_readings = [r for r in readings if r.occupancy_pct < self.occupied_threshold]
            avg_occ = sum(r.occupancy_pct for r in free_readings) / max(1, len(free_readings))
            avg_conf = sum(r.confidence for r in free_readings) / max(1, len(free_readings))

        return FusedOccupancy(
            space_id=space_id,
            occupancy_pct=round(avg_occ, 1),
            confidence=round(avg_conf, 3),
            camera_count=len(readings),
            readings=readings,
            strategy_used=self.strategy.value,
        )
