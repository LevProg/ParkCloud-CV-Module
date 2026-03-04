"""
Анализатор занятости парковочных мест.

Для каждого парковочного места, видимого камерой, определяет занятость
путём вычисления перекрытия (IoU) между обнаруженными ТС
и спроецированным полигоном парковочного места.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .calibration import CameraCalibration
from .detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class SpaceOccupancy:
    """Результат анализа занятости одного места от одной камеры."""
    space_id: str
    camera_id: str
    occupancy_pct: float          # 0.0 – 100.0
    confidence: float             # 0.0 – 1.0 (уверенность детектора лучшего совпадения)
    matched_detection: Optional[Detection] = None
    projected_polygon: Optional[np.ndarray] = None  # полигон в пикс. координатах


@dataclass
class ParkingSpace:
    """Парковочное место из векторной карты."""
    id: str
    polygon_map: np.ndarray   # (N, 2) в картовых координатах
    space_type: str = "standard"


class ParkingAnalyzer:
    """
    Анализирует занятость парковочных мест для одной камеры.

    Конвейер на кадр:
    1. Получить обнаруженные ТС (список ``Detection``).
    2. Для каждого места, покрываемого этой камерой:
       a. Спроецировать полигон карты на плоскость изображения.
       b. Вычислить перекрытие с каждым bbox ТС.
       c. Выбрать наилучшее совпадение (максимальное перекрытие).
       d. Сформировать процент занятости.
    """

    def __init__(
        self,
        calibration: CameraCalibration,
        spaces: List[ParkingSpace],
        covered_space_ids: Optional[List[str]] = None,
        overlap_threshold: float = 0.15,
    ):
        self.calibration = calibration
        self.overlap_threshold = overlap_threshold

        # Фильтрация мест, покрываемых этой камерой
        if covered_space_ids is not None:
            covered = set(covered_space_ids)
            self.spaces = [s for s in spaces if s.id in covered]
        else:
            self.spaces = list(spaces)

        # Предварительное проецирование полигонов
        self._projected: Dict[str, Optional[np.ndarray]] = {}
        for sp in self.spaces:
            proj = self.calibration.project_parking_space(sp.polygon_map)
            self._projected[sp.id] = proj

        n_visible = sum(1 for v in self._projected.values() if v is not None)
        logger.info(
            "ParkingAnalyzer [%s]: %d/%d spaces visible",
            calibration.camera_id, n_visible, len(self.spaces),
        )

    def analyze(self, detections: List[Detection]) -> List[SpaceOccupancy]:
        """Анализировать занятость по списку детекций текущего кадра.

        Возвращает список ``SpaceOccupancy`` для каждого покрываемого места.
        """
        results: List[SpaceOccupancy] = []

        for space in self.spaces:
            proj = self._projected.get(space.id)
            if proj is None:
                # Место не видно этой камере
                continue

            best_overlap = 0.0
            best_conf = 0.0
            best_det: Optional[Detection] = None

            for det in detections:
                overlap = self._compute_overlap(proj, det.bbox)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_conf = det.confidence
                    best_det = det

            # Процент занятости: комбинация перекрытия и уверенности детектора
            if best_overlap >= self.overlap_threshold:
                occupancy = min(100.0, best_overlap * best_conf * 100.0 / 0.5)
                # Ограничение: высокое перекрытие + высокая уверенность → близко к 100
                occupancy = min(100.0, max(0.0, occupancy))
            else:
                occupancy = 0.0
                best_conf = 0.0
                best_det = None

            results.append(SpaceOccupancy(
                space_id=space.id,
                camera_id=self.calibration.camera_id,
                occupancy_pct=round(occupancy, 1),
                confidence=round(best_conf, 3),
                matched_detection=best_det,
                projected_polygon=proj,
            ))

        return results

    def annotate_frame(
        self,
        frame: np.ndarray,
        occupancy_results: List[SpaceOccupancy],
    ) -> np.ndarray:
        """Нарисовать наложения парковочных мест на кадре.

        Зелёный = свободно, Красный = занято, Жёлтый = неопределённо.
        """
        annotated = frame.copy()

        for occ in occupancy_results:
            poly = occ.projected_polygon
            if poly is None:
                continue

            pts = poly.astype(np.int32)

            if occ.occupancy_pct >= 70:
                color = (0, 0, 200)       # Красный → занято
                fill_alpha = 0.35
            elif occ.occupancy_pct >= 30:
                color = (0, 200, 200)     # Жёлтый → неопределённо
                fill_alpha = 0.25
            else:
                color = (0, 200, 0)       # Зелёный → свободно
                fill_alpha = 0.20

            # Полупрозрачная заливка
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, fill_alpha, annotated, 1 - fill_alpha, 0, annotated)

            # Граница
            cv2.polylines(annotated, [pts], True, color, 2)

            # Подпись
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            label = f"{occ.space_id}: {occ.occupancy_pct:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(
                annotated,
                (cx - tw // 2 - 2, cy - th // 2 - 2),
                (cx + tw // 2 + 2, cy + th // 2 + 2),
                (0, 0, 0), -1,
            )
            cv2.putText(
                annotated, label, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )

        return annotated

    @staticmethod
    def _compute_overlap(
        polygon: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> float:
        """Вычислить коэффициент перекрытия между полигоном и bbox.

        Возвращает отношение площади пересечения к площади полигона (0–1).
        Показывает, какая доля парковочного места покрыта ТС.
        """
        x1, y1, x2, y2 = bbox
        bbox_poly = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ], dtype=np.float32)

        space_poly = polygon.astype(np.float32)

        # Пересечение контуров через cv2
        ret, intersection = cv2.intersectConvexConvex(space_poly, bbox_poly)
        if ret <= 0 or intersection is None:
            return 0.0

        inter_area = cv2.contourArea(intersection)
        space_area = cv2.contourArea(space_poly)

        if space_area <= 0:
            return 0.0

        return float(inter_area / space_area)


def build_spaces_from_config(parking_cfg: dict) -> List[ParkingSpace]:
    """Создать объекты ``ParkingSpace`` из конфигурации карты парковки."""
    spaces = []
    for sp_cfg in parking_cfg.get("parking_spaces", []):
        spaces.append(ParkingSpace(
            id=sp_cfg["id"],
            polygon_map=np.array(sp_cfg["polygon"], dtype=np.float64),
            space_type=sp_cfg.get("type", "standard"),
        ))
    return spaces
