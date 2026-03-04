"""
Модуль калибровки камер и восстановления перспективы.

Обеспечивает:
- Вычисление матриц гомографии по реперным (калибровочным) точкам.
- Преобразование координат между картой (мировые) и изображением (пиксельные).
- Проецирование полигонов парковочных мест с векторной карты на плоскость изображения.
- Применение масок рабочих зон для исключения ненадёжных участков изображения.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationPoint:
    """Одна калибровочная реперная точка."""
    map_xy: Tuple[float, float]   # (x, y) в мировых/картовых координатах (метры)
    image_xy: Tuple[float, float]  # (x, y) в пиксельных координатах изображения


@dataclass
class CameraCalibration:
    """
    Хранит данные калибровки одной камеры.

    Гомография H отображает *картовые* координаты в *пиксельные*:
        image_point = H @ map_point

    Обратная H_inv отображает *пиксельные* координаты в *картовые*:
        map_point = H_inv @ image_point
    """

    camera_id: str
    resolution: Tuple[int, int]  # (ширина, высота)
    calibration_points: List[CalibrationPoint]
    work_zone: Optional[np.ndarray] = None  # полигон в пикс. координатах, форма (N, 2)
    H: Optional[np.ndarray] = None          # гомография 3x3, карта -> изображение
    H_inv: Optional[np.ndarray] = None      # обратная 3x3, изображение -> карта
    _work_zone_mask: Optional[np.ndarray] = field(default=None, repr=False)

    def calibrate(self) -> None:
        """Вычислить гомографию по заданным реперным точкам.

        Требуется минимум 4 неколлинеарных точки соответствия.
        Используется ``cv2.findHomography`` с RANSAC для устойчивости.
        """
        if len(self.calibration_points) < 4:
            raise ValueError(
                f"Camera {self.camera_id}: need >= 4 calibration points, "
                f"got {len(self.calibration_points)}"
            )

        src = np.array([p.map_xy for p in self.calibration_points], dtype=np.float64)
        dst = np.array([p.image_xy for p in self.calibration_points], dtype=np.float64)

        self.H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if self.H is None:
            raise RuntimeError(
                f"Camera {self.camera_id}: homography computation failed"
            )
        self.H_inv = np.linalg.inv(self.H)

        inliers = int(mask.sum()) if mask is not None else len(self.calibration_points)
        logger.info(
            "Camera %s calibrated: %d/%d inliers",
            self.camera_id, inliers, len(self.calibration_points),
        )

    def map_to_image(self, points: np.ndarray) -> np.ndarray:
        """Преобразовать точки из картовых (мировых) координат в пиксельные.

        Параметры
        ---------
        points : np.ndarray
            Массив формы (N, 2) с координатами (x, y) в метрах.

        Возвращает
        ----------
        np.ndarray
            Массив формы (N, 2) с координатами (x, y) в пикселях.
        """
        assert self.H is not None, "Camera not calibrated"
        return cv2.perspectiveTransform(
            points.reshape(-1, 1, 2).astype(np.float64), self.H
        ).reshape(-1, 2)

    def image_to_map(self, points: np.ndarray) -> np.ndarray:
        """Преобразовать точки из пиксельных координат в картовые (мировые)."""
        assert self.H_inv is not None, "Camera not calibrated"
        return cv2.perspectiveTransform(
            points.reshape(-1, 1, 2).astype(np.float64), self.H_inv
        ).reshape(-1, 2)

    def get_work_zone_mask(self) -> np.ndarray:
        """Вернуть бинарную маску рабочей зоны камеры.

        Маска: ``255`` внутри рабочей зоны, ``0`` за её пределами.
        """
        if self._work_zone_mask is not None:
            return self._work_zone_mask

        w, h = self.resolution
        mask = np.zeros((h, w), dtype=np.uint8)

        if self.work_zone is not None and len(self.work_zone) >= 3:
            pts = np.array(self.work_zone, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        else:
            mask[:] = 255  # рабочая зона не задана → весь кадр валиден

        self._work_zone_mask = mask
        return mask

    def is_inside_work_zone(self, point_img: Tuple[float, float]) -> bool:
        """Проверить, находится ли точка изображения внутри рабочей зоны."""
        mask = self.get_work_zone_mask()
        x, y = int(round(point_img[0])), int(round(point_img[1]))
        h, w = mask.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            return bool(mask[y, x] > 0)
        return False

    def project_parking_space(
        self, polygon_map: np.ndarray
    ) -> Optional[np.ndarray]:
        """Спроецировать полигон парковочного места из картовых координат в пиксельные.

        Параметры
        ---------
        polygon_map : np.ndarray
            Вершины полигона в картовых координатах, форма (N, 2).

        Возвращает
        ----------
        Optional[np.ndarray]
            Вершины полигона в пиксельных координатах, форма (N, 2).
            Возвращает ``None``, если проекция полностью за пределами
            рабочей зоны.
        """
        projected = self.map_to_image(polygon_map)

        # Проверяем, что хотя бы одна вершина внутри рабочей зоны
        inside = any(self.is_inside_work_zone(tuple(pt)) for pt in projected)
        if not inside:
            return None

        return projected

    def reprojection_error(self) -> float:
        """Вычислить среднюю ошибку репроекции в пикселях."""
        src = np.array([p.map_xy for p in self.calibration_points], dtype=np.float64)
        dst_expected = np.array(
            [p.image_xy for p in self.calibration_points], dtype=np.float64
        )
        dst_projected = self.map_to_image(src)
        errors = np.linalg.norm(dst_expected - dst_projected, axis=1)
        return float(errors.mean())


def build_calibration_from_config(camera_cfg: dict) -> CameraCalibration:
    """Создать ``CameraCalibration`` из словаря конфигурации камеры.

    Ожидаемый формат соответствует JSON в ``parking_map.json``.
    """
    cal_points = [
        CalibrationPoint(
            map_xy=tuple(cp["map"]),
            image_xy=tuple(cp["image"]),
        )
        for cp in camera_cfg["calibration_points"]
    ]

    work_zone = None
    if "work_zone" in camera_cfg:
        work_zone = np.array(camera_cfg["work_zone"], dtype=np.int32)

    cam = CameraCalibration(
        camera_id=camera_cfg["id"],
        resolution=tuple(camera_cfg["resolution"]),
        calibration_points=cal_points,
        work_zone=work_zone,
    )
    cam.calibrate()

    err = cam.reprojection_error()
    logger.info("Camera %s reprojection error: %.2f px", cam.camera_id, err)

    return cam
