"""
Модуль детекции транспортных средств.

Использует YOLOv8 (ultralytics) для обнаружения ТС.
При недоступности модели переключается на симулятор детекций,
что позволяет запускать демо без GPU / загрузки тяжёлых моделей.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Идентификаторы классов COCO для транспорта
VEHICLE_CLASS_IDS = {2, 5, 7}  # легковой, автобус, грузовик
VEHICLE_CLASS_NAMES = {"car", "bus", "truck"}


@dataclass
class Detection:
    """Одна обнаруженная сущность."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) в пикселях
    confidence: float                 # 0.0 – 1.0
    class_id: int
    class_name: str
    center: Tuple[float, float]       # (cx, cy) центр bbox

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


class YOLODetector:
    """
    Детектор транспорта на базе YOLOv8.

    Оборачивает ``ultralytics.YOLO`` и фильтрует детекции только по классам ТС.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        device: str = "cpu",
    ):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self._model_path = model_path
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            self.model = YOLO(self._model_path)
            logger.info("YOLO model loaded: %s (device=%s)", self._model_path, self.device)
        except Exception as e:
            logger.warning("Could not load YOLO model: %s. Using simulated detector.", e)
            self.model = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Запустить детекцию на одном кадре.

        Параметры
        ---------
        frame : np.ndarray
            BGR-изображение (H, W, 3).

        Возвращает
        ----------
        List[Detection]
            Обнаруженные ТС.
        """
        if self.model is None:
            return []

        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                cls_name = r.names.get(cls_id, "unknown")
                if cls_name not in VEHICLE_CLASS_NAMES and cls_id not in VEHICLE_CLASS_IDS:
                    continue

                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                detections.append(Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    center=(cx, cy),
                ))

        logger.debug("Detected %d vehicles in frame", len(detections))
        return detections

    def detect_and_annotate(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Detection]]:
        """Обнаружить ТС и нарисовать рамки на копии кадра."""
        detections = self.detect(frame)
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(
                annotated, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )

        return annotated, detections


class SimulatedDetector:
    """
    Симулятор детекций, генерирует случайные детекции для демо.
    Используется, когда модель YOLO недоступна.
    """

    def __init__(self, avg_detections: int = 8, seed: Optional[int] = None):
        self.avg_detections = avg_detections
        self.rng = np.random.default_rng(seed)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        n = self.rng.poisson(self.avg_detections)
        detections: List[Detection] = []

        for _ in range(n):
            bw = self.rng.integers(60, 180)
            bh = self.rng.integers(40, 120)
            x1 = self.rng.integers(0, max(1, w - bw))
            y1 = self.rng.integers(0, max(1, h - bh))
            x2 = x1 + bw
            y2 = y1 + bh
            conf = float(self.rng.uniform(0.4, 0.98))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            detections.append(Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=conf,
                class_id=2,
                class_name="car",
                center=(cx, cy),
            ))

        return detections


def create_detector(
    use_yolo: bool = True,
    model_path: str = "yolov8n.pt",
    confidence_threshold: float = 0.35,
    device: str = "cpu",
) -> YOLODetector | SimulatedDetector:
    """Фабрика: создать наилучший доступный детектор."""
    if use_yolo:
        det = YOLODetector(model_path, confidence_threshold, device)
        if det.model is not None:
            return det
        logger.info("Falling back to simulated detector")

    return SimulatedDetector()
