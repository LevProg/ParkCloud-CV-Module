"""
Генератор демо-данных.

Создаёт синтетические изображения парковки с «машинами» в виде цветных
прямоугольников для тестирования конвейера без реальных камер.
Также демонстрирует полный конвейер обработки от начала до конца.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Добавляем корневой каталог проекта в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration import build_calibration_from_config
from src.parking_analyzer import ParkingAnalyzer, build_spaces_from_config


def generate_parking_image(
    width: int = 1920,
    height: int = 1080,
    occupied_spaces: list[str] | None = None,
    parking_map: dict | None = None,
    calibration=None,
) -> np.ndarray:
    """Сгенерировать синтетическое изображение парковки как вид с камеры.

    Параметры
    ----------
    width, height : int
        Размер выходного изображения.
    occupied_spaces : list[str]
        ID мест, на которых должна быть нарисована машина.
    parking_map : dict
        Полная конфигурация карты парковки.
    calibration : CameraCalibration
        Объект калибровки камеры.

    Возвращает
    ----------
    np.ndarray
        BGR-изображение.
    """
    img = np.full((height, width, 3), (60, 65, 70), dtype=np.uint8)

    # Рисуем текстуру асфальта (шум)
    noise = np.random.randint(0, 15, (height, width), dtype=np.uint8)
    img[:, :, 0] = np.clip(img[:, :, 0].astype(int) + noise, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip(img[:, :, 1].astype(int) + noise, 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip(img[:, :, 2].astype(int) + noise, 0, 255).astype(np.uint8)

    if parking_map is None or calibration is None:
        return img

    occupied_set = set(occupied_spaces or [])
    spaces = build_spaces_from_config(parking_map)

    for space in spaces:
        # Проецируем полигон места на изображение
        proj = calibration.project_parking_space(space.polygon_map)
        if proj is None:
            continue

        pts = proj.astype(np.int32)

        # Рисуем разметку парковочного места (белые линии)
        cv2.polylines(img, [pts], True, (200, 200, 200), 2)

        # Номер места
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())

        if space.id in occupied_set:
            # Рисуем «машину» — залитый прямоугольник со случайным цветом
            rng = np.random.RandomState(hash(space.id) % 2**31)
            car_color = (
                int(rng.randint(30, 200)),
                int(rng.randint(30, 200)),
                int(rng.randint(30, 200)),
            )
            # Уменьшаем полигон для машины
            center = pts.mean(axis=0)
            shrunk = (pts - center) * 0.8 + center
            shrunk = shrunk.astype(np.int32)
            cv2.fillPoly(img, [shrunk], car_color)
            # Контур машины
            cv2.polylines(img, [shrunk], True, (20, 20, 20), 2)
            # Эффект лобового стекла
            ws = (shrunk[0] + shrunk[1]) // 2
            we = (shrunk[0] * 3 + shrunk[3]) // 4
            cv2.line(img, tuple(ws), tuple(we), (150, 180, 200), 2)
        else:
            # Номер места на пустом месте
            cv2.putText(
                img, space.id, (cx - 15, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
            )

    # Рисуем разметку проезда
    if calibration:
        lane_y_map = 12.5  # координата Y центра проезда в системе координат карты
        left = calibration.map_to_image(np.array([[0.0, lane_y_map]]))[0]
        right = calibration.map_to_image(np.array([[60.0, lane_y_map]]))[0]
        pt1 = (int(left[0]), int(left[1]))
        pt2 = (int(right[0]), int(right[1]))
        # Пунктирная осевая линия
        for i in range(0, 20):
            t1 = i / 20.0
            t2 = (i + 0.5) / 20.0
            p1 = (int(left[0] + (right[0] - left[0]) * t1),
                   int(left[1] + (right[1] - left[1]) * t1))
            p2 = (int(left[0] + (right[0] - left[0]) * t2),
                   int(left[1] + (right[1] - left[1]) * t2))
            cv2.line(img, p1, p2, (0, 200, 255), 2)

    return img


def generate_demo_images(output_dir: str = "demo/images"):
    """Сгенерировать набор демо-изображений для тестирования."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "parking_map.json"
    with open(config_path, "r", encoding="utf-8") as f:
        parking_map = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Определяем сценарии
    scenarios = {
        "empty": [],
        "half_full": ["A01", "A03", "A05", "A07", "A09", "B02", "B04", "B06", "B08", "B10"],
        "full": [f"{r}{i:02d}" for r in "AB" for i in range(1, 11)],
        "rush_hour": ["A01", "A02", "A03", "A04", "A05", "A06", "A08",
                       "B01", "B02", "B03", "B05", "B07", "B08", "B09", "B10"],
    }

    for cam_cfg in parking_map["cameras"]:
        cal = build_calibration_from_config(cam_cfg)

        for scenario_name, occupied in scenarios.items():
            img = generate_parking_image(
                width=cam_cfg["resolution"][0],
                height=cam_cfg["resolution"][1],
                occupied_spaces=occupied,
                parking_map=parking_map,
                calibration=cal,
            )

            filename = f"{cam_cfg['id']}_{scenario_name}.jpg"
            filepath = out / filename
            cv2.imwrite(str(filepath), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"  Generated: {filepath}")

    print(f"\nDone! {len(scenarios) * len(parking_map['cameras'])} images generated in {out}")


def run_pipeline_demo():
    """Запустить полный конвейер детекции на сгенерированных демо-изображениях."""
    from src.detector import create_detector
    from src.multi_camera import MultiCameraFusion, FusionStrategy
    from src.event_stream import EventBus

    config_path = Path(__file__).resolve().parent.parent / "config" / "parking_map.json"
    with open(config_path, "r", encoding="utf-8") as f:
        parking_map = json.load(f)

    spaces = build_spaces_from_config(parking_map)
    space_ids = [s.id for s in spaces]
    event_bus = EventBus(space_ids=space_ids)
    detector = create_detector(use_yolo=True)
    fusion = MultiCameraFusion(strategy=FusionStrategy.WEIGHTED_AVERAGE)

    # Калибровка и анализаторы
    calibrations = {}
    analyzers = {}
    for cam_cfg in parking_map["cameras"]:
        cal = build_calibration_from_config(cam_cfg)
        calibrations[cal.camera_id] = cal
        analyzers[cal.camera_id] = ParkingAnalyzer(
            calibration=cal,
            spaces=spaces,
            covered_space_ids=cam_cfg.get("covered_spaces"),
        )

    # Генерация тестового изображения
    occupied = ["A01", "A03", "A05", "B02", "B04", "B06", "B08"]
    print(f"\n{'='*60}")
    print(f"Pipeline Demo — occupied spaces: {occupied}")
    print(f"{'='*60}\n")

    all_results = []
    for cam_id, cal in calibrations.items():
        cam_cfg = next(c for c in parking_map["cameras"] if c["id"] == cam_id)
        img = generate_parking_image(
            width=cam_cfg["resolution"][0],
            height=cam_cfg["resolution"][1],
            occupied_spaces=occupied,
            parking_map=parking_map,
            calibration=cal,
        )

        detections = detector.detect(img)
        print(f"Camera {cam_id}: {len(detections)} vehicles detected")

        results = analyzers[cam_id].analyze(detections)
        all_results.append(results)

        for r in results:
            print(f"  {r.space_id}: {r.occupancy_pct:.0f}% (conf={r.confidence:.2f})")

    # Фузия результатов
    fused = fusion.fuse(all_results)
    print(f"\n{'='*60}")
    print("Fused results:")
    print(f"{'='*60}")
    for sid in sorted(fused.keys()):
        f = fused[sid]
        status = "OCCUPIED" if f.occupancy_pct >= 60 else "FREE"
        print(f"  {sid}: {f.occupancy_pct:.0f}% ({f.camera_count} cam) → {status}")

    # Обновление шины событий
    for sid, f in fused.items():
        event_bus.update(sid, f.occupancy_pct, f.confidence)

    summary = event_bus.summary
    print(f"\nSummary: {summary['occupied']}/{summary['total']} occupied "
          f"({summary['occupancy_rate']}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ParkCloud CV Demo Tools")
    parser.add_argument(
        "action",
        choices=["generate", "pipeline"],
        help="Action: 'generate' demo images or run 'pipeline' demo",
    )
    parser.add_argument(
        "--output", "-o",
        default="demo/images",
        help="Output directory for generated images",
    )
    args = parser.parse_args()

    if args.action == "generate":
        print("Generating demo parking lot images...")
        generate_demo_images(args.output)
    elif args.action == "pipeline":
        run_pipeline_demo()
