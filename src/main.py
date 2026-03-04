"""
CV-модуль ParkCloud — FastAPI-приложение.

Обеспечивает:
- REST API для статуса занятости парковки
- SSE-эндпоинт для потока событий в реальном времени
- Эндпоинты обработки видеокадров
- Раздачу веб-интерфейса
- Демо-режим с синтетическими данными
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .calibration import CameraCalibration, build_calibration_from_config
from .detector import Detection, YOLODetector, SimulatedDetector, create_detector
from .event_stream import EventBus, EventType, OccupancyEvent
from .multi_camera import FusionStrategy, MultiCameraFusion
from .parking_analyzer import ParkingAnalyzer, ParkingSpace, SpaceOccupancy, build_spaces_from_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("parkcloud_cv")

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
STATIC_DIR = BASE_DIR / "static"
DEFAULT_MAP = CONFIG_DIR / "parking_map.json"

parking_map: Dict[str, Any] = {}
spaces: List[ParkingSpace] = []
calibrations: Dict[str, CameraCalibration] = {}
analyzers: Dict[str, ParkingAnalyzer] = {}
detector: YOLODetector | SimulatedDetector = None  # type: ignore
fusion: MultiCameraFusion = MultiCameraFusion()
event_bus: EventBus = None  # type: ignore
demo_task: Optional[asyncio.Task] = None


async def demo_loop():
    """Периодически обновляет места симулированными данными для демо."""
    rng = np.random.default_rng(42)
    # Начальное состояние: случайная занятость
    occupied_set = set(rng.choice(
        [s.id for s in spaces], size=min(8, len(spaces)), replace=False
    ))

    while True:
        now_str = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat()

        updates = []
        for s in spaces:
            if s.id in occupied_set:
                occ = float(rng.uniform(65, 98))
                conf = float(rng.uniform(0.7, 0.98))
            else:
                occ = float(rng.uniform(0, 20))
                conf = float(rng.uniform(0.1, 0.5))

            updates.append({
                "space_id": s.id,
                "occupancy_pct": round(occ, 1),
                "confidence": round(conf, 3),
                "camera_id": "demo",
            })

        events = event_bus.update_batch(updates)

        # Отправка периодического снимка
        snapshot = event_bus.get_snapshot_event()
        for q in list(event_bus._subscribers):
            try:
                q.put_nowait(snapshot)
            except asyncio.QueueFull:
                pass

        # Случайные переходы состояния
        for s in spaces:
            if rng.random() < 0.08:  # ~8% вероятность за цикл на место
                if s.id in occupied_set:
                    occupied_set.discard(s.id)
                else:
                    occupied_set.add(s.id)

        await asyncio.sleep(2.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global parking_map, spaces, calibrations, analyzers, detector, fusion, event_bus, demo_task

    # Загрузка карты парковки
    map_path = os.environ.get("PARKING_MAP", str(DEFAULT_MAP))
    logger.info("Loading parking map: %s", map_path)
    with open(map_path, "r", encoding="utf-8") as f:
        parking_map = json.load(f)

    # Создание парковочных мест
    spaces = build_spaces_from_config(parking_map)
    logger.info("Loaded %d parking spaces", len(spaces))

    # Калибровка камер
    for cam_cfg in parking_map.get("cameras", []):
        cal = build_calibration_from_config(cam_cfg)
        calibrations[cal.camera_id] = cal
        analyzer = ParkingAnalyzer(
            calibration=cal,
            spaces=spaces,
            covered_space_ids=cam_cfg.get("covered_spaces"),
        )
        analyzers[cal.camera_id] = analyzer
    logger.info("Configured %d cameras", len(calibrations))

    # Детектор
    use_yolo = os.environ.get("USE_YOLO", "true").lower() == "true"
    model_path = os.environ.get("YOLO_MODEL", "yolov8n.pt")
    device = os.environ.get("DEVICE", "cpu")
    detector = create_detector(use_yolo=use_yolo, model_path=model_path, device=device)

    # Фузия
    strategy_name = os.environ.get("FUSION_STRATEGY", "weighted_average")
    fusion = MultiCameraFusion(strategy=FusionStrategy(strategy_name))

    # Шина событий
    space_ids = [s.id for s in spaces]
    event_bus = EventBus(space_ids=space_ids)

    # Демо-режим
    if os.environ.get("DEMO_MODE", "true").lower() == "true":
        logger.info("Starting demo mode with simulated occupancy changes")
        demo_task = asyncio.create_task(demo_loop())

    yield

    # Завершение
    if demo_task is not None:
        demo_task.cancel()
        try:
            await demo_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="ParkCloud CV Module",
    description="Parking occupancy detection via computer vision",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Раздача статических файлов
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Отдать веб-интерфейс."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>ParkCloud CV Module</h1><p>Web UI not found.</p>")


@app.get("/api/status")
async def get_status():
    """Получить текущее сводное состояние парковки."""
    return event_bus.summary


@app.get("/api/spaces")
async def get_spaces():
    """Получить все парковочные места с текущей занятостью."""
    return {
        "spaces": [
            {
                "id": s.id,
                "type": s.space_type,
                "polygon": s.polygon_map.tolist(),
                "occupancy_pct": event_bus._state.get(s.id, 0),
                "confidence": event_bus._confidence.get(s.id, 0),
                "is_occupied": event_bus._is_occupied.get(s.id, False),
            }
            for s in spaces
        ]
    }


@app.get("/api/spaces/{space_id}")
async def get_space(space_id: str):
    """Получить статус занятости конкретного места."""
    if space_id not in event_bus._state:
        raise HTTPException(status_code=404, detail=f"Space {space_id} not found")
    return {
        "id": space_id,
        "occupancy_pct": event_bus._state[space_id],
        "confidence": event_bus._confidence[space_id],
        "is_occupied": event_bus._is_occupied[space_id],
    }


@app.get("/api/cameras")
async def get_cameras():
    """Получить информацию о камерах и статус калибровки."""
    return {
        "cameras": [
            {
                "id": cam_id,
                "resolution": list(cal.resolution),
                "calibration_points": len(cal.calibration_points),
                "reprojection_error": cal.reprojection_error(),
                "covered_spaces": [
                    s.id for s in analyzers[cam_id].spaces
                ],
            }
            for cam_id, cal in calibrations.items()
        ]
    }


@app.get("/api/map")
async def get_map():
    """Получить полную конфигурацию карты парковки."""
    return parking_map


@app.get("/api/events")
async def get_events(limit: int = Query(default=50, le=500)):
    """Получить недавние события занятости."""
    return {"events": event_bus.get_recent_events(limit)}


@app.get("/api/stream")
async def event_stream(request: Request):
    """SSE-эндпоинт для обновлений занятости в реальном времени."""
    q = event_bus.subscribe()

    async def generate():
        try:
            async for chunk in event_bus.sse_generator(q):
                if await request.is_disconnected():
                    break
                yield chunk
        finally:
            event_bus.unsubscribe(q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/process_frame/{camera_id}")
async def process_frame(camera_id: str, file: UploadFile = File(...)):
    """Обработать один видеокадр от камеры.

    Загрузите JPEG/PNG-изображение. Возвращает обнаруженные ТС и анализ занятости.
    """
    if camera_id not in analyzers:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Детекция ТС
    detections = detector.detect(frame)

    # Анализ занятости
    analyzer = analyzers[camera_id]
    occupancy_results = analyzer.analyze(detections)

    # Обновление шины событий
    for occ in occupancy_results:
        event_bus.update(
            space_id=occ.space_id,
            occupancy_pct=occ.occupancy_pct,
            confidence=occ.confidence,
            camera_id=camera_id,
        )

    return {
        "camera_id": camera_id,
        "detections": [
            {
                "bbox": det.bbox,
                "confidence": det.confidence,
                "class_name": det.class_name,
                "center": det.center,
            }
            for det in detections
        ],
        "occupancy": [
            {
                "space_id": occ.space_id,
                "occupancy_pct": occ.occupancy_pct,
                "confidence": occ.confidence,
            }
            for occ in occupancy_results
        ],
    }


@app.post("/api/process_frame/{camera_id}/annotated")
async def process_frame_annotated(camera_id: str, file: UploadFile = File(...)):
    """Обработать кадр и вернуть аннотированное изображение с наложениями."""
    if camera_id not in analyzers:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Детекция и аннотация ТС
    if isinstance(detector, YOLODetector):
        annotated, detections = detector.detect_and_annotate(frame)
    else:
        detections = detector.detect(frame)
        annotated = frame.copy()

    # Анализ и наложение парковочных мест
    analyzer = analyzers[camera_id]
    occupancy_results = analyzer.analyze(detections)
    annotated = analyzer.annotate_frame(annotated, occupancy_results)

    # Кодирование в JPEG
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])

    return StreamingResponse(
        iter([buffer.tobytes()]),
        media_type="image/jpeg",
    )


@app.post("/api/config/fusion")
async def set_fusion_strategy(strategy: str = Query(...)):
    """Изменить стратегию фузии нескольких камер."""
    global fusion
    try:
        fusion = MultiCameraFusion(strategy=FusionStrategy(strategy))
        return {"status": "ok", "strategy": strategy}
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy. Available: {[s.value for s in FusionStrategy]}",
        )


@app.get("/api/config")
async def get_config():
    """Получить текущую конфигурацию модуля."""
    return {
        "parking_map": os.environ.get("PARKING_MAP", str(DEFAULT_MAP)),
        "use_yolo": os.environ.get("USE_YOLO", "true"),
        "yolo_model": os.environ.get("YOLO_MODEL", "yolov8n.pt"),
        "device": os.environ.get("DEVICE", "cpu"),
        "fusion_strategy": fusion.strategy.value,
        "demo_mode": os.environ.get("DEMO_MODE", "true"),
        "cameras_configured": len(calibrations),
        "total_spaces": len(spaces),
    }
