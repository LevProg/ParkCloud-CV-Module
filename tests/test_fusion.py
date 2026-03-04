"""
Тесты модуля фузии нескольких камер.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.multi_camera import FusionStrategy, MultiCameraFusion, FusedOccupancy
from src.parking_analyzer import SpaceOccupancy


def _occ(space_id: str, camera_id: str, pct: float, conf: float) -> SpaceOccupancy:
    return SpaceOccupancy(
        space_id=space_id,
        camera_id=camera_id,
        occupancy_pct=pct,
        confidence=conf,
    )


class TestMultiCameraFusion:

    def test_single_camera(self):
        fusion = MultiCameraFusion(strategy=FusionStrategy.WEIGHTED_AVERAGE)
        results = [[_occ("A01", "cam1", 80.0, 0.9)]]
        fused = fusion.fuse(results)
        assert "A01" in fused
        assert fused["A01"].occupancy_pct == 80.0
        assert fused["A01"].camera_count == 1

    def test_weighted_average(self):
        fusion = MultiCameraFusion(strategy=FusionStrategy.WEIGHTED_AVERAGE)
        results = [
            [_occ("A01", "cam1", 90.0, 0.9)],
            [_occ("A01", "cam2", 60.0, 0.3)],
        ]
        fused = fusion.fuse(results)
        f = fused["A01"]
        assert f.camera_count == 2
        # Взвешенное среднее: (90*0.9 + 60*0.3) / (0.9+0.3) = 99/1.2 = 82.5
        assert abs(f.occupancy_pct - 82.5) < 0.5

    def test_max_confidence(self):
        fusion = MultiCameraFusion(strategy=FusionStrategy.MAX_CONFIDENCE)
        results = [
            [_occ("A01", "cam1", 50.0, 0.5)],
            [_occ("A01", "cam2", 80.0, 0.95)],
        ]
        fused = fusion.fuse(results)
        assert fused["A01"].occupancy_pct == 80.0

    def test_max_occupancy(self):
        fusion = MultiCameraFusion(strategy=FusionStrategy.MAX_OCCUPANCY)
        results = [
            [_occ("A01", "cam1", 30.0, 0.9)],
            [_occ("A01", "cam2", 85.0, 0.5)],
        ]
        fused = fusion.fuse(results)
        assert fused["A01"].occupancy_pct == 85.0

    def test_vote_strategy_occupied(self):
        fusion = MultiCameraFusion(strategy=FusionStrategy.VOTE)
        results = [
            [_occ("A01", "cam1", 80.0, 0.9)],
            [_occ("A01", "cam2", 70.0, 0.8)],
            [_occ("A01", "cam3", 20.0, 0.4)],
        ]
        fused = fusion.fuse(results)
        # 2 из 3 камер считают занятым → занято
        assert fused["A01"].occupancy_pct > 60

    def test_vote_strategy_free(self):
        fusion = MultiCameraFusion(strategy=FusionStrategy.VOTE)
        results = [
            [_occ("A01", "cam1", 10.0, 0.9)],
            [_occ("A01", "cam2", 20.0, 0.8)],
            [_occ("A01", "cam3", 80.0, 0.4)],
        ]
        fused = fusion.fuse(results)
        # 2 из 3 считают свободным → свободно
        assert fused["A01"].occupancy_pct < 60

    def test_multiple_spaces(self):
        fusion = MultiCameraFusion(strategy=FusionStrategy.WEIGHTED_AVERAGE)
        results = [
            [_occ("A01", "cam1", 90.0, 0.9), _occ("A02", "cam1", 10.0, 0.8)],
            [_occ("A01", "cam2", 80.0, 0.7), _occ("B01", "cam2", 50.0, 0.6)],
        ]
        fused = fusion.fuse(results)
        assert len(fused) == 3  # A01, A02, B01
        assert "A01" in fused
        assert "A02" in fused
        assert "B01" in fused
