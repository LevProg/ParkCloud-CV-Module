"""
Тесты модуля калибровки.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration import CalibrationPoint, CameraCalibration, build_calibration_from_config


class TestCameraCalibration:
    """Тесты для CameraCalibration."""

    def _make_calibration(self):
        """Создать тестовую калибровку с известными точками."""
        points = [
            CalibrationPoint(map_xy=(0.0, 0.0), image_xy=(100.0, 100.0)),
            CalibrationPoint(map_xy=(10.0, 0.0), image_xy=(500.0, 100.0)),
            CalibrationPoint(map_xy=(10.0, 10.0), image_xy=(500.0, 500.0)),
            CalibrationPoint(map_xy=(0.0, 10.0), image_xy=(100.0, 500.0)),
        ]
        cal = CameraCalibration(
            camera_id="test_cam",
            resolution=(640, 480),
            calibration_points=points,
        )
        return cal

    def test_calibrate_success(self):
        cal = self._make_calibration()
        cal.calibrate()
        assert cal.H is not None
        assert cal.H_inv is not None
        assert cal.H.shape == (3, 3)

    def test_calibrate_requires_4_points(self):
        cal = CameraCalibration(
            camera_id="test",
            resolution=(640, 480),
            calibration_points=[
                CalibrationPoint(map_xy=(0, 0), image_xy=(0, 0)),
                CalibrationPoint(map_xy=(1, 0), image_xy=(1, 0)),
                CalibrationPoint(map_xy=(1, 1), image_xy=(1, 1)),
            ],
        )
        with pytest.raises(ValueError, match="need >= 4"):
            cal.calibrate()

    def test_map_to_image_roundtrip(self):
        cal = self._make_calibration()
        cal.calibrate()

        original = np.array([[5.0, 5.0], [2.0, 8.0]], dtype=np.float64)
        image_pts = cal.map_to_image(original)
        recovered = cal.image_to_map(image_pts)

        np.testing.assert_allclose(recovered, original, atol=0.5)

    def test_reprojection_error_low(self):
        cal = self._make_calibration()
        cal.calibrate()
        error = cal.reprojection_error()
        # Для простого аффинного преобразования ошибка должна быть минимальной
        assert error < 2.0, f"Reprojection error too high: {error}"

    def test_work_zone_mask(self):
        cal = self._make_calibration()
        cal.work_zone = np.array([[50, 50], [600, 50], [600, 400], [50, 400]], dtype=np.int32)
        cal.calibrate()

        mask = cal.get_work_zone_mask()
        assert mask.shape == (480, 640)
        assert mask[200, 300] == 255  # внутри
        assert mask[10, 10] == 0     # снаружи

    def test_is_inside_work_zone(self):
        cal = self._make_calibration()
        cal.work_zone = np.array([[50, 50], [600, 50], [600, 400], [50, 400]], dtype=np.int32)
        cal.calibrate()

        assert cal.is_inside_work_zone((300, 200)) is True
        assert cal.is_inside_work_zone((10, 10)) is False

    def test_project_parking_space(self):
        cal = self._make_calibration()
        cal.work_zone = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], dtype=np.int32)
        cal.calibrate()

        polygon = np.array([[2.0, 2.0], [5.0, 2.0], [5.0, 7.0], [2.0, 7.0]], dtype=np.float64)
        projected = cal.project_parking_space(polygon)
        assert projected is not None
        assert projected.shape == (4, 2)


class TestBuildFromConfig:
    """Тест создания калибровки из JSON-конфига."""

    def test_build_from_config(self):
        cam_cfg = {
            "id": "cam_test",
            "resolution": [1920, 1080],
            "calibration_points": [
                {"map": [0.0, 0.0], "image": [100, 100]},
                {"map": [40.0, 0.0], "image": [1800, 100]},
                {"map": [40.0, 25.0], "image": [1800, 900]},
                {"map": [0.0, 25.0], "image": [100, 900]},
            ],
            "work_zone": [[50, 50], [1870, 50], [1870, 1030], [50, 1030]],
        }

        cal = build_calibration_from_config(cam_cfg)
        assert cal.camera_id == "cam_test"
        assert cal.H is not None
        assert cal.reprojection_error() < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
