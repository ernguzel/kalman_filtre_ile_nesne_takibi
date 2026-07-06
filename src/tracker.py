from typing import Dict, Optional, Tuple

from src.kalman_bbox import BBoxMeasurement, KalmanFilterBBox
from src.utils import xyxy_to_cxcywh


class SinglePersonKalmanTracker:
    """
    Tek insan için basit Kalman tracker.

    Amaç:
        YOLO bbox gelirse Kalman update
        YOLO bbox gelmezse Kalman predict
        Çok uzun süre ölçüm gelmezse track pasif
        Tekrar ölçüm gelirse yeni track başlat
    """

    def __init__(
        self,
        kalman_config: Dict,
        max_lost_count: int = 10,
        predict_when_missing: bool = True,
        restart_when_detection_returns: bool = True,
    ):
        self.kalman_config = kalman_config
        self.max_lost_count = max_lost_count
        self.predict_when_missing = predict_when_missing
        self.restart_when_detection_returns = restart_when_detection_returns

        self.kf: Optional[KalmanFilterBBox] = None

        self.active = False
        self.track_id = 0
        self.lost_count = 0

        self.last_status = "inactive"

    def _create_kalman(self) -> KalmanFilterBBox:
        return KalmanFilterBBox(
            dt=float(self.kalman_config["dt"]),
            initial_p=float(self.kalman_config["initial_p"]),
            accel_noise_pos=float(self.kalman_config["accel_noise_pos"]),
            accel_noise_size=float(self.kalman_config["accel_noise_size"]),
            measurement_noise_pos=float(self.kalman_config["measurement_noise_pos"]),
            measurement_noise_size=float(self.kalman_config["measurement_noise_size"]),
        )

    def _start_new_track(self, measurement: BBoxMeasurement) -> Dict:
        self.track_id += 1
        self.kf = self._create_kalman()
        self.kf.initialize(measurement)

        self.active = True
        self.lost_count = 0
        self.last_status = "new_track"

        return self._build_output()

    def _deactivate(self) -> None:
        self.active = False
        self.kf = None
        self.lost_count = 0
        self.last_status = "deactivated"

    def update(self, detection: Optional[Dict]) -> Optional[Dict]:
        """
        detection:
            None veya detector.py çıktısından bir dict.

        Output:
            None veya tracker sonucu dict.
        """

        measurement = None

        if detection is not None:
            measurement = xyxy_to_cxcywh(detection["bbox_xyxy"])

        # Track pasifse
        if not self.active:
            if measurement is None:
                self.last_status = "inactive_waiting"
                return None

            if self.restart_when_detection_returns:
                return self._start_new_track(measurement)

            return None

        # Track aktifse
        if measurement is None:
            self.lost_count += 1

            if self.lost_count > self.max_lost_count:
                self._deactivate()
                return None

            if not self.predict_when_missing:
                self.last_status = "missing_no_predict"
                return None

            assert self.kf is not None
            self.kf.step(None)
            self.last_status = "predict_only"
            return self._build_output()

        # Ölçüm varsa predict + update
        self.lost_count = 0

        assert self.kf is not None
        self.kf.step(measurement)

        self.last_status = "update"
        return self._build_output()

    def _build_output(self) -> Dict:
        assert self.kf is not None

        bbox_cxcywh = self.kf.get_bbox_cxcywh()
        velocity = self.kf.get_velocity()

        return {
            "track_id": self.track_id,
            "bbox_cxcywh": bbox_cxcywh,
            "velocity": velocity,
            "lost_count": self.lost_count,
            "status": self.last_status,
            "active": self.active,
        }