from typing import Optional, Tuple

import numpy as np


BBoxMeasurement = Tuple[float, float, float, float]


class KalmanFilterBBox:
    """
    2D bbox Kalman Filter.

    State:
        x = [cx, cy, w, h, vx, vy, vw, vh]^T

    Measurement:
        z = [cx, cy, w, h]^T

    YOLO bize cx, cy, w, h verir.
    Kalman ise hızları da içeride tahmin eder:
        vx, vy, vw, vh
    """

    def __init__(
        self,
        dt: float = 1.0,
        initial_p: float = 10.0,
        accel_noise_pos: float = 0.1,
        accel_noise_size: float = 0.05,
        measurement_noise_pos: float = 64.0,
        measurement_noise_size: float = 36.0,
    ):
        self.dt = dt

        self.x = np.zeros((8, 1), dtype=float)

        # State transition matrix F
        # cx_new = cx + vx * dt
        # cy_new = cy + vy * dt
        # w_new  = w  + vw * dt
        # h_new  = h  + vh * dt
        self.F = np.array(
            [
                [1, 0, 0, 0, dt, 0, 0, 0],
                [0, 1, 0, 0, 0, dt, 0, 0],
                [0, 0, 1, 0, 0, 0, dt, 0],
                [0, 0, 0, 1, 0, 0, 0, dt],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )

        # Measurement matrix H
        # State içinden sadece cx, cy, w, h ölçülüyor.
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=float,
        )

        self.P = np.eye(8, dtype=float) * initial_p

        self.Q = self._build_process_noise(
            dt=dt,
            accel_noise_pos=accel_noise_pos,
            accel_noise_size=accel_noise_size,
        )

        self.R = np.array(
            [
                [measurement_noise_pos, 0, 0, 0],
                [0, measurement_noise_pos, 0, 0],
                [0, 0, measurement_noise_size, 0],
                [0, 0, 0, measurement_noise_size],
            ],
            dtype=float,
        )

        self.I = np.eye(8, dtype=float)
        self.initialized = False

    @staticmethod
    def _constant_velocity_q(dt: float, accel_noise: float) -> np.ndarray:
        """
        1D sabit hız modeli için Q bloğu.

        State parçası:
            [position, velocity]
        """
        return accel_noise**2 * np.array(
            [
                [dt**4 / 4, dt**3 / 2],
                [dt**3 / 2, dt**2],
            ],
            dtype=float,
        )

    def _build_process_noise(
        self,
        dt: float,
        accel_noise_pos: float,
        accel_noise_size: float,
    ) -> np.ndarray:
        """
        8x8 Q matrisi oluşturur.

        Ayrı bloklar:
            cx-vx
            cy-vy
            w-vw
            h-vh
        """
        Q = np.zeros((8, 8), dtype=float)

        q_pos = self._constant_velocity_q(dt, accel_noise_pos)
        q_size = self._constant_velocity_q(dt, accel_noise_size)

        # cx, vx
        Q[np.ix_([0, 4], [0, 4])] = q_pos

        # cy, vy
        Q[np.ix_([1, 5], [1, 5])] = q_pos

        # w, vw
        Q[np.ix_([2, 6], [2, 6])] = q_size

        # h, vh
        Q[np.ix_([3, 7], [3, 7])] = q_size

        return Q

    def initialize(self, measurement: BBoxMeasurement) -> np.ndarray:
        """
        İlk YOLO bbox ölçümü ile filtreyi başlatır.

        measurement:
            (cx, cy, w, h)
        """
        cx, cy, w, h = measurement

        self.x = np.array(
            [
                [cx],
                [cy],
                [max(1.0, w)],
                [max(1.0, h)],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
            ],
            dtype=float,
        )

        self.initialized = True
        return self.x

    def predict(self) -> np.ndarray:
        """
        Ölçüm gelmeden önce bbox'ın yeni durumunu tahmin eder.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self._clamp_size()
        return self.x

    def update(self, measurement: BBoxMeasurement) -> np.ndarray:
        """
        YOLO bbox ölçümüyle update yapar.

        measurement:
            (cx, cy, w, h)
        """
        cx, cy, w, h = measurement

        z = np.array(
            [
                [cx],
                [cy],
                [max(1.0, w)],
                [max(1.0, h)],
            ],
            dtype=float,
        )

        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        self.P = (self.I - K @ self.H) @ self.P

        self._clamp_size()
        return self.x

    def step(self, measurement: Optional[BBoxMeasurement]) -> Optional[np.ndarray]:
        """
        measurement:
            None veya (cx, cy, w, h)

        Ölçüm varsa:
            predict + update

        Ölçüm yoksa:
            sadece predict
        """
        if not self.initialized:
            if measurement is None:
                return None

            return self.initialize(measurement)

        self.predict()

        if measurement is not None:
            self.update(measurement)

        return self.x

    def _clamp_size(self) -> None:
        """
        Bbox genişlik/yükseklik negatif veya sıfır olmasın.
        """
        self.x[2, 0] = max(1.0, self.x[2, 0])
        self.x[3, 0] = max(1.0, self.x[3, 0])

    def get_bbox_cxcywh(self) -> BBoxMeasurement:
        """
        Güncel filtrelenmiş bbox'ı döndürür.
        """
        return (
            float(self.x[0, 0]),
            float(self.x[1, 0]),
            float(self.x[2, 0]),
            float(self.x[3, 0]),
        )

    def get_velocity(self) -> Tuple[float, float, float, float]:
        """
        Güncel hız tahminlerini döndürür.
        """
        return (
            float(self.x[4, 0]),
            float(self.x[5, 0]),
            float(self.x[6, 0]),
            float(self.x[7, 0]),
        )