import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter2D:
    def __init__(self, dt=1.0):
        self.dt = dt

        # State vector:
        # x = [cx, cy, vx, vy]^T
        self.x = np.zeros((4, 1))

        # State transition matrix
        # cx_new = cx + vx * dt
        # cy_new = cy + vy * dt
        # vx_new = vx
        # vy_new = vy
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

        # Measurement matrix
        # YOLO bize sadece cx, cy ölçüyor.
        # vx, vy doğrudan ölçülmüyor.
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        # Initial uncertainty
        # Başta konum ve hızdan çok emin değiliz.
        self.P = np.eye(4) * 10.0

        # Process noise
        # Hareket modeline ne kadar güvenmediğimizi belirtir.
        self.Q = np.array([
            [0.1, 0,   0,   0],
            [0,   0.1, 0,   0],
            [0,   0,   0.1, 0],
            [0,   0,   0,   0.1]
        ], dtype=float)

        # self.R = np.array([
        #     [1, 0],
        #     [0, 1]
        # ], dtype=float)
        # Measurement noise
        # YOLO ölçümünün ne kadar gürültülü olduğunu belirtir.
        self.R = np.array([
            [500, 0],
            [0, 500]
        ], dtype=float)

        self.I = np.eye(4)

        self.initialized = False

    def initialize(self, cx, cy):
        """
        İlk YOLO ölçümü gelince filtreyi başlatıyoruz.
        Başta hızı bilmiyoruz, 0 veriyoruz.
        """
        self.x = np.array([
            [cx],
            [cy],
            [0.0],
            [0.0]
        ], dtype=float)

        self.initialized = True

    def predict(self):
        """
        Ölçüm gelmeden önce hedefin nerede olacağını tahmin eder.
        
        """

        print(f"predicttin içinde ilk p {self.P}")
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        print(f"predicttin içinde haat eklenen p {self.P}")
        
        return self.x

    def update(self, cx_meas, cy_meas):
        """
        YOLO'dan gelen gürültülü cx, cy ölçümüyle tahmini düzeltir.
        """
        z = np.array([
            [cx_meas],
            [cy_meas]
        ], dtype=float)

        # Innovation / residual
        # Ölçüm ile tahmin arasındaki fark
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        print(f"KALMAN KAZANCİ {K}")
        # State update
        self.x = self.x + K @ y

        # Covariance update
        self.P = (self.I - K @ self.H) @ self.P
        print(f"UPDATEDA SON  p {self.P}")
        return self.x

    def step(self, measurement=None):
        """
        measurement:
            None olabilir. Bu durumda YOLO hedefi kaçırmış gibi davranırız.
            Veya (cx, cy) olabilir.
        """
        if measurement is not None and not self.initialized:
            cx, cy = measurement
            self.initialize(cx, cy)
            return self.x

        if not self.initialized:
            return None

        self.predict()

        if measurement is not None:
            cx, cy = measurement
            self.update(cx, cy)

        return self.x


def generate_dummy_yolo_data(num_frames=80):
    """
    Sahte YOLO verisi üretir.

    true_positions:
        Gerçek hedef merkezi.

    noisy_measurements:
        YOLO'dan gelmiş gibi gürültülü ölçümler.
    """

    true_positions = []
    noisy_measurements = []

    np.random.seed(42)

    # Gerçek başlangıç konumu
    cx = 100.0
    cy = 100.0

    # Gerçek hız
    vx = 4.0
    vy = 2.0

    for frame_idx in range(num_frames):
        # Gerçek hareket
        cx = cx + vx
        cy = cy + vy

        true_positions.append((cx, cy))

        # YOLO ölçüm gürültüsü
        noise_x = np.random.normal(0, 8)
        noise_y = np.random.normal(0, 8)

        measured_cx = cx + noise_x
        measured_cy = cy + noise_y

        noisy_measurements.append((measured_cx, measured_cy))

    return true_positions, noisy_measurements


def main():
    true_positions, noisy_measurements = generate_dummy_yolo_data(num_frames=80)

    kf = KalmanFilter2D(dt=1.0)

    filtered_positions = []
    estimated_velocities = []

    for measurement in noisy_measurements:
        state = kf.step(measurement)
        

        filtered_cx = state[0, 0]
        filtered_cy = state[1, 0]
        estimated_vx = state[2, 0]
        estimated_vy = state[3, 0]

        filtered_positions.append((filtered_cx, filtered_cy))
        estimated_velocities.append((estimated_vx, estimated_vy))

    true_positions = np.array(true_positions)
    noisy_measurements = np.array(noisy_measurements)
    filtered_positions = np.array(filtered_positions)
    estimated_velocities = np.array(estimated_velocities)

    print("Son tahmin:")
    print(f"cx: {filtered_positions[-1, 0]:.2f}")
    print(f"cy: {filtered_positions[-1, 1]:.2f}")
    print(f"vx: {estimated_velocities[-1, 0]:.2f}")
    print(f"vy: {estimated_velocities[-1, 1]:.2f}")

    # Grafik çiz
    plt.figure(figsize=(10, 7))

    plt.plot(
        true_positions[:, 0],
        true_positions[:, 1],
        label="Gerçek hedef yolu",
        linewidth=3
    )

    plt.scatter(
        noisy_measurements[:, 0],
        noisy_measurements[:, 1],
        label="Gürültülü YOLO ölçümleri",
        alpha=0.5
    )

    plt.plot(
        filtered_positions[:, 0],
        filtered_positions[:, 1],
        label="Kalman filtrelenmiş yol",
        linewidth=3
    )

    plt.xlabel("cx")
    plt.ylabel("cy")
    plt.title("2D Kalman Filter ile Sahte YOLO Merkez Takibi")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()