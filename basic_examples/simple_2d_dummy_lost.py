import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter2D:
    def __init__(self, dt=1.0):
        self.dt = dt

        # State:
        # x = [cx, cy, vx, vy]^T
        self.x = np.zeros((4, 1))

        # Hareket modeli:
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

        # Ölçüm modeli:
        # YOLO sadece cx, cy ölçüyor.
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        # Başlangıç belirsizliği
        self.P = np.eye(4) * 10.0

        # Hareket modeli gürültüsü
        # Küçük seçersek sabit hız modeline daha çok güvenir.
        accel_noise = 0.1
        self.Q = accel_noise ** 2 * np.array([
            [dt**4 / 4, 0,         dt**3 / 2, 0],
            [0,         dt**4 / 4, 0,         dt**3 / 2],
            [dt**3 / 2, 0,         dt**2,     0],
            [0,         dt**3 / 2, 0,         dt**2]
        ], dtype=float)

        # Ölçüm gürültüsü
        # Dummy YOLO noise std=8 olduğu için varyans 8^2=64
        self.R = np.array([
            [300, 0],
            [0, 300]
        ], dtype=float)

        self.I = np.eye(4)
        self.initialized = False

    def initialize(self, cx, cy):
        """
        Yeni track başlatıldığında Kalman başlangıcı yapılır.
        İlk anda hızı bilmiyoruz, 0 kabul ediyoruz.
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
        Ölçüm olsa da olmasa da önce predict yapılır.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, cx_meas, cy_meas):
        """
        YOLO ölçümü geldiyse Kalman update yapılır.
        """
        z = np.array([
            [cx_meas],
            [cy_meas]
        ], dtype=float)

        # Innovation / residual:
        # YOLO ölçümü - Kalman'ın beklediği ölçüm
        y = z - self.H @ self.x

        # Ölçüm tarafındaki toplam belirsizlik
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        self.P = (self.I - K @ self.H) @ self.P

        return self.x

    def step(self, measurement=None):
        """
        measurement:
            None      -> YOLO hedefi bulamadı.
            (cx, cy)  -> YOLO hedef merkezi verdi.
        """
        if not self.initialized:
            if measurement is None:
                return None

            cx, cy = measurement
            self.initialize(cx, cy)
            return self.x

        # Her durumda önce predict
        self.predict()

        # Ölçüm varsa update
        if measurement is not None:
            cx, cy = measurement
            self.update(cx, cy)

        return self.x


class SimpleTrack:
    def __init__(self, max_lost_count=10):
        self.max_lost_count = max_lost_count

        self.kf = None
        self.active = False

        self.lost_count = 0
        self.track_id = 0

        # Grafik ve analiz için geçmiş kayıtlar
        self.history = []
        self.prediction_only_flags = []
        self.track_ids = []
        self.status_history = []

    def start_new_track(self, measurement):
        """
        Track pasifken yeni ölçüm gelirse yeni track başlatılır.
        """
        cx, cy = measurement

        self.track_id += 1
        self.kf = KalmanFilter2D(dt=1.0)
        self.kf.initialize(cx, cy)

        self.active = True
        self.lost_count = 0

        state = self.kf.x

        cx = state[0, 0]
        cy = state[1, 0]
        vx = state[2, 0]
        vy = state[3, 0]

        self.history.append((cx, cy, vx, vy))
        self.prediction_only_flags.append(False)
        self.track_ids.append(self.track_id)
        self.status_history.append("new_track")

        return cx, cy, vx, vy

    def deactivate_track(self):
        """
        Track çok uzun süre ölçüm alamazsa pasif yapılır.
        """
        self.active = False
        self.kf = None
        self.lost_count = 0

    def step(self, measurement):
        """
        Ana tracker adımı.

        Durumlar:
        1. Track pasif + ölçüm yok  -> bekle
        2. Track pasif + ölçüm var  -> yeni track başlat
        3. Track aktif + ölçüm var  -> predict + update
        4. Track aktif + ölçüm yok  -> sadece predict
        5. Track aktif + uzun kayıp -> pasif yap
        """

        # Track pasifken
        if not self.active:
            if measurement is None:
                self.history.append(None)
                self.prediction_only_flags.append(False)
                self.track_ids.append(None)
                self.status_history.append("inactive_waiting")
                return None

            return self.start_new_track(measurement)

        # Track aktifken
        if measurement is None:
            self.lost_count += 1
            prediction_only = True
        else:
            self.lost_count = 0
            prediction_only = False

        # Önce Kalman adımı yapalım.
        # Böylece kayıp frame'lerde max_lost_count'a kadar predict görebiliriz.
        state = self.kf.step(measurement)

        if state is None:
            self.history.append(None)
            self.prediction_only_flags.append(False)
            self.track_ids.append(None)
            self.status_history.append("not_initialized")
            return None

        cx = state[0, 0]
        cy = state[1, 0]
        vx = state[2, 0]
        vy = state[3, 0]

        # Eğer kayıp sayısı sınırı geçtiyse bu frame'den sonra track'i pasif yap.
        if self.lost_count > self.max_lost_count:
            self.history.append(None)
            self.prediction_only_flags.append(False)
            self.track_ids.append(None)
            self.status_history.append("deactivated")

            self.deactivate_track()
            return None

        self.history.append((cx, cy, vx, vy))
        self.prediction_only_flags.append(prediction_only)
        self.track_ids.append(self.track_id)

        if measurement is None:
            self.status_history.append("predict_only")
        else:
            self.status_history.append("update")

        return cx, cy, vx, vy


def generate_dummy_yolo_data(num_frames=100):
    """
    Sahte gerçek hedef yolu ve YOLO ölçümleri üretir.

    Bazı frame'lerde YOLO ölçümü None yapılır.
    Bu sayede:
    - track aktif olur
    - pasif olur
    - tekrar aktif olur
    - tekrar pasif olur
    - tekrar aktif olup devam eder
    """

    true_positions = []
    measurements = []
    missing_flags = []

    np.random.seed(42)

    cx = 100.0
    cy = 100.0

    vx = 4.0
    vy = 2.0

    for frame_idx in range(num_frames):
        # Gerçek hedef hareketi
        cx += vx
        cy += vy

        true_positions.append((cx, cy))

        # YOLO ölçüm gürültüsü
        noise_x = np.random.normal(0, 8)
        noise_y = np.random.normal(0, 8)

        measured_cx = cx + noise_x
        measured_cy = cy + noise_y

        # Kayıp aralıkları
        # max_lost_count=10 olduğu için bu aralıklar track'i pasif yapacak kadar uzun.
        #
        # 0-19:   YOLO var -> Track ID 1 aktif
        # 20-33:  YOLO yok -> Track ID 1 predict, sonra pasif
        # 34-49:  YOLO var -> Track ID 2 aktif
        # 50-63:  YOLO yok -> Track ID 2 predict, sonra pasif
        # 64-99:  YOLO var -> Track ID 3 aktif ve devam
        is_missing = (
            (20 <= frame_idx <= 33) or
            (50 <= frame_idx <= 63)
        )

        if is_missing:
            measurements.append(None)
            missing_flags.append(True)
        else:
            measurements.append((measured_cx, measured_cy))
            missing_flags.append(False)

    return true_positions, measurements, missing_flags


def main():
    true_positions, measurements, missing_flags = generate_dummy_yolo_data(
        num_frames=100
    )

    track = SimpleTrack(max_lost_count=10)

    for frame_idx, measurement in enumerate(measurements):
        result = track.step(measurement)

        if result is None:
            print(
                f"Frame {frame_idx:02d}: "
                f"TRACK YOK / PASIF | "
                f"active={track.active}"
            )
        else:
            cx, cy, vx, vy = result

            if measurement is None:
                status = "YOLO YOK -> sadece predict"
            else:
                status = "YOLO VAR -> predict + update"

            print(
                f"Frame {frame_idx:02d}: "
                f"Track ID={track.track_id} | "
                f"{status} | "
                f"cx={cx:.2f}, cy={cy:.2f}, "
                f"vx={vx:.2f}, vy={vy:.2f}, "
                f"lost_count={track.lost_count}"
            )

    true_positions = np.array(true_positions)

    measured_x = []
    measured_y = []

    missing_true_x = []
    missing_true_y = []

    filtered_x = []
    filtered_y = []

    predict_only_x = []
    predict_only_y = []

    inactive_x = []
    inactive_y = []

    deactivated_x = []
    deactivated_y = []

    new_track_x = []
    new_track_y = []

    # YOLO ölçüm noktaları
    for idx, measurement in enumerate(measurements):
        if measurement is None:
            missing_true_x.append(true_positions[idx, 0])
            missing_true_y.append(true_positions[idx, 1])
        else:
            measured_x.append(measurement[0])
            measured_y.append(measurement[1])

    # Kalman geçmişi
    for idx, item in enumerate(track.history):
        status = track.status_history[idx]

        if item is None:
            inactive_x.append(true_positions[idx, 0])
            inactive_y.append(true_positions[idx, 1])

            if status == "deactivated":
                deactivated_x.append(true_positions[idx, 0])
                deactivated_y.append(true_positions[idx, 1])

            continue

        cx, cy, vx, vy = item

        filtered_x.append(cx)
        filtered_y.append(cy)

        if track.prediction_only_flags[idx]:
            predict_only_x.append(cx)
            predict_only_y.append(cy)

        if status == "new_track":
            new_track_x.append(cx)
            new_track_y.append(cy)

    plt.figure(figsize=(12, 8))

    # Gerçek yol
    plt.plot(
        true_positions[:, 0],
        true_positions[:, 1],
        label="Gerçek hedef yolu",
        linewidth=3
    )

    # YOLO ölçümleri
    plt.scatter(
        measured_x,
        measured_y,
        label="YOLO ölçümleri",
        alpha=0.45
    )

    # YOLO'nun kaçırdığı gerçek pozisyonlar
    plt.scatter(
        missing_true_x,
        missing_true_y,
        marker="x",
        s=80,
        label="YOLO ölçüm yok"
    )

    # Kalman yolu
    plt.plot(
        filtered_x,
        filtered_y,
        label="Kalman takip yolu",
        linewidth=3
    )

    # Sadece predict yapılan noktalar
    plt.scatter(
        predict_only_x,
        predict_only_y,
        marker="^",
        s=90,
        label="Sadece predict"
    )

    # Yeni track başlangıçları
    plt.scatter(
        new_track_x,
        new_track_y,
        marker="o",
        s=160,
        facecolors="none",
        edgecolors="black",
        label="Yeni track başladı"
    )

    # Pasif / bekleme noktaları
    if inactive_x:
        plt.scatter(
            inactive_x,
            inactive_y,
            marker="s",
            s=55,
            label="Track pasif / bekliyor"
        )

    # Deaktif olduğu anlar
    if deactivated_x:
        plt.scatter(
            deactivated_x,
            deactivated_y,
            marker="D",
            s=130,
            label="Track pasif oldu"
        )

    plt.xlabel("cx")
    plt.ylabel("cy")
    plt.title("2D Kalman Tracker - Pasif Olup Tekrar Aktif Olan Track")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()