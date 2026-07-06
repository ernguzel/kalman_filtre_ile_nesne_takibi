import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class KalmanFilterBBox:
    def __init__(self, dt=1.0):
        self.dt = dt

        # State:
        # x = [cx, cy, w, h, vx, vy, vw, vh]^T
        self.x = np.zeros((8, 1))

        # Hareket modeli:
        # cx_new = cx + vx * dt
        # cy_new = cy + vy * dt
        # w_new  = w  + vw * dt
        # h_new  = h  + vh * dt
        # vx_new = vx
        # vy_new = vy
        # vw_new = vw
        # vh_new = vh
        self.F = np.array([
            [1, 0, 0, 0, dt, 0,  0,  0],
            [0, 1, 0, 0, 0,  dt, 0,  0],
            [0, 0, 1, 0, 0,  0,  dt, 0],
            [0, 0, 0, 1, 0,  0,  0,  dt],
            [0, 0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0, 0,  0,  1,  0],
            [0, 0, 0, 0, 0,  0,  0,  1],
        ], dtype=float)

        # Ölçüm modeli:
        # YOLO bize sadece cx, cy, w, h verir.
        # vx, vy, vw, vh ölçülmez.
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=float)

        # Başlangıç belirsizliği
        self.P = np.eye(8) * 10.0

        # Hareket modeli gürültüsü
        # Konum ve bbox boyutu için ayrı ayrı sabit hız modeli kullanıyoruz.
        accel_noise_pos = 0.2
        accel_noise_size = 0.1

        self.Q = np.zeros((8, 8), dtype=float)

        # cx, vx bloğu
        self.Q[np.ix_([0, 4], [0, 4])] = accel_noise_pos ** 2 * np.array([
            [dt**4 / 4, dt**3 / 2],
            [dt**3 / 2, dt**2]
        ])

        # cy, vy bloğu
        self.Q[np.ix_([1, 5], [1, 5])] = accel_noise_pos ** 2 * np.array([
            [dt**4 / 4, dt**3 / 2],
            [dt**3 / 2, dt**2]
        ])

        # w, vw bloğu
        self.Q[np.ix_([2, 6], [2, 6])] = accel_noise_size ** 2 * np.array([
            [dt**4 / 4, dt**3 / 2],
            [dt**3 / 2, dt**2]
        ])

        # h, vh bloğu
        self.Q[np.ix_([3, 7], [3, 7])] = accel_noise_size ** 2 * np.array([
            [dt**4 / 4, dt**3 / 2],
            [dt**3 / 2, dt**2]
        ])

        # Ölçüm gürültüsü
        # YOLO merkez ölçümü genelde daha oynak olabilir.
        # Bbox boyutu da gürültülü olabilir ama burada biraz daha düşük verdik.
        self.R = np.array([
            [300, 0,  0,  0],
            [0,  300, 0,  0],
            [0,  0,  300, 0],
            [0,  0,  0,  300],
        ], dtype=float)

        self.I = np.eye(8)
        self.initialized = False

    def initialize(self, cx, cy, w, h):
        """
        İlk bbox ölçümü ile filtreyi başlatır.
        Hızları başlangıçta 0 kabul eder.
        """
        self.x = np.array([
            [cx],
            [cy],
            [w],
            [h],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
        ], dtype=float)

        self.initialized = True

    def predict(self):
        """
        Ölçüm olmasa bile hedef bbox'ının bir sonraki halini tahmin eder.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Genişlik ve yükseklik negatif olmasın.
        self.x[2, 0] = max(1.0, self.x[2, 0])
        self.x[3, 0] = max(1.0, self.x[3, 0])

        return self.x

    def update(self, cx_meas, cy_meas, w_meas, h_meas):
        """
        YOLO'dan gelen bbox ölçümüyle update yapar.
        """
        z = np.array([
            [cx_meas],
            [cy_meas],
            [w_meas],
            [h_meas],
        ], dtype=float)

        # Innovation / residual
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        self.P = (self.I - K @ self.H) @ self.P

        # Bbox boyutu negatif olmasın.
        self.x[2, 0] = max(1.0, self.x[2, 0])
        self.x[3, 0] = max(1.0, self.x[3, 0])

        return self.x

    def step(self, measurement=None):
        """
        measurement:
            None veya (cx, cy, w, h)
        """
        if not self.initialized:
            if measurement is None:
                return None

            cx, cy, w, h = measurement
            self.initialize(cx, cy, w, h)
            return self.x

        self.predict()

        if measurement is not None:
            cx, cy, w, h = measurement
            self.update(cx, cy, w, h)

        return self.x


class SimpleBBoxTrack:
    def __init__(self, max_lost_count=10):
        self.max_lost_count = max_lost_count

        self.kf = None
        self.active = False

        self.lost_count = 0
        self.track_id = 0

        self.history = []
        self.prediction_only_flags = []
        self.track_ids = []
        self.status_history = []

    def start_new_track(self, measurement):
        cx, cy, w, h = measurement

        self.track_id += 1
        self.kf = KalmanFilterBBox(dt=1.0)
        self.kf.initialize(cx, cy, w, h)

        self.active = True
        self.lost_count = 0

        state = self.kf.x
        result = self.state_to_tuple(state)

        self.history.append(result)
        self.prediction_only_flags.append(False)
        self.track_ids.append(self.track_id)
        self.status_history.append("new_track")

        return result

    def deactivate_track(self):
        self.active = False
        self.kf = None
        self.lost_count = 0

    @staticmethod
    def state_to_tuple(state):
        return (
            state[0, 0],  # cx
            state[1, 0],  # cy
            state[2, 0],  # w
            state[3, 0],  # h
            state[4, 0],  # vx
            state[5, 0],  # vy
            state[6, 0],  # vw
            state[7, 0],  # vh
        )

    def step(self, measurement):
        """
        measurement:
            None veya (cx, cy, w, h)
        """

        # Track pasifken:
        # Ölçüm yoksa bekle, ölçüm varsa yeni track başlat.
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

        state = self.kf.step(measurement)

        if state is None:
            self.history.append(None)
            self.prediction_only_flags.append(False)
            self.track_ids.append(None)
            self.status_history.append("not_initialized")
            return None

        result = self.state_to_tuple(state)

        # Çok uzun kayıpsa track'i pasif yap.
        if self.lost_count > self.max_lost_count:
            self.history.append(None)
            self.prediction_only_flags.append(False)
            self.track_ids.append(None)
            self.status_history.append("deactivated")

            self.deactivate_track()
            return None

        self.history.append(result)
        self.prediction_only_flags.append(prediction_only)
        self.track_ids.append(self.track_id)

        if measurement is None:
            self.status_history.append("predict_only")
        else:
            self.status_history.append("update")

        return result


def bbox_center_to_xyxy(cx, cy, w, h):
    """
    cx, cy, w, h formatını x1, y1, x2, y2 formatına çevirir.
    """
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return x1, y1, x2, y2


def generate_dummy_yolo_bbox_data(num_frames=100):
    """
    Sahte gerçek bbox yolu ve YOLO bbox ölçümleri üretir.

    Gerçek hedef:
    - Sağ aşağı hareket ediyor.
    - Zamanla biraz büyüyor.
    """

    true_bboxes = []
    measurements = []
    missing_flags = []

    np.random.seed(42)

    # Gerçek başlangıç bbox
    cx = 100.0
    cy = 100.0
    w = 50.0
    h = 80.0

    # Gerçek hızlar
    vx = 4.0
    vy = 2.0
    vw = 0.35
    vh = 0.20

    for frame_idx in range(num_frames):
        # Gerçek bbox hareketi
        cx += vx
        cy += vy
        w += vw
        h += vh

        true_bboxes.append((cx, cy, w, h))

        # YOLO ölçüm gürültüsü
        noise_cx = np.random.normal(0, 8)
        noise_cy = np.random.normal(0, 8)
        noise_w = np.random.normal(0, 6)
        noise_h = np.random.normal(0, 6)

        measured_cx = cx + noise_cx
        measured_cy = cy + noise_cy
        measured_w = max(1.0, w + noise_w)
        measured_h = max(1.0, h + noise_h)

        # Ölçüm kaybı aralıkları
        is_missing = (
            (20 <= frame_idx <= 33) or
            (50 <= frame_idx <= 63)
        )

        if is_missing:
            measurements.append(None)
            missing_flags.append(True)
        else:
            measurements.append((measured_cx, measured_cy, measured_w, measured_h))
            missing_flags.append(False)

    return true_bboxes, measurements, missing_flags


def draw_bbox(ax, bbox, edge_label=None, linewidth=2, linestyle="-", alpha=1.0):
    """
    bbox: (cx, cy, w, h)
    """
    cx, cy, w, h = bbox
    x1, y1, x2, y2 = bbox_center_to_xyxy(cx, cy, w, h)

    rect = Rectangle(
        (x1, y1),
        w,
        h,
        fill=False,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        label=edge_label
    )

    ax.add_patch(rect)


def main():
    true_bboxes, measurements, missing_flags = generate_dummy_yolo_bbox_data(
        num_frames=100
    )

    track = SimpleBBoxTrack(max_lost_count=10)

    for frame_idx, measurement in enumerate(measurements):
        result = track.step(measurement)

        if result is None:
            print(
                f"Frame {frame_idx:02d}: "
                f"TRACK YOK / PASIF | "
                f"active={track.active}"
            )
        else:
            cx, cy, w, h, vx, vy, vw, vh = result

            if measurement is None:
                status = "YOLO YOK -> sadece predict"
            else:
                status = "YOLO VAR -> predict + update"

            print(
                f"Frame {frame_idx:02d}: "
                f"Track ID={track.track_id} | "
                f"{status} | "
                f"cx={cx:.2f}, cy={cy:.2f}, "
                f"w={w:.2f}, h={h:.2f}, "
                f"vx={vx:.2f}, vy={vy:.2f}, "
                f"vw={vw:.2f}, vh={vh:.2f}, "
                f"lost_count={track.lost_count}"
            )

    true_bboxes_np = np.array(true_bboxes)

    measured_centers_x = []
    measured_centers_y = []

    true_centers_x = true_bboxes_np[:, 0]
    true_centers_y = true_bboxes_np[:, 1]

    filtered_centers_x = []
    filtered_centers_y = []

    predict_only_x = []
    predict_only_y = []

    missing_true_x = []
    missing_true_y = []

    new_track_x = []
    new_track_y = []

    inactive_x = []
    inactive_y = []

    # Ölçüm ve kayıp noktaları
    for idx, measurement in enumerate(measurements):
        if measurement is None:
            missing_true_x.append(true_bboxes[idx][0])
            missing_true_y.append(true_bboxes[idx][1])
        else:
            measured_centers_x.append(measurement[0])
            measured_centers_y.append(measurement[1])

    # Kalman geçmişi
    for idx, item in enumerate(track.history):
        status = track.status_history[idx]

        if item is None:
            inactive_x.append(true_bboxes[idx][0])
            inactive_y.append(true_bboxes[idx][1])
            continue

        cx, cy, w, h, vx, vy, vw, vh = item

        filtered_centers_x.append(cx)
        filtered_centers_y.append(cy)

        if track.prediction_only_flags[idx]:
            predict_only_x.append(cx)
            predict_only_y.append(cy)

        if status == "new_track":
            new_track_x.append(cx)
            new_track_y.append(cy)

    fig, ax = plt.subplots(figsize=(13, 9))

    # Gerçek merkez yolu
    ax.plot(
        true_centers_x,
        true_centers_y,
        label="Gerçek bbox merkez yolu",
        linewidth=3
    )

    # YOLO ölçüm merkezleri
    ax.scatter(
        measured_centers_x,
        measured_centers_y,
        label="YOLO bbox merkez ölçümleri",
        alpha=0.45
    )

    # Kalman merkez yolu
    ax.plot(
        filtered_centers_x,
        filtered_centers_y,
        label="Kalman bbox merkez yolu",
        linewidth=3
    )

    # Ölçüm olmayan gerçek noktalar
    ax.scatter(
        missing_true_x,
        missing_true_y,
        marker="x",
        s=80,
        label="YOLO ölçüm yok"
    )

    # Sadece predict noktaları
    ax.scatter(
        predict_only_x,
        predict_only_y,
        marker="^",
        s=90,
        label="Sadece predict"
    )

    # Yeni track başlangıçları
    ax.scatter(
        new_track_x,
        new_track_y,
        marker="o",
        s=160,
        facecolors="none",
        edgecolors="black",
        label="Yeni track başladı"
    )

    # Pasif bekleme noktaları
    if inactive_x:
        ax.scatter(
            inactive_x,
            inactive_y,
            marker="s",
            s=55,
            label="Track pasif / bekliyor"
        )

    # Birkaç bbox çizelim: gerçek, ölçüm ve Kalman
    selected_frames = [5, 18, 38, 48, 75, 95]

    true_label_used = False
    meas_label_used = False
    kf_label_used = False

    for frame_idx in selected_frames:
        if frame_idx >= len(true_bboxes):
            continue

        # Gerçek bbox
        draw_bbox(
            ax,
            true_bboxes[frame_idx],
            edge_label="Gerçek bbox" if not true_label_used else None,
            linewidth=2,
            linestyle="-",
            alpha=0.8
        )
        true_label_used = True

        # YOLO ölçümü varsa çiz
        if measurements[frame_idx] is not None:
            draw_bbox(
                ax,
                measurements[frame_idx],
                edge_label="YOLO bbox ölçümü" if not meas_label_used else None,
                linewidth=1,
                linestyle="--",
                alpha=0.6
            )
            meas_label_used = True

        # Kalman bbox varsa çiz
        if frame_idx < len(track.history) and track.history[frame_idx] is not None:
            cx, cy, w, h, vx, vy, vw, vh = track.history[frame_idx]
            draw_bbox(
                ax,
                (cx, cy, w, h),
                edge_label="Kalman bbox" if not kf_label_used else None,
                linewidth=3,
                linestyle="-.",
                alpha=0.8
            )
            kf_label_used = True

    ax.set_xlabel("cx")
    ax.set_ylabel("cy")
    ax.set_title("2D Kalman Tracker - Bbox Boyutu Dahil [cx, cy, w, h, vx, vy, vw, vh]")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()