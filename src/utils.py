import os
from typing import Any, Dict, List, Optional, Tuple ,Sequence

import cv2
import yaml

import csv


BBoxXYXY = Tuple[float, float, float, float]
BBoxCXCYWH = Tuple[float, float, float, float]


def load_config(config_path: str) -> Dict[str, Any]:
    """
    YAML config dosyasını okur.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config dosyası bulunamadı: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config dosyası boş: {config_path}")

    return config


def ensure_dir(path: str) -> None:
    """
    Klasör yoksa oluşturur.
    """
    if path:
        os.makedirs(path, exist_ok=True)


def xyxy_to_cxcywh(bbox_xyxy: BBoxXYXY) -> BBoxCXCYWH:
    """
    YOLO bbox formatı:
        [x1, y1, x2, y2]

    Kalman ölçüm formatı:
        [cx, cy, w, h]
    """
    x1, y1, x2, y2 = bbox_xyxy

    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    return cx, cy, w, h


def cxcywh_to_xyxy(bbox_cxcywh: BBoxCXCYWH) -> BBoxXYXY:
    """
    Kalman state/measurement formatı:
        [cx, cy, w, h]

    Çizim için:
        [x1, y1, x2, y2]
    """
    cx, cy, w, h = bbox_cxcywh

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    return x1, y1, x2, y2


def clip_bbox_xyxy(
    bbox_xyxy: BBoxXYXY,
    frame_width: int,
    frame_height: int
) -> BBoxXYXY:
    """
    Bbox koordinatlarını görüntü sınırları içinde tutar.
    """
    x1, y1, x2, y2 = bbox_xyxy

    x1 = max(0, min(frame_width - 1, x1))
    y1 = max(0, min(frame_height - 1, y1))
    x2 = max(0, min(frame_width - 1, x2))
    y2 = max(0, min(frame_height - 1, y2))

    return x1, y1, x2, y2


def draw_bbox_xyxy(
    frame,
    bbox_xyxy: BBoxXYXY,
    color: Tuple[int, int, int],
    label: Optional[str] = None,
    thickness: int = 2
):
    """
    Frame üstüne bbox çizer.
    OpenCV BGR renk formatı kullanır.
    """
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = clip_bbox_xyxy(bbox_xyxy, frame_width, frame_height)

    x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])

    cv2.rectangle(
        frame,
        (x1_i, y1_i),
        (x2_i, y2_i),
        color,
        thickness
    )

    if label is not None:
        cv2.putText(
            frame,
            label,
            (x1_i, max(20, y1_i - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )

    return frame


def draw_center(
    frame,
    cx: float,
    cy: float,
    color: Tuple[int, int, int],
    radius: int = 4
):
    """
    Bbox merkez noktasını çizer.
    """
    cv2.circle(
        frame,
        (int(cx), int(cy)),
        radius,
        color,
        -1
    )

    return frame


def select_best_detection(
    detections: List[Dict[str, Any]],
    strategy: str = "highest_confidence"
) -> Optional[Dict[str, Any]]:
    """
    Tek insan takibi için detection listesinden bir bbox seçer.

    İlk aşamada sadece highest_confidence kullanıyoruz.
    """
    if len(detections) == 0:
        return None

    if strategy == "highest_confidence":
        return max(detections, key=lambda det: det["conf"])

    raise ValueError(f"Bilinmeyen detection seçme stratejisi: {strategy}")


def put_status_text(
    frame,
    text: str,
    position: Tuple[int, int] = (20, 40),
    color: Tuple[int, int, int] = (255, 255, 255)
):
    """
    Frame üzerine durum yazısı yazar.
    """
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA
    )

    return frame


def resize_for_display(frame, max_width: int = 1280, max_height: int = 720):
    """
    Sadece ekranda göstermek için frame'i oranı bozmadan küçültür.
    Output video kaydını etkilemez.
    """
    height, width = frame.shape[:2]

    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)

    if scale == 1.0:
        return frame

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    return resized



def draw_polyline(
    frame,
    points: Sequence[Tuple[float, float]],
    color: Tuple[int, int, int],
    thickness: int = 2,
):
    """
    Merkez noktalarının geçmişini çizgi olarak çizer.
    """
    if len(points) < 2:
        return frame

    for i in range(1, len(points)):
        p1 = points[i - 1]
        p2 = points[i]

        if p1 is None or p2 is None:
            continue

        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
            cv2.LINE_AA,
        )

    return frame


def draw_legend(
    frame,
    items,
    top_right_margin: int = 20,
    box_width: int = 330,
    row_height: int = 28,
):
    """
    Sağ üstte renk açıklama kutusu çizer.

    items:
        [
            ("YOLO bbox", (0, 0, 255)),
            ("Kalman bbox", (0, 255, 0)),
        ]
    """
    frame_h, frame_w = frame.shape[:2]

    box_height = 20 + len(items) * row_height
    x1 = frame_w - box_width - top_right_margin
    y1 = top_right_margin
    x2 = frame_w - top_right_margin
    y2 = y1 + box_height

    # Arka plan
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (30, 30, 30),
        -1,
    )

    alpha = 0.65
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Kenarlık
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (220, 220, 220),
        1,
    )

    title_y = y1 + 22
    cv2.putText(
        frame,
        "Legend",
        (x1 + 12, title_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for idx, (label, color) in enumerate(items):
        y = y1 + 45 + idx * row_height

        cv2.rectangle(
            frame,
            (x1 + 14, y - 14),
            (x1 + 34, y + 4),
            color,
            -1,
        )

        cv2.putText(
            frame,
            label,
            (x1 + 45, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


def init_csv_log(log_path: str) -> None:
    """
    Log CSV dosyasını başlık satırıyla oluşturur.
    """
    log_dir = os.path.dirname(log_path)
    ensure_dir(log_dir)

    headers = [
        "frame_idx",
        "yolo_detected",
        "yolo_conf",
        "raw_cx",
        "raw_cy",
        "raw_w",
        "raw_h",
        "kalman_active",
        "track_id",
        "kalman_status",
        "lost_count",
        "kalman_cx",
        "kalman_cy",
        "kalman_w",
        "kalman_h",
        "vx",
        "vy",
        "vw",
        "vh",
    ]

    with open(log_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)


def append_csv_log(log_path: str, row: dict) -> None:
    """
    Her frame için CSV log satırı ekler.
    """
    headers = [
        "frame_idx",
        "yolo_detected",
        "yolo_conf",
        "raw_cx",
        "raw_cy",
        "raw_w",
        "raw_h",
        "kalman_active",
        "track_id",
        "kalman_status",
        "lost_count",
        "kalman_cx",
        "kalman_cy",
        "kalman_w",
        "kalman_h",
        "vx",
        "vy",
        "vw",
        "vh",
    ]

    with open(log_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writerow(row)