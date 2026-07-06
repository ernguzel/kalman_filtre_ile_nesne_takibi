import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import yaml


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