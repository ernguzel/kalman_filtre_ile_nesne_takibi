import argparse
import csv
import math
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def to_float(value) -> Optional[float]:
    """
    CSV'den gelen boş string / None değerlerini güvenli float'a çevirir.
    """
    if value is None:
        return None

    if value == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def to_bool(value) -> bool:
    """
    CSV bool alanlarını güvenli şekilde bool'a çevirir.
    """
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    return str(value).lower() in ["true", "1", "yes"]


def read_log_csv(csv_path: str) -> Dict[str, List]:
    """
    Kalman log CSV dosyasını okur.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV log bulunamadı: {csv_path}")

    data = {
        "frame_idx": [],
        "yolo_detected": [],
        "raw_cx": [],
        "raw_cy": [],
        "kalman_active": [],
        "kalman_cx": [],
        "kalman_cy": [],
        "kalman_status": [],
        "lost_count": [],
        "center_error": [],
    }

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            frame_idx = int(row["frame_idx"])

            raw_cx = to_float(row.get("raw_cx"))
            raw_cy = to_float(row.get("raw_cy"))
            kalman_cx = to_float(row.get("kalman_cx"))
            kalman_cy = to_float(row.get("kalman_cy"))

            yolo_detected = to_bool(row.get("yolo_detected"))
            kalman_active = to_bool(row.get("kalman_active"))

            kalman_status = row.get("kalman_status", "")
            lost_count = to_float(row.get("lost_count"))

            if (
                raw_cx is not None
                and raw_cy is not None
                and kalman_cx is not None
                and kalman_cy is not None
            ):
                error = math.sqrt(
                    (raw_cx - kalman_cx) ** 2
                    + (raw_cy - kalman_cy) ** 2
                )
            else:
                error = None

            data["frame_idx"].append(frame_idx)
            data["yolo_detected"].append(yolo_detected)
            data["raw_cx"].append(raw_cx)
            data["raw_cy"].append(raw_cy)
            data["kalman_active"].append(kalman_active)
            data["kalman_cx"].append(kalman_cx)
            data["kalman_cy"].append(kalman_cy)
            data["kalman_status"].append(kalman_status)
            data["lost_count"].append(lost_count)
            data["center_error"].append(error)

    return data


def filter_valid_xy(x_values, y_values):
    """
    None olmayan x, y çiftlerini döndürür.
    """
    filtered_x = []
    filtered_y = []

    for x, y in zip(x_values, y_values):
        if x is None or y is None:
            continue

        filtered_x.append(x)
        filtered_y.append(y)

    return filtered_x, filtered_y


def filter_valid_frame_value(frame_values, values):
    """
    None olmayan frame-value çiftlerini döndürür.
    """
    filtered_frames = []
    filtered_values = []

    for frame_idx, value in zip(frame_values, values):
        if value is None:
            continue

        filtered_frames.append(frame_idx)
        filtered_values.append(value)

    return filtered_frames, filtered_values


def save_center_path_comparison(data: Dict[str, List], output_dir: str):
    """
    Raw YOLO center yolu ile Kalman center yolunu 2D olarak çizer.
    """
    raw_x, raw_y = filter_valid_xy(data["raw_cx"], data["raw_cy"])
    kf_x, kf_y = filter_valid_xy(data["kalman_cx"], data["kalman_cy"])

    plt.figure(figsize=(10, 8))

    plt.plot(
        raw_x,
        raw_y,
        marker="o",
        markersize=3,
        linewidth=1,
        label="Raw YOLO center path",
    )

    plt.plot(
        kf_x,
        kf_y,
        marker="o",
        markersize=3,
        linewidth=2,
        label="Kalman center path",
    )

    plt.xlabel("cx")
    plt.ylabel("cy")
    plt.title("Raw YOLO Center vs Kalman Center Path")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    output_path = os.path.join(output_dir, "center_path_comparison.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Kaydedildi: {output_path}")


def save_center_error_over_time(data: Dict[str, List], output_dir: str):
    """
    Raw YOLO center ile Kalman center arasındaki piksel mesafesini frame'e göre çizer.
    """
    frames, errors = filter_valid_frame_value(
        data["frame_idx"],
        data["center_error"],
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        frames,
        errors,
        linewidth=2,
        label="Center distance error",
    )

    plt.xlabel("Frame")
    plt.ylabel("Distance error [pixel]")
    plt.title("Raw YOLO Center - Kalman Center Distance Over Time")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(output_dir, "center_error_over_time.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Kaydedildi: {output_path}")


def save_cx_cy_over_time(data: Dict[str, List], output_dir: str):
    """
    cx ve cy değerlerini zaman ekseninde raw/kalman karşılaştırmalı çizer.
    """
    frames_raw_cx, raw_cx = filter_valid_frame_value(
        data["frame_idx"],
        data["raw_cx"],
    )
    frames_kf_cx, kf_cx = filter_valid_frame_value(
        data["frame_idx"],
        data["kalman_cx"],
    )

    frames_raw_cy, raw_cy = filter_valid_frame_value(
        data["frame_idx"],
        data["raw_cy"],
    )
    frames_kf_cy, kf_cy = filter_valid_frame_value(
        data["frame_idx"],
        data["kalman_cy"],
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        frames_raw_cx,
        raw_cx,
        linewidth=1,
        label="Raw cx",
    )

    plt.plot(
        frames_kf_cx,
        kf_cx,
        linewidth=2,
        label="Kalman cx",
    )

    plt.plot(
        frames_raw_cy,
        raw_cy,
        linewidth=1,
        label="Raw cy",
    )

    plt.plot(
        frames_kf_cy,
        kf_cy,
        linewidth=2,
        label="Kalman cy",
    )

    plt.xlabel("Frame")
    plt.ylabel("Pixel coordinate")
    plt.title("Raw YOLO vs Kalman cx/cy Over Time")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(output_dir, "cx_cy_over_time.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Kaydedildi: {output_path}")


def save_lost_count_over_time(data: Dict[str, List], output_dir: str):
    """
    lost_count değerini frame'e göre çizer.
    """
    frames, lost_counts = filter_valid_frame_value(
        data["frame_idx"],
        data["lost_count"],
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        frames,
        lost_counts,
        linewidth=2,
        label="lost_count",
    )

    plt.xlabel("Frame")
    plt.ylabel("lost_count")
    plt.title("Tracker lost_count Over Time")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(output_dir, "lost_count_over_time.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Kaydedildi: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="videos/output/kalman_log_video9.csv",
        help="Kalman log CSV path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="videos/output/plots",
        help="Grafiklerin kaydedileceği klasör",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = read_log_csv(args.csv)

    save_center_path_comparison(data, args.output_dir)
    save_center_error_over_time(data, args.output_dir)
    save_cx_cy_over_time(data, args.output_dir)
    save_lost_count_over_time(data, args.output_dir)

    print("Tüm grafikler oluşturuldu.")


if __name__ == "__main__":
    main()