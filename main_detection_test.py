import os

import cv2

from src.detector import YOLOPersonDetector
from src.utils import (
    draw_bbox_xyxy,
    draw_center,
    ensure_dir,
    load_config,
    put_status_text,
    select_best_detection,
    xyxy_to_cxcywh,
    resize_for_display
)


def main():
    config = load_config("configs/config.yaml")

    model_path = config["paths"]["model_path"]
    input_video_path = config["paths"]["input_video"]
    output_video_path = config["paths"]["output_video"]
    output_dir = config["paths"].get("output_dir", os.path.dirname(output_video_path))

    ensure_dir(output_dir)

    detector_cfg = config["detector"]
    draw_cfg = config["draw"]
    video_cfg = config["video"]

    detector = YOLOPersonDetector(
        model_path=model_path,
        conf_threshold=detector_cfg["conf_threshold"],
        person_class_id=detector_cfg["person_class_id"],
        device=detector_cfg["device"],
        imgsz=detector_cfg["imgsz"],
    )

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {input_video_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if input_fps <= 0:
        input_fps = 30.0

    output_fps = input_fps if video_cfg["use_original_fps"] else 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        output_fps,
        (frame_width, frame_height)
    )

    if not writer.isOpened():
        raise RuntimeError(f"Output video oluşturulamadı: {output_video_path}")

    frame_idx = 0

    print("Detection test başladı.")
    print(f"Input video : {input_video_path}")
    print(f"Output video: {output_video_path}")
    print(f"FPS         : {output_fps}")
    print(f"Resolution  : {frame_width}x{frame_height}")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        detections = detector.detect(frame)

        best_detection = select_best_detection(
            detections,
            strategy=detector_cfg["select_strategy"]
        )

        if best_detection is not None:
            bbox_xyxy = best_detection["bbox_xyxy"]
            conf = best_detection["conf"]

            cx, cy, w, h = xyxy_to_cxcywh(bbox_xyxy)

            if draw_cfg["show_raw_detection"]:
                label = f"YOLO person {conf:.2f}"
                draw_bbox_xyxy(
                    frame,
                    bbox_xyxy,
                    color=tuple(draw_cfg["raw_bbox_color"]),
                    label=label,
                    thickness=draw_cfg["bbox_thickness"]
                )

            if draw_cfg["show_centers"]:
                draw_center(
                    frame,
                    cx,
                    cy,
                    color=tuple(draw_cfg["center_color"]),
                    radius=draw_cfg["center_radius"]
                )

            status_text = (
                f"Frame: {frame_idx} | "
                f"Person detected | "
                f"conf={conf:.2f} | "
                f"cx={cx:.1f}, cy={cy:.1f}, w={w:.1f}, h={h:.1f}"
            )
        else:
            status_text = f"Frame: {frame_idx} | Person not detected"

        if draw_cfg["show_status_text"]:
            put_status_text(
                frame,
                status_text,
                position=(20, 40),
                color=tuple(draw_cfg["text_color"])
            )

        writer.write(frame)

        if video_cfg["show_live"]:
            display_frame = frame

            if video_cfg.get("display_resize", False):
                display_frame = resize_for_display(
                    frame,
                    max_width=video_cfg.get("display_max_width", 1280),
                    max_height=video_cfg.get("display_max_height", 720)
                )

            cv2.imshow("YOLO Person Detection Test", display_frame)

            key = cv2.waitKey(video_cfg.get("wait_key_delay", 1)) & 0xFF

            if key == 27 or key == ord("q"):
                print("Kullanıcı çıkış yaptı.")
                break

        if frame_idx % 30 == 0:
            print(status_text)

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("Detection test bitti.")
    print(f"Kaydedilen video: {output_video_path}")


if __name__ == "__main__":
    main()