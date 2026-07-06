import os
from typing import Dict

import cv2

from src.detector import YOLOPersonDetector
from src.tracker import SinglePersonKalmanTracker
from src.utils import (
    cxcywh_to_xyxy,
    draw_bbox_xyxy,
    draw_center,
    ensure_dir,
    put_status_text,
    resize_for_display,
    select_best_detection,
    xyxy_to_cxcywh,
)


class VideoProcessor:
    """
    Video işleme sınıfı.

    Görevi:
        - Videoyu açmak
        - Her frame'de YOLO person detection yapmak
        - Detection sonucunu Kalman tracker'a vermek
        - Raw YOLO bbox ve Kalman bbox çizmek
        - Output video kaydetmek
        - İstenirse canlı göstermek
    """

    def __init__(
        self,
        detector: YOLOPersonDetector,
        tracker: SinglePersonKalmanTracker,
        config: Dict,
    ):
        self.detector = detector
        self.tracker = tracker
        self.config = config

        self.detector_cfg = config["detector"]
        self.draw_cfg = config["draw"]
        self.video_cfg = config["video"]

    def run(self, input_video_path: str, output_video_path: str) -> None:
        output_dir = self.config["paths"].get(
            "output_dir",
            os.path.dirname(output_video_path),
        )
        ensure_dir(output_dir)

        cap = cv2.VideoCapture(input_video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Video açılamadı: {input_video_path}")

        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if input_fps <= 0:
            input_fps = 30.0

        output_fps = (
            input_fps
            if self.video_cfg.get("use_original_fps", True)
            else 30.0
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_video_path,
            fourcc,
            output_fps,
            (frame_width, frame_height),
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Output video oluşturulamadı: {output_video_path}")

        print("Video processing başladı.")
        print(f"Input video : {input_video_path}")
        print(f"Output video: {output_video_path}")
        print(f"FPS         : {output_fps}")
        print(f"Resolution  : {frame_width}x{frame_height}")

        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                processed_frame, status_text = self.process_frame(frame, frame_idx)

                writer.write(processed_frame)

                if self.video_cfg.get("show_live", False):
                    display_frame = processed_frame

                    if self.video_cfg.get("display_resize", False):
                        display_frame = resize_for_display(
                            processed_frame,
                            max_width=self.video_cfg.get("display_max_width", 1280),
                            max_height=self.video_cfg.get("display_max_height", 720),
                        )

                    cv2.imshow("YOLO + Kalman BBox Tracker", display_frame)

                    key = cv2.waitKey(
                        self.video_cfg.get("wait_key_delay", 1)
                    ) & 0xFF

                    if key == 27 or key == ord("q"):
                        print("Kullanıcı çıkış yaptı.")
                        break

                if frame_idx % 30 == 0:
                    print(status_text)

                frame_idx += 1

        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()

        print("Video processing bitti.")
        print(f"Kaydedilen video: {output_video_path}")

    def process_frame(self, frame, frame_idx: int):
        """
        Tek bir frame işler.

        Akış:
            1. YOLO detect
            2. En iyi person bbox seç
            3. Tracker update
            4. Raw YOLO bbox çiz
            5. Kalman bbox çiz
            6. Status text yaz
        """

        detections = self.detector.detect(frame)

        best_detection = select_best_detection(
            detections,
            strategy=self.detector_cfg["select_strategy"],
        )

        tracker_result = self.tracker.update(best_detection)

        status_parts = [f"Frame: {frame_idx}"]

        self._draw_raw_detection(frame, best_detection, status_parts)
        self._draw_kalman_result(frame, tracker_result, status_parts)

        status_text = " | ".join(status_parts)

        if self.draw_cfg.get("show_status_text", True):
            put_status_text(
                frame,
                status_text,
                position=(20, 40),
                color=tuple(self.draw_cfg["text_color"]),
            )

        return frame, status_text

    def _draw_raw_detection(self, frame, detection, status_parts) -> None:
        """
        Raw YOLO detection çizimi.
        """

        if detection is None:
            status_parts.append("YOLO no detection")
            return

        raw_bbox_xyxy = detection["bbox_xyxy"]
        conf = detection["conf"]

        raw_cx, raw_cy, raw_w, raw_h = xyxy_to_cxcywh(raw_bbox_xyxy)

        status_parts.append(f"YOLO conf={conf:.2f}")

        if self.draw_cfg.get("show_raw_detection", True):
            draw_bbox_xyxy(
                frame,
                raw_bbox_xyxy,
                color=tuple(self.draw_cfg["raw_bbox_color"]),
                label=f"YOLO {conf:.2f}",
                thickness=self.draw_cfg["bbox_thickness"],
            )

        if self.draw_cfg.get("show_centers", True):
            draw_center(
                frame,
                raw_cx,
                raw_cy,
                color=tuple(self.draw_cfg["center_color"]),
                radius=self.draw_cfg["center_radius"],
            )

    def _draw_kalman_result(self, frame, tracker_result, status_parts) -> None:
        """
        Kalman tracker sonucunu çizer.
        """

        if tracker_result is None:
            status_parts.append(f"Tracker status={self.tracker.last_status}")
            return

        kalman_bbox_cxcywh = tracker_result["bbox_cxcywh"]
        kalman_bbox_xyxy = cxcywh_to_xyxy(kalman_bbox_cxcywh)

        kcx, kcy, kw, kh = kalman_bbox_cxcywh
        vx, vy, vw, vh = tracker_result["velocity"]

        status = tracker_result["status"]
        lost_count = tracker_result["lost_count"]
        track_id = tracker_result["track_id"]

        status_parts.append(
            f"Track ID={track_id} | {status} | lost={lost_count}"
        )

        if self.draw_cfg.get("show_kalman_bbox", True):
            draw_bbox_xyxy(
                frame,
                kalman_bbox_xyxy,
                color=tuple(self.draw_cfg["kalman_bbox_color"]),
                label=f"KF ID:{track_id} {status}",
                thickness=self.draw_cfg["kalman_bbox_thickness"],
            )

        if self.draw_cfg.get("show_centers", True):
            draw_center(
                frame,
                kcx,
                kcy,
                color=tuple(self.draw_cfg["kalman_bbox_color"]),
                radius=self.draw_cfg["center_radius"],
            )

        velocity_text = (
            f"vx={vx:.2f}, vy={vy:.2f}, "
            f"vw={vw:.2f}, vh={vh:.2f}"
        )

        put_status_text(
            frame,
            velocity_text,
            position=(20, 75),
            color=tuple(self.draw_cfg["text_color"]),
        )