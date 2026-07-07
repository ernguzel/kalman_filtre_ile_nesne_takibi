import os
from typing import Dict, Optional, Tuple

import cv2

from src.detector import YOLOPersonDetector
from src.tracker import SinglePersonKalmanTracker
from src.utils import (
    append_csv_log,
    cxcywh_to_xyxy,
    draw_bbox_xyxy,
    draw_center,
    draw_legend,
    draw_polyline,
    ensure_dir,
    init_csv_log,
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
        - Merkez geçmişlerini çizmek
        - Sağ üstte legend göstermek
        - CSV log kaydetmek
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

        self.raw_center_history = []
        self.kalman_center_history = []

        self.log_path = config["paths"].get("log_csv", None)

    def run(self, input_video_path: str, output_video_path: str) -> None:
        output_dir = self.config["paths"].get(
            "output_dir",
            os.path.dirname(output_video_path),
        )
        ensure_dir(output_dir)

        if self.log_path is not None:
            init_csv_log(self.log_path)
            print(f"Log CSV: {self.log_path}")

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

        if self.log_path is not None:
            print(f"Kaydedilen log: {self.log_path}")

    def process_frame(self, frame, frame_idx: int):
        detections = self.detector.detect(frame)

        best_detection = select_best_detection(
            detections,
            strategy=self.detector_cfg["select_strategy"],
        )

        tracker_result = self.tracker.update(best_detection)

        status_parts = [f"Frame: {frame_idx}"]

        raw_info = self._draw_raw_detection(frame, best_detection, status_parts)
        kalman_info = self._draw_kalman_result(frame, tracker_result, status_parts)

        self._draw_trails(frame)
        self._draw_legend(frame)

        status_text = " | ".join(status_parts)

        if self.draw_cfg.get("show_status_text", True):
            put_status_text(
                frame,
                status_text,
                position=(20, 40),
                color=tuple(self.draw_cfg["text_color"]),
            )

        self._write_log(
            frame_idx=frame_idx,
            raw_info=raw_info,
            kalman_info=kalman_info,
            tracker_result=tracker_result,
        )

        return frame, status_text

    def _draw_raw_detection(self, frame, detection, status_parts) -> Dict:
        """
        Raw YOLO detection çizimi ve raw center history kaydı.
        """
        raw_info = {
            "yolo_detected": False,
            "yolo_conf": None,
            "raw_cx": None,
            "raw_cy": None,
            "raw_w": None,
            "raw_h": None,
        }

        if detection is None:
            status_parts.append("YOLO no detection")
            self.raw_center_history.append(None)
            return raw_info

        raw_bbox_xyxy = detection["bbox_xyxy"]
        conf = detection["conf"]

        raw_cx, raw_cy, raw_w, raw_h = xyxy_to_cxcywh(raw_bbox_xyxy)

        raw_info = {
            "yolo_detected": True,
            "yolo_conf": conf,
            "raw_cx": raw_cx,
            "raw_cy": raw_cy,
            "raw_w": raw_w,
            "raw_h": raw_h,
        }

        self.raw_center_history.append((raw_cx, raw_cy))

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

        return raw_info

    def _draw_kalman_result(self, frame, tracker_result, status_parts) -> Dict:
        """
        Kalman tracker sonucunu çizer ve Kalman center history kaydı yapar.
        """
        kalman_info = {
            "kalman_active": False,
            "track_id": None,
            "kalman_status": self.tracker.last_status,
            "lost_count": None,
            "kalman_cx": None,
            "kalman_cy": None,
            "kalman_w": None,
            "kalman_h": None,
            "vx": None,
            "vy": None,
            "vw": None,
            "vh": None,
        }

        if tracker_result is None:
            status_parts.append(f"Tracker status={self.tracker.last_status}")
            self.kalman_center_history.append(None)
            return kalman_info

        kalman_bbox_cxcywh = tracker_result["bbox_cxcywh"]
        kalman_bbox_xyxy = cxcywh_to_xyxy(kalman_bbox_cxcywh)

        kcx, kcy, kw, kh = kalman_bbox_cxcywh
        vx, vy, vw, vh = tracker_result["velocity"]

        status = tracker_result["status"]
        lost_count = tracker_result["lost_count"]
        track_id = tracker_result["track_id"]

        self.kalman_center_history.append((kcx, kcy))

        kalman_info = {
            "kalman_active": tracker_result["active"],
            "track_id": track_id,
            "kalman_status": status,
            "lost_count": lost_count,
            "kalman_cx": kcx,
            "kalman_cy": kcy,
            "kalman_w": kw,
            "kalman_h": kh,
            "vx": vx,
            "vy": vy,
            "vw": vw,
            "vh": vh,
        }

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

        return kalman_info

    def _draw_trails(self, frame) -> None:
        """
        İlk frame'den itibaren raw center ve Kalman center geçmişini çizer.
        """
        if not self.draw_cfg.get("show_trails", True):
            return

        trail_max_length = int(self.draw_cfg.get("trail_max_length", 0))

        raw_history = self.raw_center_history
        kalman_history = self.kalman_center_history

        if trail_max_length > 0:
            raw_history = raw_history[-trail_max_length:]
            kalman_history = kalman_history[-trail_max_length:]

        draw_polyline(
            frame,
            raw_history,
            color=tuple(self.draw_cfg["raw_center_trail_color"]),
            thickness=self.draw_cfg.get("trail_thickness", 2),
        )

        draw_polyline(
            frame,
            kalman_history,
            color=tuple(self.draw_cfg["kalman_center_trail_color"]),
            thickness=self.draw_cfg.get("trail_thickness", 2),
        )

    def _draw_legend(self, frame) -> None:
        """
        Sağ üstte belirgin legend çizer.
        """
        if not self.draw_cfg.get("show_legend", True):
            return

        items = [
            ("YOLO bbox", tuple(self.draw_cfg["raw_bbox_color"])),
            ("Kalman bbox", tuple(self.draw_cfg["kalman_bbox_color"])),
            ("YOLO center", tuple(self.draw_cfg["center_color"])),
            ("YOLO center trail", tuple(self.draw_cfg["raw_center_trail_color"])),
            ("Kalman center trail", tuple(self.draw_cfg["kalman_center_trail_color"])),
        ]

        draw_legend(frame, items)

    def _write_log(
        self,
        frame_idx: int,
        raw_info: Dict,
        kalman_info: Dict,
        tracker_result: Optional[Dict],
    ) -> None:
        """
        Her frame için CSV log yazar.
        """
        if self.log_path is None:
            return

        row = {
            "frame_idx": frame_idx,
            "yolo_detected": raw_info["yolo_detected"],
            "yolo_conf": raw_info["yolo_conf"],
            "raw_cx": raw_info["raw_cx"],
            "raw_cy": raw_info["raw_cy"],
            "raw_w": raw_info["raw_w"],
            "raw_h": raw_info["raw_h"],
            "kalman_active": kalman_info["kalman_active"],
            "track_id": kalman_info["track_id"],
            "kalman_status": kalman_info["kalman_status"],
            "lost_count": kalman_info["lost_count"],
            "kalman_cx": kalman_info["kalman_cx"],
            "kalman_cy": kalman_info["kalman_cy"],
            "kalman_w": kalman_info["kalman_w"],
            "kalman_h": kalman_info["kalman_h"],
            "vx": kalman_info["vx"],
            "vy": kalman_info["vy"],
            "vw": kalman_info["vw"],
            "vh": kalman_info["vh"],
        }

        append_csv_log(self.log_path, row)


    def run_rtsp_gst_python(
        self,
        rtsp_url: str,
        output_video_path: str = None,
        latency: int = 0,
        flip_enabled: bool = False,
        flip_code: int = -1,
        max_frames: int = 0,
    ) -> None:
        """
        OpenCV CAP_GSTREAMER kullanmadan, Python GStreamer appsink ile RTSP işler.

        Akış:
            RTSP kamera
            -> GStreamer appsink
            -> numpy BGR frame
            -> YOLO
            -> Kalman
            -> çizim
            -> output video / CSV log
            -> GStreamer appsrc ile canlı gösterim

        Not:
            Bu fonksiyon cv2.imshow(), cv2.waitKey(), cv2.destroyAllWindows()
            kullanmaz. Bu yüzden OpenCV GUI desteği olmayan env'lerde de çalışır.
        """

        from src.gstreamer_capture import GStreamerRTSPCapture

        cap = GStreamerRTSPCapture(
            rtsp_url=rtsp_url,
            latency=latency,
        )

        writer = None
        display = None

        if self.log_path is not None:
            init_csv_log(self.log_path)
            print(f"Log CSV: {self.log_path}")

        print("RTSP GStreamer Python Kalman processing başladı.")
        print(f"RTSP URL: {rtsp_url}")
        print("show_live true ise GStreamer appsrc ile canlı görüntü gösterilir.")
        print("Çıkmak için terminalde Ctrl+C kullanabilirsin.")

        frame_idx = 0
        failed_count = 0

        try:
            while True:
                ret, frame = cap.read()

                if not ret or frame is None:
                    failed_count += 1

                    if failed_count % 30 == 0:
                        print(f"Frame alınamadı. failed_count={failed_count}")

                    continue

                failed_count = 0

                # Kamera ters ise düzelt
                if flip_enabled:
                    frame = cv2.flip(frame, flip_code)

                # Output writer ilk geçerli frame geldikten sonra açılır.
                if output_video_path is not None and writer is None:
                    output_dir = self.config["paths"].get(
                        "output_dir",
                        os.path.dirname(output_video_path),
                    )
                    ensure_dir(output_dir)

                    h, w = frame.shape[:2]

                    fps = self.video_cfg.get("rtsp_output_fps", 30.0)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                    writer = cv2.VideoWriter(
                        output_video_path,
                        fourcc,
                        fps,
                        (w, h),
                    )

                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Output video oluşturulamadı: {output_video_path}"
                        )

                    print(f"Output video: {output_video_path}")
                    print(f"Writer resolution: {w}x{h}")
                    print(f"Writer FPS: {fps}")

                # YOLO + Kalman + çizimler + legend + log hazırlığı
                processed_frame, status_text = self.process_frame(frame, frame_idx)

                # Video kaydı
                if writer is not None:
                    writer.write(processed_frame)

                # Canlı görüntü: cv2.imshow yerine GStreamer appsrc kullanıyoruz.
                if self.video_cfg.get("show_live", False):
                    display_frame = processed_frame

                    if self.video_cfg.get("display_resize", False):
                        display_frame = resize_for_display(
                            processed_frame,
                            max_width=self.video_cfg.get("display_max_width", 1280),
                            max_height=self.video_cfg.get("display_max_height", 720),
                        )

                    if display is None:
                        from src.gstreamer_display import GStreamerDisplay

                        display_h, display_w = display_frame.shape[:2]

                        display = GStreamerDisplay(
                            width=display_w,
                            height=display_h,
                            fps=self.video_cfg.get("rtsp_output_fps", 30.0),
                        )

                        print(f"Live display açıldı: {display_w}x{display_h}")

                    display.show(display_frame)

                if frame_idx % 30 == 0:
                    print(status_text)

                frame_idx += 1

                if max_frames > 0 and frame_idx >= max_frames:
                    print(f"max_frames={max_frames} değerine ulaşıldı.")
                    break

        except KeyboardInterrupt:
            print("Ctrl+C ile çıkış yapıldı.")

        finally:
            cap.release()

            if writer is not None:
                writer.release()

            if display is not None:
                display.release()

        print("RTSP GStreamer Python Kalman processing bitti.")

        if output_video_path is not None:
            print(f"Kaydedilen video: {output_video_path}")

        if self.log_path is not None:
            print(f"Kaydedilen log: {self.log_path}")