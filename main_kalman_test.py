from src.detector import YOLOPersonDetector
from src.tracker import SinglePersonKalmanTracker
from src.utils import load_config
from src.video_processor import VideoProcessor


def main():
    config = load_config("configs/config.yaml")

    detector_cfg = config["detector"]
    tracker_cfg = config["tracker"]
    kalman_cfg = config["kalman"]

    detector = YOLOPersonDetector(
        model_path=config["paths"]["model_path"],
        conf_threshold=detector_cfg["conf_threshold"],
        person_class_id=detector_cfg["person_class_id"],
        device=detector_cfg["device"],
        imgsz=detector_cfg["imgsz"],
    )

    tracker = SinglePersonKalmanTracker(
        kalman_config=kalman_cfg,
        max_lost_count=tracker_cfg["max_lost_count"],
        predict_when_missing=tracker_cfg["predict_when_missing"],
        restart_when_detection_returns=tracker_cfg["restart_when_detection_returns"],
    )

    processor = VideoProcessor(
        detector=detector,
        tracker=tracker,
        config=config,
    )

    processor.run(
        input_video_path=config["paths"]["input_video"],
        output_video_path=config["paths"]["output_video"],
    )


if __name__ == "__main__":
    main()