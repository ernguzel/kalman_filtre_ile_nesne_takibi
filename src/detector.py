from typing import Any, Dict, List

from ultralytics import YOLO


class YOLOPersonDetector:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        person_class_id: int = 0,
        device: Any = 0,
        imgsz: int = 640
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.person_class_id = person_class_id
        self.device = device
        self.imgsz = imgsz

        self.model = YOLO(model_path)

    def detect(self, frame) -> List[Dict[str, Any]]:
        """
        Frame üzerinde YOLO predict çalıştırır.
        Sadece person class detection'larını döndürür.

        Çıktı:
        [
            {
                "bbox_xyxy": [x1, y1, x2, y2],
                "conf": 0.87,
                "cls": 0
            }
        ]
        """

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )

        detections: List[Dict[str, Any]] = []

        if len(results) == 0:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        boxes = result.boxes

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if cls_id != self.person_class_id:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()

            detections.append(
                {
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "conf": conf,
                    "cls": cls_id
                }
            )

        return detections