import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst


class GStreamerDisplay:
    """
    OpenCV imshow kullanmadan numpy BGR frame'i GStreamer ile ekrana basar.

    Python/Numpy BGR frame
        -> appsrc
        -> videoconvert
        -> autovideosink
    """

    def __init__(self, width: int, height: int, fps: float = 30.0):
        Gst.init(None)

        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)

        fps_num = int(round(self.fps))
        fps_den = 1

        self.frame_duration = int(1e9 / self.fps)
        self.frame_count = 0

        self.pipeline_str = (
            "appsrc name=src "
            "is-live=true "
            "block=false "
            "format=time "
            f"caps=video/x-raw,format=BGR,width={self.width},height={self.height},framerate={fps_num}/{fps_den} "
            "! queue leaky=downstream max-size-buffers=1 max-size-time=0 max-size-bytes=0 "
            "! videoconvert "
            "! autovideosink sync=false"
        )

        self.pipeline = Gst.parse_launch(self.pipeline_str)
        self.appsrc = self.pipeline.get_by_name("src")

        if self.appsrc is None:
            raise RuntimeError("appsrc bulunamadı.")

        self.pipeline.set_state(Gst.State.PLAYING)

    def show(self, frame: np.ndarray) -> None:
        """
        Frame'i ekrana yollar.
        Frame BGR ve uint8 olmalı.
        """

        if frame is None:
            return

        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(
                f"Frame boyutu yanlış. Beklenen: {self.width}x{self.height}, "
                f"gelen: {frame.shape[1]}x{frame.shape[0]}"
            )

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        frame = np.ascontiguousarray(frame)

        data = frame.tobytes()
        buffer = Gst.Buffer.new_allocate(None, len(data), None)
        buffer.fill(0, data)

        pts = self.frame_count * self.frame_duration
        buffer.pts = pts
        buffer.dts = pts
        buffer.duration = self.frame_duration

        self.frame_count += 1

        ret = self.appsrc.emit("push-buffer", buffer)

        if ret != Gst.FlowReturn.OK:
            print(f"GStreamer display push-buffer uyarısı: {ret}")

    def release(self) -> None:
        try:
            self.appsrc.emit("end-of-stream")
        except Exception:
            pass

        self.pipeline.set_state(Gst.State.NULL)