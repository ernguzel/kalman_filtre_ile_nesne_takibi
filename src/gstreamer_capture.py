import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst


class GStreamerRTSPCapture:
    """
    OpenCV GStreamer backend kullanmadan RTSP frame almak için.

    RTSP -> GStreamer appsink -> numpy BGR frame
    """

    def __init__(self, rtsp_url: str, latency: int = 0):
        Gst.init(None)

        self.rtsp_url = rtsp_url
        self.latency = latency

        self.pipeline_str = self._build_pipeline(rtsp_url, latency)

        self.pipeline = Gst.parse_launch(self.pipeline_str)
        self.appsink = self.pipeline.get_by_name("sink")

        if self.appsink is None:
            raise RuntimeError("appsink bulunamadı. Pipeline içinde name=sink olmalı.")

        self.pipeline.set_state(Gst.State.PLAYING)

    @staticmethod
    def _build_pipeline(rtsp_url: str, latency: int) -> str:
        return (
            f'rtspsrc location="{rtsp_url}" '
            f"latency={latency} "
            f"protocols=tcp "
            f"buffer-mode=0 "
            f"drop-on-latency=true "
            f"do-retransmission=false "
            f"ntp-sync=false "
            f"short-header=true ! "
            f"rtph264depay ! "
            f"h264parse ! "
            f"avdec_h264 ! "
            f"queue leaky=2 max-size-buffers=1 max-size-time=0 max-size-bytes=0 ! "
            f"videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"appsink name=sink emit-signals=false sync=false "
            f"drop=true max-buffers=1"
        )

    def read(self):
        sample = self.appsink.emit("try-pull-sample", 100000000)

        if sample is None:
            return False, None

        buffer = sample.get_buffer()
        caps = sample.get_caps()

        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")

        success, map_info = buffer.map(Gst.MapFlags.READ)

        if not success:
            return False, None

        try:
            frame = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = frame.reshape((height, width, 3))
            frame = frame.copy()
        finally:
            buffer.unmap(map_info)

        return True, frame

    def release(self):
        self.pipeline.set_state(Gst.State.NULL)