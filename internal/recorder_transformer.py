import numpy as np
import threading
from streamlit_webrtc import VideoTransformerBase
import av

from app import App


class RecorderVideoTransformer(VideoTransformerBase):
    frame_lock: threading.Lock
    in_image: np.ndarray | None
    out_image: np.ndarray | None
    samples: int
    detected: bool

    app = None

    @classmethod
    def set_app(cls, app: App):
        cls.app = app

    def __init__(self) -> None:
        self._recorder = self.app.get_recorder()
        self.frame_lock = threading.Lock()
        self.in_image = None
        self.out_image = None
        self.detected = False
        self_samples = 0

    def recv(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")

        with self.frame_lock:
            out_image, detected, samples = self._recorder.capture(img)

            message = "No detection"
            if detected:
                message = f"Face detected ({samples} recorded)"
            print(f"\b\r{message}", end="")

            out_image = av.VideoFrame.from_ndarray(out_image, format="bgr24")
            self.in_image = img
            self.out_image = out_image
            self.detected = detected
            self.samples = samples

        return self.out_image
