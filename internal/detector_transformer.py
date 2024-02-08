import numpy as np
import threading
from streamlit_webrtc import VideoTransformerBase
import av

from app import App


class DetectorVideoTransformer(VideoTransformerBase):
    frame_lock: threading.Lock
    in_image: np.ndarray | None
    out_image: np.ndarray | None

    app = None

    @classmethod
    def set_app(cls, app: App):
        cls.app = app

    def __init__(self) -> None:
        self._detector = self.app.get_detector()

        self.frame_lock = threading.Lock()
        self.in_image = None
        self.out_image = None
        self.detections = []

    def recv(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")

        with self.frame_lock:
            out_image, names = img, []
            if self._detector.faces_count > 0:
                out_image, detections = self._detector.face_prediction(img)
                names = [d.get("name") for d in detections]
            self.detections = names

            message = "No detection"
            if names:
                message = f"Detections: {', '.join(names)}"
            print(f"\b\r{message}", end="")

            out_image = av.VideoFrame.from_ndarray(out_image, format="bgr24")
            self.in_image = img
            self.out_image = out_image

        return self.out_image
