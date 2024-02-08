import streamlit as st

from app import load_app
from streamlit_webrtc import webrtc_streamer
import time

from internal.detector_transformer import DetectorVideoTransformer

app = load_app()

st.set_page_config("FRS - Detector", layout="centered")
st.subheader("🔎 Face recognition")

waitTime = 5  # time in sec
t0 = time.time()
detector = app.get_detector()

detection_widget = st.empty()
DetectorVideoTransformer.set_app(app)
cam_widget = webrtc_streamer(key="snapshot", video_processor_factory=DetectorVideoTransformer,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

while cam_widget.video_transformer:
    transformer = cam_widget.video_transformer
    if not transformer:
        break
    with transformer.frame_lock:
        detections = transformer.detections

    if not detections:
        detection_widget.error("No detection")
    elif "Unknown" in detections:
        detections = ", ".join(detections)
        detection_widget.warning(detections)
    else:
        detections = ", ".join(detections)
        detection_widget.success(detections)
