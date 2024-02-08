import streamlit as st
from internal.recorder import InvalidNameError, EmbeddingDataNotFoundError
from streamlit_webrtc import webrtc_streamer
from app import load_app
from internal.recorder_transformer import RecorderVideoTransformer

app = load_app()

st.set_page_config("FRS - Recorder", layout="centered")
st.subheader("⭕ Registration Form")

recorder = app.get_recorder()

name = st.text_input(label="Name", placeholder="Enter username")

if st.button("Submit", disabled=name == ""):
    saved, error = recorder.save(name, "user")
    if saved:
        st.success(f"{name} registered successfully")
    elif error == InvalidNameError:
        st.error("Please enter the name: Name cannot be empty or spaces")
    elif error == EmbeddingDataNotFoundError:
        st.error("face_embedding.txt is not found. Please refresh the page and execute again.")

detected_widget = st.empty()
RecorderVideoTransformer.set_app(app)
cam_widget = webrtc_streamer(key="snapshot", video_processor_factory=RecorderVideoTransformer, rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

while cam_widget.video_transformer:
    transformer = cam_widget.video_transformer
    if not transformer:
        break
    with transformer.frame_lock:
        if transformer.detected:
            message = f"Face detected: ({transformer.samples} samples)"
            detected_widget.success(message)
        else:
            detected_widget.error("No face detected")
