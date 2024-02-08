import streamlit as st
from app import load_app

title = "Face Recognition System"
st.set_page_config(page_title=title, layout="wide")

st.header(title)
st.image("./resources/logo.png", width=512)

with st.sidebar:
    state_widget = st.warning("Loading...")
    app = load_app()

    state_widget.empty()

    if app.database is None:
        st.error("Error connecting to database")
    else:
        st.success("Database connected")

    if app.faces_model is None:
        st.error("Error loading faces model")
    else:
        st.success("Faces model loaded")
