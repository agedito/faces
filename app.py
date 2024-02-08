from dotenv import dotenv_values
from internal.database import Database
from internal.detector import RealTimeDetector
from internal.faces_model import FacesModel

import streamlit as st

from internal.recorder import Recorder


@st.cache_resource(show_spinner=False)
def load_app():
    app = App()
    app.initialize()
    st.session_state.app = app
    return app


class App:
    def __init__(self):
        self._database = None
        self._faces_model = None

    @property
    def database(self) -> Database:
        return self._database

    @property
    def faces_model(self) -> FacesModel:
        return self._faces_model

    def get_recorder(self):
        return Recorder(self._database, self._faces_model)

    def get_detector(self):
        return RealTimeDetector(self._database, self._faces_model)

    def initialize(self):
        cfg = dotenv_values("./.env")
        user_cfg = dotenv_values("./.secret")

        self._database = Database(cfg)
        self._database.connect(user_cfg)
        self._database.load_data(cfg.get("REDIS_TABLE"))

        self._faces_model = FacesModel(cfg)
        self._faces_model.load_model()
