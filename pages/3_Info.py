import streamlit as st
from app import load_app

app = load_app()

st.set_page_config("FRS - Info", layout="wide")
st.subheader("📋 Records info")


def _show_data(widget, reload: bool):
    dataframe = app.database.load_data() if reload else app.database.dataframe

    if dataframe.empty:
        widget.warning("No data found")
    else:
        fields = dataframe[["Name", "Role"]]
        widget.dataframe(fields)


def _show_logs(widget):
    logs = app.database.load_logs()
    if not logs:
        widget.warning("No logs found")
    else:
        widget.write(logs)


data_tab, logs_tab = st.tabs(["Registered Data", "Logs"])
with data_tab:
    data_column, actions_column = st.columns([1, 2])
    with data_column:
        table_widget = st.empty()
        _show_data(table_widget, reload=True)
    with actions_column:
        combo_widget = st.empty()
        options = app.database.list_keys()
        option = combo_widget.selectbox("Record to remove?", options)
        if st.button("Delete record", disabled=len(options) == 0):
            app.database.delete_record(option)
            _show_data(table_widget, reload=True)
        if st.button("Delete all", disabled=len(options) == 0):
            app.database.delete_all_records()
            _show_data(table_widget, reload=True)

    if st.button("Refresh Data"):
        with st.spinner("Retrieving data from database..."):
            _show_data(table_widget, reload=True)

with logs_tab:
    refresh_column, delete_column, _ = st.columns([1, 1, 8])
    logs_widget = st.empty()

    with refresh_column:
        if st.button("Refresh Logs"):
            logs = app.database.load_logs()
            _show_logs(logs_widget)
    with delete_column:
        if st.button("Delete Logs"):
            app.database.delete_logs()
            _show_logs(logs_widget)

    _show_logs(logs_widget)
