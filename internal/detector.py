import pandas as pd
import cv2

from datetime import datetime


class RealTimeDetector:
    def __init__(self, database, faces_model):
        self._logs = dict(name=[], role=[], current_time=[])
        self._database = database
        self._faces_model = faces_model

    @property
    def faces_count(self) -> int:
        return self._database.get_records_count()

    def reset_dict(self):
        self._logs = dict(name=[], role=[], current_time=[])

    def save_logs(self):
        dataframe = pd.DataFrame(self._logs)
        dataframe.drop_duplicates("name", inplace=True)
        name_list = dataframe["name"].tolist()
        role_list = dataframe["role"].tolist()
        ctime_list = dataframe["current_time"].tolist()
        encoded_data = []
        for name, role, ctime in zip(name_list, role_list, ctime_list):
            if name != "Unknown":
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)

        self._database.save_logs(encoded_data)

        self.reset_dict()

    def face_prediction(self, image, threshold=0.5):
        dataframe = self._database.dataframe
        if dataframe.empty:
            print("EEE")
            return image, []

        detections = self._faces_model.face_prediction(image, dataframe, threshold)
        final_img = image.copy()

        for detection in detections:
            self._draw_detection(final_img, detection)

        return final_img, detections

    def _draw_detection(self, frame, detection):
        if not detection.get("is_detected"):
            color = (0, 0, 255)  # bgr
        else:
            color = (0, 255, 0)

        x1, y1, x2, y2 = detection.get("bbox")
        cv2.rectangle(frame, (x1, y1), (x2, y2), color)

        name = detection.get("name")
        role = detection.get("role")

        cv2.putText(frame, name, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)

        self._store_log(name, role)

    def _store_log(self, name, role):
        current_time = str(datetime.now())
        self._logs["name"].append(name)
        self._logs["role"].append(role)
        self._logs["current_time"].append(current_time)
