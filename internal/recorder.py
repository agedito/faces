import numpy as np
import cv2

import os

InvalidNameError = "Invalid name"
EmbeddingDataNotFoundError = "Embedding data not found"


class Recorder:
    _embedding_file = "face_embedding.txt"

    def __init__(self, database, faces_model):
        self._sample = 0
        self._database = database
        self._faces_model = faces_model

    def reset(self):
        self._sample = 0

    def capture(self, frame):
        detection = self._faces_model.find_unique_face(frame)
        if detection is None:
            return frame, False, self._sample

        self._sample += 1
        detected = self._save_detection(detection)
        if not detected:
            return frame, False, self._sample

        frame = self._draw(frame, detection)
        return frame, True, self._sample

    def _save_detection(self, detection):
        embedding = detection["embedding"]
        if embedding is None:
            return False

        with open(self._embedding_file, mode="ab") as f:
            np.savetxt(f, embedding)

        return True

    def _draw(self, frame, detection):
        x1, y1, x2, y2 = detection["bbox"].astype(int)
        color = (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # text = f"samples = {self._sample}"
        # cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)

        return frame

    def _generate_id(self, name: str, role: str) -> (str, str | None):
        if name is None or role is None:
            return "", InvalidNameError

        name = name.strip()
        if name == "":
            return False, InvalidNameError

        key = f"{name}@{role}"
        return key, None

    def save(self, name: str, role: str) -> (bool, str):
        if self._embedding_file not in os.listdir():
            return False, EmbeddingDataNotFoundError

        record_id, error = self._generate_id(name, role)
        if error is not None:
            return False, error

        self._save(record_id)
        self.reset()

        return True, ""

    def _save(self, record_id: str):
        x_array = np.loadtxt(self._embedding_file, dtype=np.float32)  # flatten array

        received_samples = int(x_array.size / 512)
        x_array = x_array.reshape(received_samples, 512)
        x_array = np.asarray(x_array)

        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()

        self._database.save_record(record_id, x_mean_bytes)

        os.remove(self._embedding_file)
