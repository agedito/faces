from insightface.app import FaceAnalysis
import numpy as np
from sklearn.metrics import pairwise

unknown = "Unknown"


class FacesModel:
    def __init__(self, config: dict):
        self._app = None
        self._config = config

    def load_model(self):
        model_name = self._config.get("FACES_MODEL")
        self._app = FaceAnalysis(name=model_name, root=".", providers=self._get_providers())
        self._app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
        return self._app

    @staticmethod
    def _get_providers():
        return ["CPUExecutionProvider"]

    def find_unique_face(self, image: np.ndarray):
        detections = self._app.get(image, max_num=1)
        if not detections:
            return
        return detections[0]

    def face_prediction(self, image, dataframe, threshold=0.5) -> list:
        faces_data = []
        for face_detection in self._app.get(image):
            bbox = face_detection['bbox'].astype(int)
            embedding = face_detection['embedding']

            name, role = self._ml_search_algorithm(dataframe,
                                                   feature_column="facial_features",
                                                   test_vector=embedding,
                                                   name_role=['Name', 'Role'],
                                                   threshold=threshold)

            face_data = {
                "is_detected": name != unknown,
                "embedding": embedding,
                "bbox": bbox,
                "name": name,
                "role": role,
            }

            faces_data.append(face_data)

        return faces_data

    @staticmethod
    def _ml_search_algorithm(dataframe, feature_column, test_vector, name_role, threshold):
        dataframe = dataframe.copy()
        x = dataframe[feature_column].tolist()
        x = np.asarray(x)

        similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
        similar_arr = np.array(similar).flatten()
        dataframe["cosine"] = similar_arr

        data_filter = dataframe.query(f"cosine >= {threshold}")
        if len(data_filter) > 0:
            data_filter.reset_index(drop=True, inplace=True)
            argmax = data_filter["cosine"].argmax()
            person_name, person_role = data_filter.loc[argmax][name_role]

        else:
            person_name = unknown
            person_role = unknown

        return person_name, person_role
