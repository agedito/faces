from redis import Redis
import pandas as pd
import numpy as np


class Database:
    def __init__(self, config: dict):
        self._config = config
        self._host = config.get("REDIS_HOST")
        self._port = config.get("REDIS_PORT")
        self._table_key = config.get("REDIS_TABLE")
        self._logs_key = config.get("REDIS_LOGS_KEY")

        self._redis = None
        self._dataframe = None

    @property
    def is_connected(self) -> bool:
        return self._redis is not None

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    def delete_logs(self):
        self._redis.delete(self._logs_key)

    def load_logs(self):
        logs_list = self._redis.lrange(self._logs_key, start=0, end=-1)
        return logs_list

    def save_logs(self, data):
        if len(data) <= 0:
            return

        self._redis.lpush(self._logs_key, *data)

    def list_keys(self) -> list[str]:
        keys = [x.decode() for x in self._redis.hkeys(self._table_key)]
        return keys

    def load_data(self, table_name: str | None = None) -> pd.DataFrame:
        table_name = table_name or self._table_key
        table_data = self._redis.hgetall(table_name)
        if not table_data:
            dataframe = pd.DataFrame()
            self._dataframe = dataframe
            return dataframe
        data_series = pd.Series(table_data)
        retrieve_series = data_series.apply(lambda x: np.frombuffer(x, dtype=np.float32))
        index = retrieve_series.index
        index = list(map(lambda x: x.decode(), index))
        retrieve_series.index = index
        dataframe = retrieve_series.to_frame().reset_index()
        dataframe.columns = ["name_role", "facial_features"]
        dataframe[["Name", "Role"]] = dataframe["name_role"].apply(lambda x: x.split("@")).apply(pd.Series)
        dataframe = dataframe[["Name", "Role", "facial_features"]]

        self._dataframe = dataframe
        return dataframe

    def delete_all_records(self):
        self._redis.delete(self._table_key)

    def delete_record(self, key: str):
        self._redis.hdel(self._table_key, key)

    def save_record(self, person_id: str, data):
        self._redis.hset(name=self._table_key, key=person_id, value=data)

    def get_records_count(self) -> int:
        return self._redis.hlen(self._table_key)

    def connect(self, login_config: dict) -> bool:
        try:
            self._redis = Redis(
                host=self._host,
                port=self._port,
                username=login_config.get("REDIS_USER"),
                password=login_config.get("REDIS_PASSWORD"),
            )

            is_connected = self._redis.ping()
        except:
            is_connected = False
        if not is_connected:
            self._redis = None
        return is_connected
