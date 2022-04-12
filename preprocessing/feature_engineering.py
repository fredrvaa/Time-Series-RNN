import pandas as pd
from datetime import datetime


def to_datetime(time: str) -> datetime:
    return datetime.strptime(time, '%Y-%m-%d %H:%M:%S')


def add_time_of_day(data: pd.DataFrame) -> pd.DataFrame:
    def to_time_of_day(idx: str):
        idx = to_datetime(idx)
        return idx.hour // 4

    prefix = 'time_of_day'
    data[prefix] = data.index.map(to_time_of_day)
    one_hot = pd.get_dummies(data[prefix], prefix=prefix)
    data = data.drop(prefix, axis=1)
    data = data.join(one_hot)
    return data


def add_time_of_week(data: pd.DataFrame) -> pd.DataFrame:
    def to_time_of_week(time: str) -> int:
        time = to_datetime(time)
        return 0 if time.weekday() < 4 else 1

    data['time_of_week'] = data.index.map(to_time_of_week)
    return data


def add_time_of_year(data: pd.DataFrame) -> pd.DataFrame:
    def to_time_of_year(time: str) -> int:
        time = to_datetime(time)
        return (time.month + 1) // 4

    prefix = 'time_of_year'
    data[prefix] = data.index.map(to_time_of_year)
    one_hot = pd.get_dummies(data[prefix], prefix=prefix)
    data = data.drop(prefix, axis=1)
    data = data.join(one_hot)
    return data


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    data = add_time_of_year(data)
    data = add_time_of_week(data)
    data = add_time_of_day(data)
    return data
