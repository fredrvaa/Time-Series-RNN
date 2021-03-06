import datetime
import math
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from preprocessing.utils import to_datetime, get_day_seconds, get_year_seconds


class Preprocessor:
    """
    Class housing methods for preprocessing time series data.

    Note: Preprocessing is here defined as both standard
          data preprocessing and feature engineering.
    """

    #######################
    #    PREPROCESSING    #
    #######################

    @staticmethod
    def clip(data: pd.DataFrame,
             clip_percent: float = .01,
             q_low: float = .25,
             q_high: float = .75,
             columns: Optional[list[str]] = None) -> pd.DataFrame:

        n_clip = int(data.shape[0] * clip_percent)

        if columns is None:
            columns = data.columns

        for column in columns:
            dev = (data[column] - data[column].mean()).abs().sort_values(ascending=False)[:n_clip]
            low, high = data[column].quantile([q_low, q_high])

            data.loc[dev.index.intersection(data[data[column] >= high].index), column] = high
            data.loc[dev.index.intersection(data[data[column] <= low].index), column] = low

        return data

    @staticmethod
    def scale(data: pd.DataFrame, scaler=StandardScaler()):
        data[:] = scaler.fit_transform(data)
        return data

    #######################
    # FEATURE ENGINEERING #
    #######################

    @staticmethod
    def add_time_of_day_categorical(data: pd.DataFrame) -> pd.DataFrame:
        def to_time_of_day(idx: str):
            idx = to_datetime(idx)
            return idx.hour // 6

        prefix = 'time_of_day'
        data[prefix] = data.index.map(to_time_of_day)
        one_hot = pd.get_dummies(data[prefix], prefix=prefix)
        data = data.drop(prefix, axis=1)
        data = data.join(one_hot)
        return data

    @staticmethod
    def add_time_of_week_categorical(data: pd.DataFrame) -> pd.DataFrame:
        def to_time_of_week(time: str) -> int:
            time = to_datetime(time)
            return 0 if time.weekday() < 4 else 1

        data['time_of_week'] = data.index.map(to_time_of_week)
        return data

    @staticmethod
    def add_time_of_year_categorical(data: pd.DataFrame) -> pd.DataFrame:
        def to_time_of_year(time: str) -> int:
            month = to_datetime(time).month
            return month // 4

        prefix = 'time_of_year'
        data[prefix] = data.index.map(to_time_of_year)
        one_hot = pd.get_dummies(data[prefix], prefix=prefix)
        data = data.drop(prefix, axis=1)
        data = data.join(one_hot)
        return data

    @staticmethod
    def add_seconds(data: pd.DataFrame, key: str = 'seconds') -> pd.DataFrame:
        def to_total_seconds(time: str) -> float:
            return to_datetime(time).timestamp()

        data[key] = data.index.map(to_total_seconds)
        return data

    @staticmethod
    def add_time_of_day_trigonometric(data: pd.DataFrame) -> pd.DataFrame:
        key = 'temp_seconds1'
        data = Preprocessor.add_seconds(data, key)
        data['time_of_day_sin'] = np.sin(data[key] * (2 * np.pi / get_day_seconds()))
        data['time_of_day_cos'] = np.cos(data[key] * (2 * np.pi / get_day_seconds()))
        data.drop(columns=[key], inplace=True)
        return data

    @staticmethod
    def add_time_of_year_trigonometric(data: pd.DataFrame) -> pd.DataFrame:
        key = 'temp_seconds2'
        data = Preprocessor.add_seconds(data, key)
        data['time_of_year_sin'] = np.sin(data[key] * (2 * np.pi / get_year_seconds()))
        data['time_of_year_cos'] = np.cos(data[key] * (2 * np.pi / get_year_seconds()))
        data.drop(columns=[key], inplace=True)
        return data

    @staticmethod
    def add_prev_target(data: pd.DataFrame) -> pd.DataFrame:
        data['prev_y'] = data['y'].shift(1)
        return data

    @staticmethod
    def add_prev_day_target(data: pd.DataFrame) -> pd.DataFrame:
        increment = 5
        data['prev_day_y'] = data['y'].shift(24 * 60 // increment)
        return data

    @staticmethod
    def add_rolling_mean(data: pd.DataFrame) -> pd.DataFrame:
        data['rolling_mean'] = data['y'].shift(1).rolling(6).mean()
        return data

    @staticmethod
    def pipeline(data: pd.DataFrame, preprocessing: bool = True, feature_engineering: bool = True) -> pd.DataFrame:
        if preprocessing:
            #scaler = MinMaxScaler(feature_range=(-1, 1))
            data = Preprocessor.clip(data, columns=['y'])
            data = Preprocessor.scale(data)

        if feature_engineering:
            data = Preprocessor.add_time_of_day_categorical(data)
            data = Preprocessor.add_time_of_week_categorical(data)
            data = Preprocessor.add_time_of_year_categorical(data)

            data = Preprocessor.add_time_of_day_trigonometric(data)
            data = Preprocessor.add_time_of_year_trigonometric(data)

            data = Preprocessor.add_prev_target(data)
            data = Preprocessor.add_prev_day_target(data)
            data = Preprocessor.add_rolling_mean(data)
            data.dropna(inplace=True)
        return data

