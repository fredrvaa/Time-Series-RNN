import math
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from preprocessing.utils import to_datetime


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
    def scale(data: pd.DataFrame, scaler=StandardScaler):
        data[:] = scaler().fit_transform(data)
        return data

    #######################
    # FEATURE ENGINEERING #
    #######################

    @staticmethod
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

    @staticmethod
    def add_time_of_week(data: pd.DataFrame) -> pd.DataFrame:
        def to_time_of_week(time: str) -> int:
            time = to_datetime(time)
            return 0 if time.weekday() < 4 else 1

        data['time_of_week'] = data.index.map(to_time_of_week)
        return data

    @staticmethod
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

    @staticmethod
    def pipeline(data: pd.DataFrame, preprocessing: bool = True, feature_engineering: bool = True) -> pd.DataFrame:
        if preprocessing:
            data = Preprocessor.clip(data)
            data = Preprocessor.scale(data)
        if feature_engineering:
            data = Preprocessor.add_time_of_day(data)
            data = Preprocessor.add_time_of_week(data)
            data = Preprocessor.add_time_of_year(data)
        return data

