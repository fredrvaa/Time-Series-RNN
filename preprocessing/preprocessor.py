import math
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessor:
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

    @staticmethod
    def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
        data = Preprocessor.clip(data)
        data = Preprocessor.scale(data)
        return data
