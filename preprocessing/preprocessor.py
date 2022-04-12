import math
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    @staticmethod
    def clip(data: pd.DataFrame,
             clip_low: float = .05,
             clip_high: float = .05,
             q_low: float = .25,
             q_high: float = .75,
             columns: Optional[list[str]] = None) -> pd.DataFrame:

        n_low = int(data.shape[0] * clip_low)
        n_high = int(data.shape[0] * clip_high)

        if columns is None:
            columns = data.columns

        for column in columns:
            low, high = data[column].quantile([q_low, q_high])

            data[column][data.nsmallest(n_low, columns=[column]).index] = low
            data[column][data.nlargest(n_high, columns=[column]).index] = high

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
