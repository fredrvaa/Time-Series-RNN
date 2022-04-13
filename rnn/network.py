from typing import Optional

import numpy as np
import pandas as pd
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


class Network:
    def __init__(self,
                 input_shape: tuple,
                 output_units: int,
                 hidden_units: list[int],
                 dropout_rate: float = 1e-1,
                 learning_rate: float = 1e-3,
                 ):

        self.sequence_length = input_shape[0]

        self.model = tfk.Sequential(layers=[
                tfkl.InputLayer(input_shape=input_shape),
                *[tfkl.LSTM(units=units, dropout=dropout_rate, return_sequences=True) for units in hidden_units[:-2]],
                tfkl.LSTM(units=hidden_units[-2], dropout=dropout_rate),
                tfkl.Dense(units=hidden_units[-1], activation='relu'),
                tfkl.Dense(units=output_units, activation=None)
            ]
        )

        self.model.compile(optimizer=tfk.optimizers.Adam(learning_rate), loss='mse')
        self.model.summary()

    def fit(self,
            x: pd.DataFrame,
            y: pd.DataFrame,
            train_ratio: float = 0.7,
            epochs: int = 10,
            sequence_length: int = 144,
            batch_size: int = 256,
            callbacks: Optional[list] = None
            ):

        end_index = int(x.shape[0] * train_ratio)
        train_gen = TimeseriesGenerator(x, y, length=sequence_length, batch_size=batch_size, end_index=end_index)
        val_gen = TimeseriesGenerator(x, y, length=sequence_length, batch_size=batch_size, start_index=end_index + 1)

        self.model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)

    def multistep_predict(self, x: pd.DataFrame, forecast_start: int = 0, forecast_steps: int = 24) -> np.ndarray:
        if forecast_start < self.sequence_length:
            forecast_start = self.sequence_length
            print(f'Forecast start was set to {self.sequence_length}')
        if forecast_start > x.shape[0] - forecast_steps:
            forecast_start = x.shape[0] - forecast_steps
            print(f'Forecast start was set to {x.shape[0] - forecast_steps}')

        model_input = x[forecast_start:forecast_start+self.sequence_length]
        model_input = np.expand_dims(model_input, axis=0)
        forecasts = []
        forecast = float(self.model(model_input)[0][0])
        forecasts.append(forecast)
        for _ in range(self.sequence_length-1):
            forecast_start += 1
            model_input = x[forecast_start:forecast_start+self.sequence_length]
            # Replace prev_y with prev forecast
            model_input.at[model_input.index[self.sequence_length - 1], 'prev_y'] = forecast
            model_input = np.expand_dims(model_input, axis=0)
            forecast = float(self.model(model_input)[0][0])
            forecasts.append(forecast)

        return np.array(forecasts)



