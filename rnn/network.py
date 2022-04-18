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
                 dropout_rate: float = 0.2,
                 learning_rate: float = 1e-3,
                 decay: float = 1e-6
                 ):

        self.sequence_length = input_shape[0]

        self.model = tfk.Sequential(layers=[
                tfkl.InputLayer(input_shape=input_shape),
                *[tfkl.LSTM(units=units, dropout=dropout_rate, return_sequences=True) for units in hidden_units[:-1]],
                tfkl.LSTM(units=hidden_units[-1], dropout=dropout_rate),
                tfkl.Dense(units=output_units, activation='linear')
            ]
        )

        self.model.compile(optimizer=tfk.optimizers.Adam(learning_rate=learning_rate, decay=decay), loss='mse')
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

    def multistep_predict(self, x: pd.DataFrame, start_idx: int = 0, forecast_steps: int = 24) -> np.ndarray:
        if start_idx > x.shape[0] - forecast_steps:
            raise ValueError(f'Start idx can max be {x.shape[0] - forecast_steps}, but {start_idx} was given')

        def get_forecast(model_input):
            forecast = self.model.predict(model_input)
            return float(forecast[0][0])

        x = x.copy()

        model_input = x[start_idx:start_idx+self.sequence_length]
        model_input = np.expand_dims(model_input, axis=0)
        forecasts = []
        forecast = get_forecast(model_input)
        forecasts.append(forecast)
        for _ in range(forecast_steps-1):
            start_idx += 1
            model_input = x[start_idx:start_idx+self.sequence_length]
            model_input.at[model_input.index[-1], 'prev_y'] = forecast
            model_input = np.expand_dims(model_input, axis=0)
            forecast = get_forecast(model_input)
            forecasts.append(forecast)

        return np.array(forecasts)
            



