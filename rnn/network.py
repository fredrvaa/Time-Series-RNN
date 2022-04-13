from typing import Optional

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

    def multistep_predict(self, ):
        pass

