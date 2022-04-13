import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl


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

