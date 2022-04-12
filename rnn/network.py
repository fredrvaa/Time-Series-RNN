import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl


class Network:
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_sizes: list[int],
                 dropout_rate: float = 1e-1,
                 learning_rate: float = 1e-3,
                 ):

        self.model = tfk.Sequential(
            tfkl.InputLayer(input_shape=(input_size, )),
            *[tfkl.LSTM(units=size, dropout=dropout_rate, return_sequences=True) for size in hidden_sizes],
            tfkl.Dense(units=output_size, dropout=dropout_rate)
        )

        self.model.compile(optimizer=tfk.optimizers.Adam(learning_rate=learning_rate))
        self.model.summary()

