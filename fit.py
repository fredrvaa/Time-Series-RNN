import os
import datetime

import pandas as pd
import tensorflow.keras as tfk

from preprocessing.preprocessor import Preprocessor
from rnn.network import Network

sequence_length = 12*12
epochs = 100
learning_rate = 1e-3
decay = 1e-6
dropout_rate = 0.3
batch_size = 128

timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

logdir = os.path.join('logs', timestamp)
modeldir = os.path.join('models', timestamp)
tensorboard_callback = tfk.callbacks.TensorBoard(logdir, histogram_freq=1)
model_checkpoint_callback = tfk.callbacks.ModelCheckpoint(filepath=modeldir, save_freq=5)

data: pd.DataFrame = pd.read_csv('data/no1_train.csv')
data.set_index('start_time', inplace=True)

data = Preprocessor.pipeline(data)

y = data['y']
x = data.drop(columns=['y', *[f'time_of_day_{i}' for i in range(4)], 'time_of_week', 'river'])

input_shape = (sequence_length, x.shape[1])
print(f'Input shape: {input_shape}')
net = Network(input_shape=input_shape,
              output_units=1,
              hidden_units=[128, 128, 64],
              dropout_rate=dropout_rate,
              learning_rate=learning_rate,
              decay=decay)

checkpoint_filepath = '/tmp/checkpoint'

net.fit(x, y,
        epochs=epochs,
        sequence_length=sequence_length,
        batch_size=batch_size,
        callbacks=[tensorboard_callback, model_checkpoint_callback])