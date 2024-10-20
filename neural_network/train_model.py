import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = np.load('dataset.npz')
x_train = data['x_train']
y_train = data['y_train']

model = Sequential(
    [
        tf.keras.Input(shape=(14,)),
        Dense(16, activation= 'relu'),
        Dense(16, activation= 'relu'),
        Dense(4, activation= 'linear'),
    ], name = "nn_model"
)

model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)

model.fit(x_train, y_train, epochs= 40)

model.save('nn_model.keras')