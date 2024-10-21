import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# Load lại dataset
data = np.load('dataset.npz')
x_train = data['x']
y_train = data['y']
print(data['x'].shape)
print(x_train.shape)

# model
model = Sequential(
    [
        tf.keras.Input(shape=(14,)),
        Dense(32, activation= 'relu'),
        Dense(16, activation= 'relu'),
        Dense(4, activation= 'linear'),
    ], name = "nn_model"
)

model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
)

# Train, sử dụng early stop
# early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
model.fit(x_train, y_train, epochs=50, batch_size=16)

print(np.mean(y_train == np.argmax(tf.nn.softmax(model.predict(x_train)).numpy(), axis = 1).reshape(-1, 1)))

# Lưu lại dùng để đánh giá
model.save('nn_model.keras')