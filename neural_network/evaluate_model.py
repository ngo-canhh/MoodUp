import numpy as np
import pandas as pd
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('nn_model.keras')

# Load data
data = np.load('dataset.npz', allow_pickle = True)
x_test, y_test = data['x_test'], data['y_test']
x_cv, y_cv = data['x_cv'], data['y_cv']
number_to_label = data['number_to_label']

# Predict
predict = model.predict(x_test)

# Hàm tính loss
scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

# Tính scce của test set
print(f'Test SCCE: {scce(y_test, predict)}')