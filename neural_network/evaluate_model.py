import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse

model = tf.keras.models.load_model('nn_model.keras')

data = np.load('dataset.npz', allow_pickle = True)
x_test, y_test = data['x_test'], data['y_test']
x_cv, y_cv = data['x_cv'], data['y_cv']
number_to_label = data['number_to_label']
yhat = np.argmax(tf.nn.softmax(model.predict(x_test)).numpy(), axis= 1)

print(f'Test MSE: {mse(y_test, yhat) / 2}')