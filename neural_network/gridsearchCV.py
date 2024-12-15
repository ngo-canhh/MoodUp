import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV 
from scikeras.wrappers import KerasClassifier 
from sklearn.metrics import make_scorer


data = np.load('dataset.npz')
x = data['x']
y = data['y']

# print(x.shape, y.shape)
# print(x[0], y[0])


def build_model(unit1=8, unit2= 8):
    model = Sequential(
        [
            tf.keras.Input(shape=(14,)),
            Dense(unit1, activation= 'relu'),
            Dense(unit2, activation= 'relu'),
            Dense(4, activation= 'linear'),
        ]
    )

    model.compile(
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    )
    return model

params = {'batch_size': [20, 32],
          'epochs': [50],
          'model__unit1': [128, 256, 512],
          'model__unit2': [128, 256, 512]}

model = KerasClassifier(model= build_model, verbose= 0)

def custom_score(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred) 
    return accuracy

custom_score = make_scorer(custom_score)

gs = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=-1, scoring = custom_score)

gs.fit(x, y)

print("Best Parameters:", gs.best_params_)
print("Best Score:", gs.best_score_)
