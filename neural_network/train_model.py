import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, classification_report

tf.random.set_seed(42)

# Load lại dataset
data = np.load('dataset.npz')
x_train = data['x']
y_train = data['y'].reshape(-1)
print(data['x'].shape)
print(x_train.shape, y_train.shape)
print(np.unique(y_train, return_counts=True))

# model
model = Sequential(
    [
        tf.keras.Input(shape=(14,)),
        Dense(32, activation= 'relu'),
        BatchNormalization(),
        Dense(64, activation= 'relu'),
        Dense(32, activation= 'relu'),
        Dense(4, activation= 'linear'),
    ], name = "nn_model"
)

model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Train, sử dụng early stop
#early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)
history = model.fit(x_train, y_train, epochs=100, batch_size=2, validation_split=0.2)


import matplotlib.pyplot as plt

# # Vẽ biểu đồ loss
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.legend()
# plt.show()

best = tf.keras.models.load_model('best_model.keras')
# Dự đoán của model
y_pred = np.argmax(tf.nn.softmax(model.predict(x_train)).numpy(), axis=1)

# Tính accuracy
acc = accuracy_score(y_train, y_pred)
print(f"Accuracy: {acc:.2f}")
print(classification_report(y_train, y_pred))


# Lưu lại dùng để đánh giá
model.save('nn_model.keras')