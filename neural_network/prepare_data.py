import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Nếu đã split data rồi thì đừng làm lại nữa
if (os.path.exists('dataset.npz')):
    print('Dataset has been splited and saved')
    exit(0)

df = pd.read_csv('../datasets/Deepression.csv')

# Chuyển từ df thành ndarray
x = df.drop(['Number ', 'Depression State'], axis= 1).values
y = df['Depression State']

unique_labels = y.unique()
label_to_number = {label: idx for idx, label in enumerate(unique_labels)}
number_to_label = {v: k for k, v in label_to_number.items()}
y = y.map(label_to_number).values

# Tách dataset
x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.60, random_state=13)
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=10)

# Lưu lại dataset
np.savez('dataset.npz', 
         x_train=x_train, y_train=y_train, 
         x_test=x_test, y_test=y_test, 
         x_cv=x_cv, y_cv=y_cv,
         number_to_label = number_to_label)
