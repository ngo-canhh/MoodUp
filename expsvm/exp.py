import pandas as pd
from sklearn import model_selection, preprocessing, svm
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

class_name = {
    0: 'Health',
    1: 'Depression'
}
data = "../datasets/final_dataset"

train_data = pd.read_csv(os.path.join(data, 'train.csv')).head(2000)
test_data = pd.read_csv(os.path.join(data, 'test.csv'))

print(train_data.__len__())

train_data = train_data.rename(columns={'Working Professional or Student': 'Worker/Student', 'Have you ever had suicidal thoughts ?': 'Suicide Attemp','Family History of Mental Illness': 'Mental illness'})
test_data = test_data.rename(columns={'Working Professional or Student': 'Worker/Student', 'Have you ever had suicidal thoughts ?': 'Suicide Attemp','Family History of Mental Illness': 'Mental illness'})
train_data.columns = train_data.columns.str.strip()
test_data.columns = test_data.columns.str.strip()
print(train_data.columns)

x = train_data.drop(['Depression', 'id', 'Name'], axis=1)
y = train_data['Depression']

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)
print(x.columns)
print(x_train.shape, x_test.shape)

count_labels = Counter(y_test)
cls_name = [class_name[id] for id in list(count_labels.keys())]
cls_num = list(count_labels.values())
plt.figure(figsize=(4, 8))
plt.bar(cls_name, cls_num, color='skyblue')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.xticks(cls_name)
for i, cnt in enumerate(cls_num):
    plt.text(cls_name[i], cnt+0.1, str(cnt), ha='center', va='bottom')
plt.show()

# t = preprocessing.OrdinalEncoder().fit_transform(x['City'].to_frame())
# print(t[:20].astype(int))
# Normalize number_data and Encode str_data
x_train_num_columns = x_train.select_dtypes(include=['int64', 'float64']).columns
x_train_str_columns = x_train.select_dtypes(include=['object']).columns
x_test_num_columns = x_test.select_dtypes(include=['int64', 'float64']).columns
x_test_str_columns = x_test.select_dtypes(include=['object']).columns
print(x_train_num_columns)
num_pipline = Pipeline([
    ('scaler', preprocessing.StandardScaler())
])
str_pipline = Pipeline([
    ('ordinal', preprocessing.OrdinalEncoder())
])
preprocessor = ColumnTransformer([
    ('num', num_pipline, x_train_num_columns),
    ('str', str_pipline, x_train_str_columns)
])
x_train_processed = preprocessor.fit_transform(x_train)
x_test_processed = preprocessor.fit_transform(x_test)

#process NaN
NaN_pipline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])
x_train_processed = NaN_pipline.fit_transform(x_train_processed)
x_test_processed = NaN_pipline.fit_transform(x_test_processed)



cls = svm.SVC()
cls.fit(x_train_processed, y_train)

y_predict = cls.predict(x_test_processed)
cnt = 0
for yp, yt in zip(y_predict, y_test):
    print("Predict: {}, Actual: {}".format(yp, yt))
    cnt += 1
    if cnt == 10:
        break
print(classification_report(y_test, y_predict))


'''data.columns = data.columns.str.strip()
print(data.columns)

x = data.drop('DepressionStatus', axis = 1)
y = data['DepressionStatus']
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)
print(y_train.shape, y_test.shape)
#t = preprocessing.OrdinalEncoder().fit_transform(x['Gender'].to_frame())
#print(t.astype(int))
#preprocessing
#x_train
x_train_num_columns = x_train.select_dtypes(include = ['int64', 'float64']).columns
x_train_str_columns = x_train.select_dtypes(include = ['object']).columns
#x_test
x_test_num_columns = x_test.select_dtypes(include = ['int64', 'float64']).columns
x_test_str_columns = x_test.select_dtypes(include = ['object']).columns
num_pipline = Pipeline([
    ('scaler', preprocessing.StandardScaler())
])
str_pipline = Pipeline([
    ('ordinal', preprocessing.OrdinalEncoder())
])
preprocessor = ColumnTransformer([
    ('num', num_pipline, x_train_num_columns),
    ('str', str_pipline, x_train_str_columns)
])
x_train_processed = preprocessor.fit_transform(x_train)
x_test_processed = preprocessor.fit_transform(x_test)

y_train_processed = preprocessing.OrdinalEncoder().fit_transform(y_train.to_frame()).flatten()
y_test_processed = preprocessing.OrdinalEncoder().fit_transform(y_test.to_frame()).flatten()

#process NaN
NaN_pipline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])
x_train_processed = NaN_pipline.fit_transform(x_train_processed)
x_test_processed = NaN_pipline.fit_transform(x_test_processed)'''
'''
for name in x_train.columns:
    if data[name][1] == str(data[name][1]):
        print("text")
        scaler = preprocessing.OrdinalEncoder()
        x_train[name] = scaler.fit_transform(x_train[name])
    else:
        print("int")
        scaler = preprocessing.StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
x_test = preprocessing.OrdinalEncoder().fit_transform(x_test)
.....'''
'''cls = svm.SVC()
cls.fit(x_train_processed, y_train_processed)

y_predict = cls.predict(x_test_processed)
for yp, yt in zip(y_predict, y_test_processed):
    print("Predict: {}, Actual: {}".format(yp, yt))
print(classification_report(y_test_processed, y_predict))'''

P = TP/(TP+FP) = 0.16
#tỉ lệ dự đoán chính xác với class positive
R = TP/(TP+FN) = 1
#độ phủ của class positive
F1 = 2*P*R/(P+R)