import pandas as pd
from sklearn import model_selection, preprocessing, svm
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import os
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE

def map_dietary_habits(value):
    if value in ['Healthy', 'More Healthy', 'Yes', 'Class 12', 'Indoor', 'Male', 'BSc', 'Pratham']:
        return 'Healthy'
    elif value in ['Unhealthy', 'No Healthy', 'Less Healthy', 'Hormonal', 'Vegas', 'Electrician']:
        return 'Unhealthy'
    elif value in ['Moderate', '3']:
        return 'Moderate'
    else:
        return 'Moderate'
def extract_sleep_hours(value):
    match = re.findall(r'\d+', value)
    if len(match) == 1:
        return float(match[0])
    elif len(match) == 2:
        return (float(match[0]) + float(match[1])) / 2
    else:
        return None

class_name = {
    0: 'Health',
    1: 'Depression'
}
data = "./datasets"

train_data = pd.read_csv(os.path.join(data, 'train.csv'))[:10000]

# print(train_data.__len__())

train_data = train_data.rename(columns={'Working Professional or Student': 'Worker/Student', 'Have you ever had suicidal thoughts ?': 'Suicide Attemp','Family History of Mental Illness': 'Mental illness'})
train_data.columns = train_data.columns.str.strip()
# print(train_data.columns)

x = train_data.drop(['Depression', 'id', 'Name'], axis=1)
y = train_data['Depression']

x.loc[x['Worker/Student'] == 'Working Professional', 'Academic Pressure'] = 0
x.loc[x['Worker/Student'] == 'Student', 'Work Pressure'] = 0
x.loc[x['Worker/Student'] == 'Student', 'Profession'] = 'Student'
professions = ['Content Writer', 'Architect', 'Consultant', 'HR Manager']

null_rows = x[x['Profession'].isnull()]
num_nulls = len(null_rows)
random_professions = np.random.choice(professions, size=num_nulls)
x.loc[x['Profession'].isnull(), 'Profession'] = random_professions

x.loc[(x['Profession'] == 'Student') &
             (x['Job Satisfaction'].isnull()), 'Job Satisfaction'] = 0
x.loc[x['Worker/Student'] != 'Student', 'CGPA'] = 0
x.loc[x['Worker/Student'] != 'Student', 'Study Satisfaction'] = 0
# x.loc[x['Academic Pressure'].isnull(), 'Academic Pressure'] = 3.0
# x = x.drop(columns=['CGPA', 'Study Satisfaction'])

print('----------------------------------------------------------')
# print(x.isnull().sum())

# x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)
print(x.columns)

print('**********************')
print(x['Sleep Duration'].value_counts())
print('-*-')
x['Sleep Duration'] = x['Sleep Duration'].apply(extract_sleep_hours)
print(x['Sleep Duration'].value_counts())
print('**********************')

print('**********************')
print(x['Dietary Habits'].value_counts())
print('-*-')
# print(x['Dietary Habits'].value_counts())
x['Dietary Habits'] = x['Dietary Habits'].apply(map_dietary_habits)
print(x['Dietary Habits'].value_counts())
print('**********************')


print(x['Degree'].value_counts())

print(x.isnull().sum())

#### Test data
test_data = pd.read_csv(os.path.join(data, 'train.csv'))[100000:]
test_data = test_data.rename(columns={'Working Professional or Student': 'Worker/Student', 'Have you ever had suicidal thoughts ?': 'Suicide Attemp','Family History of Mental Illness': 'Mental illness'})
test_data.columns = test_data.columns.str.strip()

test_data.loc[test_data['Worker/Student'] == 'Working Professional', 'Academic Pressure'] = 0
test_data.loc[test_data['Worker/Student'] == 'Student', 'Work Pressure'] = 0
test_data.loc[test_data['Worker/Student'] == 'Student', 'Profession'] = 'Student'
professions = ['Content Writer', 'Architect', 'Consultant', 'HR Manager']

test_data.loc[(test_data['Profession'] == 'Student') &
             (test_data['Job Satisfaction'].isnull()), 'Job Satisfaction'] = 0
test_data.loc[test_data['Worker/Student'] != 'Student', 'CGPA'] = 0
test_data.loc[test_data['Worker/Student'] != 'Student', 'Study Satisfaction'] = 0

test_data['Sleep Duration'] = test_data['Sleep Duration'].apply(extract_sleep_hours)
test_data['Dietary Habits'] = test_data['Dietary Habits'].apply(map_dietary_habits)

for column in test_data.columns:
    test_data = test_data[test_data[column].notnull()]
x_test = test_data.drop(['Depression', 'id', 'Name'], axis=1)
y_test = test_data['Depression']


x_train, y_train = x, y
print(test_data.isnull().sum())
print('Train: {}, Test: {}'.format(len(y), len(y_test)))
#
# count_labels = Counter(y_train)
# cls_name = [class_name[id] for id in list(count_labels.keys())]
# cls_num = list(count_labels.values())
# plt.figure(figsize=(4, 8))
# plt.bar(cls_name, cls_num, color='skyblue')
# plt.xlabel('Class')
# plt.ylabel('Count')
# plt.title('Class Distribution')
# plt.xticks(cls_name)
# for i, cnt in enumerate(cls_num):
#     plt.text(cls_name[i], cnt+0.1, str(cnt), ha='center', va='bottom')
# plt.show()
#
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

cls = svm.SVC()
smote = SMOTE()
x_train_processed, y_train = smote.fit_resample(x_train_processed, y_train)
cls.fit(x_train_processed, y_train)

y_predict = cls.predict(x_test_processed)
cnt = 0
# for yp, yt in zip(y_predict, y_test):
#     print("Predict: {}, Actual: {}".format(yp, yt))
#     cnt += 1
#     if cnt == 10:
#         break
print(classification_report(y_test, y_predict))
#
# P = TP/(TP+FP) = 0.16
# #tỉ lệ dự đoán chính xác với class positive
# R = TP/(TP+FN) = 1
# #độ phủ của class positive
# F1 = 2*P*R/(P+R)