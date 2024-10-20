import pandas as pd
from sklearn import model_selection, preprocessing, svm
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

data = pd.read_csv("../datasets/CSE_student_performances.csv")
data.columns = data.columns.str.strip()
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
x_test_processed = NaN_pipline.fit_transform(x_test_processed)
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
'''
cls = svm.SVC()
cls.fit(x_train_processed, y_train_processed)

y_predict = cls.predict(x_test_processed)
for yp, yt in zip(y_predict, y_test_processed):
    print("Predict: {}, Actual: {}".format(yp, yt))
print(classification_report(y_test_processed, y_predict))
print(y_test, y_test_processed)
