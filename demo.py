import gradio as gr
import torch
import torch.nn as nn
import pandas as pd
import random
random.seed(42)
import re
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
def extract_sleep_hours(value):
    if pd.isna(value):
        return None
    match = re.findall(r'\d+', value)
    if len(match) == 1 and float(match[0]) <= 24:
        return float(match[0])
    elif len(match) == 2 and (float(match[0]) + float(match[1])) / 2 <= 24:
        return (float(match[0]) + float(match[1])) / 2
    else:
        return None
def map_dietary_habits(value):
    if value in ['Healthy', 'More Healthy', 'Yes']:
        return 'Healthy'
    elif value in ['Unhealthy', 'No Healthy', 'Less Healthy']:
        return 'Unhealthy'
    elif value in ['Moderate']:
        return 'Moderate'
    else:
        return None

df = pd.read_csv('datasets/final_dataset/train.csv')
def process_input_test(df, test):
    test += [1]
    test = [150000, 'TEST'] + test
    df.loc[len(df)] = test
    # Điều kiện lọc
    condition = (
        (df['Working Professional or Student'] == 'Student') &
        (df['Academic Pressure'].notna()) &
        (df['Study Satisfaction'].notna()) &
        (df['CGPA'].notna())
    )

    # Cập nhật giá trị cho các hàng thỏa mãn điều kiện
    df.loc[condition, 'Profession'] = 'Student'
    df.loc[condition, 'Work Pressure'] = 0.0
    df.loc[condition, 'Job Satisfaction'] = 0.0

    # Điều kiện lọc
    condition = (
        (df['Working Professional or Student'] == 'Working Professional') &
        (df['Work Pressure'].notna()) &
        (df['Job Satisfaction'].notna())
    )

    # Cập nhật giá trị cho các hàng thỏa mãn điều kiện
    df.loc[condition, 'Academic Pressure'] = 0.0
    df.loc[condition, 'CGPA'] = -1
    df.loc[condition, 'Study Satisfaction'] = 0.0

    # Điều kiện lọc
    condition = (
        (df['Profession'].isna()) &
        (df['Degree'] == 'Class 12') &
        (df['Working Professional or Student'] == 'Working Professional')
    )

    # Cập nhật giá trị cho các hàng thỏa mãn điều kiện
    df.loc[condition, 'Profession'] = 'Other'

    info_profession = df['Profession'].value_counts()
    list_profession = []
    for index, value in info_profession.items():
      if index not in ['Student', 'Other']:
        for i in range(value):
          list_profession.append(index)
    df['Profession'] = df['Profession'].apply(
        lambda x: x if pd.notnull(x) else random.choice(list_profession)
    )

    df['Sleep Duration'] = df['Sleep Duration'].apply(extract_sleep_hours)
    df['Dietary Habits'] = df['Dietary Habits'].apply(map_dietary_habits)

    x = df.drop(columns=['Depression', 'id', 'Name'])
    test_df = x.loc[len(x)-1]
    X = x[:len(x)-1]