from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('Train.csv')
df.head()
print(df.shape)
df.info()
df.describe()
df.isnull().sum()
df.drop('Cabin', axis=1, inplace=True)
print(df['Sex'].unique())
print(df['Embarked'].unique())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
scaler = StandardScaler()
num_cols = ['Age', 'Fare']
df[num_cols] = scaler.fit_transform(df[num_cols])
