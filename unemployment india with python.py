# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 15:46:31 2024

@author: jesir
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data=pd.read_csv("C:/Users/jesir/OneDrive/Documents/Unemployment in India.csv")
data
data.info()
data.describe()
data.dropna(inplace=True)
data.isnull().sum()
data.columns
pd.DataFrame(data.iloc[:,3])
sns.histplot(data.iloc[:,3], kde=True)
plt.title('Distribution of Unemployment Rate')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Frequency')
plt.show()
data.drop(['Region', ' Frequency','Area'], axis=1, inplace=True)
data[' Date'] = pd.to_datetime(data[' Date'])
data.info()
data.set_index(' Date', inplace=True)
sns.histplot(data[' Estimated Unemployment Rate (%)'], kde=True)
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.show()
X = data.drop(' Estimated Unemployment Rate (%)', axis=1)
y = data[' Estimated Unemployment Rate (%)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)
print("Training MSE:", mean_squared_error(y_train, train_preds))
print("Testing MSE:", mean_squared_error(y_test, test_preds))
print("Training R^2:", r2_score(y_train, train_preds))
print("Testing R^2:", r2_score(y_test, test_preds))