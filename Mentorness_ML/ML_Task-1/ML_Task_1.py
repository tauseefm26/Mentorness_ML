# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 02:44:12 2024

@author: mohdt
"""

# Import Libraries
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load Data
data = {
    'feature1': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'feature2': ['A', 'B', 'B', 'A', np.nan, 'C', 'C', 'A', 'B', 'C', 'A', 'C', 'B', 'A', 'B'],
    'target': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Handle Missing Values
df['feature1'].fillna(df['feature1'].mean(), inplace=True)

# Encode Categorical Features
label_encoder = LabelEncoder()
df['feature2'] = label_encoder.fit_transform(df['feature2'].astype(str))

# Prepare Data for XGBoost
X = df[['feature1', 'feature2']]
y = df['target']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'eta': [0.01, 0.1, 0.2],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}

# Initialize XGBoost Classifier
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

# Grid Search
grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)
grid_search.fit(X_train, y_train)

# Best Parameters
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Define num_boost_round
num_boost_round = 100

# Train with Best Parameters
best_params_for_train = {
    'objective': 'binary:logistic',
    'max_depth': best_params['max_depth'],
    'eta': best_params['eta'],
    'min_child_weight': best_params['min_child_weight'],
    'subsample': best_params['subsample'],
    'colsample_bytree': best_params['colsample_bytree'],
    'eval_metric': 'logloss'
}

best_model = xgb.train(best_params_for_train, dtrain, num_boost_round=num_boost_round)

# Predictions
preds = best_model.predict(dtest)
predictions = [1 if pred > 0.5 else 0 for pred in preds]

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Feature Importance
try:
    xgb.plot_importance(best_model)
    plt.show()
except ValueError as e:
    print("Error plotting feature importance:", e)

