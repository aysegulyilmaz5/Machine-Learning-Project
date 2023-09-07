import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import warnings

# Ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Regression Analysis
reg_data = pd.read_csv('4.csv')

print(reg_data.head())
print(reg_data.isnull().sum())

x_reg = reg_data.iloc[:, :-1].values
y_reg = reg_data.iloc[:, -1].values

x_reg_train, x_reg_test, y_reg_train, y_reg_test = train_test_split(x_reg, y_reg, test_size=0.2, random_state=5)

reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('regression', LinearRegression())  
])

reg_pipelinexgb = Pipeline([
    ('scaler', StandardScaler()),  
    ('model', xgb.XGBRegressor())  
])

param_grid_xgb = {
    'model__max_depth': [3, 4, 5, 6, 7, 8],
    'model__min_child_weight': [1, 5, 10],
    'model__gamma': [0.5, 1, 1.5, 2, 5],
    'model__colsample_bytree': [0.6, 0.8, 1.0],
    'model__learning_rate': [0.01, 0.1, 0.2, 0.3]
}

grid = GridSearchCV(estimator=reg_pipelinexgb, param_grid=param_grid_xgb, cv=3, scoring='neg_mean_squared_error', n_jobs=1)
grid.fit(x_reg_train, y_reg_train)
best_model = grid.best_estimator_
y_reg_pred_xgb = best_model.predict(x_reg_test)

reg_pipeline.fit(x_reg_train, y_reg_train)
y_reg_pred_linear = reg_pipeline.predict(x_reg_test)

mse_lr = mean_squared_error(y_reg_test, y_reg_pred_linear)
mse_xgb = mean_squared_error(y_reg_test, y_reg_pred_xgb)

r2_lr = r2_score(y_reg_test, y_reg_pred_linear)
r2_xgb = r2_score(y_reg_test, y_reg_pred_xgb)

print("Results:")
print("XGBoost MSE:", mse_xgb)
print("XGBoost R^2:", r2_xgb)
print("Linear Regression MSE:", mse_lr)
print("Linear Regression R^2:", r2_lr)


