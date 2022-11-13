import os
path = "/Users/lj/ML_Python/HousePrice"

from missing_data import df_final
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression,Lasso,Ridge,BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import math
from IPython.display import Image
import warnings
warnings.filterwarnings("ignore")
sns.set(rc={"figure.figsize": (20, 15)})
sns.set_style("whitegrid")

## Divide the dataset into train and test

train_df = df_final[df_final["SalePrice"].notnull()]
test_df = df_final[df_final["SalePrice"].isnull()]

print(train_df.shape)
print(test_df.shape)

## Split the dataset into X and Y
X_train = train_df.drop(["SalePrice","LogSalePrice"],axis = 1)
y_train = train_df["LogSalePrice"]
X_test = test_df.drop(["SalePrice","LogSalePrice"],axis = 1)

## Scale the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## Cross validation to find the best predicting model
model = {
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'XGB': XGBRegressor(),
    'LGBM': LGBMRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Bayesian Ridge': BayesianRidge()
}

df_result = pd.DataFrame(columns=["Model_name", "RMSE"])

for name, mod in model.items():
    cross_val = cross_validate(mod, X=X_train, y=y_train, cv=10, scoring=(['neg_root_mean_squared_error']))

    df_result = df_result.append(
        {'Model_name': name, "RMSE": np.abs(cross_val['test_neg_root_mean_squared_error']).mean()}, ignore_index=True)

df_result = df_result.sort_values('RMSE', ascending=True)

print(df_result)

#We will choose Gradient Boosting and LGBM as it gave the best result for hyperparameter tuning.
# 1. GradientBoostingRegressor
gb = GradientBoostingRegressor()
params_gb = {
    'loss': ('squared_error', 'absolute_error', 'huber'),
    'learning_rate': (1.0, 0.1, 0.01),
    'n_estimators': (100, 200, 300)
}

mod_gb = GridSearchCV(gb, params_gb, cv=10)
mod_gb.fit(X_train, y_train)
print('Best_hyperparameter : ', mod_gb.best_params_)

pred_gb = mod_gb.predict(X_train)
print(f'RMSE : {mean_squared_error(y_train, pred_gb, squared=False)}')

# 2. LGBMRegressor
lgbm = LGBMRegressor()
params_lgbm = {
    'num_leaves' : (11, 31, 41),
    'learning_rate' : (0.5, 0.1, 0.05),
    'n_estimators' : (100, 200, 300)
}

mod_lgbm = GridSearchCV(lgbm, params_lgbm, cv=10)
mod_lgbm.fit(X_train, y_train)
print('Best_hyperparameter : ', mod_lgbm.best_params_)

pred_lgbm = mod_lgbm.predict(X_train)
print(f'RMSE : {mean_squared_error(y_train, pred_lgbm, squared=False)}')

## Predict the test data and inverse log
y_pred = mod_gb.predict(X_test)
y_pred_inv = 10 ** y_pred

submission = pd.read_csv(path+"/data/sample_submission.csv")
submission['SalePrice'] = y_pred_inv
submission.to_csv(path+'/data/final_submission.csv', index=False)
