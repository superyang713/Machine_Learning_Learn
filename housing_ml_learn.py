#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, \
    GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from housing_ml_learn_lib import CombinedAttributesAdder, DataFrameSelector, \
    LabelBinarizer_new, display_scores

# Load housing data
filename = 'housing.csv'
filepath = os.path.join(os.path.expanduser('~'), 'Dropbox (MIT)',
                        'Linux', 'ML_learn', filename)
housing = pd.read_csv(filepath)

# Create a new category for train test data split
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

# Perform stratified sampling based on the newly created category.
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Remove income_cat attribute that was used for stratified sampling.
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Now just works on the train data, and forget about the test data.
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Separate the text-based and number-based attributes. Text-based attribute
# needs to be transformed via binarizer.
# Since pipeline is going to be used, and there is a custom attribute selector
# created in the lib, we just need to list the attributes.
housing_num = housing.drop("ocean_proximity", axis=1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

# Set up the pipeline for the number-based attributes
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler(num_attribs)),
])

# Set up the pipeline for the text-based attributes
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', LabelBinarizer_new()),
])

# Combine the two pipelines
full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])
housing_prepared = full_pipeline.fit_transform(housing)

# Train a Linear Regression model.
# lin_reg = LinearRegression()
# lin_reg.fit(housing_prepared, housing_labels)
# housing_predictions = lin_reg.predict(housing_prepared)

# Evaluate the performance of the Linear Model
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse)

# Calculate the performance using cross-validation method.
# lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
#                              scoring="neg_mean_squared_error", cv=10)
# lin_rmse_scores = np.sqrt(-lin_scores)
# display_scores(lin_rmse_scores)

# save the model
# joblib.dump(lin_reg, "lin_reg.pkl")

# load the model
# lin_reg = joblib.load("lin_reg.pkl")

# Seach for the best combination of hyperparameter values for the model.
# param_grid = [
#     {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
#     {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
# ]
#
# # train forest_reg model.
# forest_reg = RandomForestRegressor()
# grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
#                            scoring='neg_mean_squared_error')
# grid_search.fit(housing_prepared, housing_labels)
# joblib.dump(grid_search, "grid_search.pkl")

grid_search = joblib.load("grid_search.pkl")
print(grid_search.best_params)
