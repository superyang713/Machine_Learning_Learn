#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from housing_ml_learn_lib import train_test_split, data_transform
from parameters import Parameter

# Load housing data and separate train and test data.
filename = 'housing.csv'
strat_train_set, strat_test_set = train_test_split(filename)

# Separate features and labels in the training data.
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Prepare the features by cleaning and transforming data so that it is ready to
# be fed to the model.
housing_prepared = data_transform(housing)

# Train forest_reg model and fine tune the model.
# Save the grid_search class into a pickle (external file).
forest_reg = RandomForestRegressor()
parameter = Parameter()
grid_search = GridSearchCV(forest_reg, parameter.grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)
joblib.dump(grid_search, "grid_search.pkl")
print('job done')
