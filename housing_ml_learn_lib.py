import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer, StandardScaler, Imputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.rooms_ix = 3
        self.bedrooms_ix = 4
        self.population_ix = 5
        self.household_ix = 6

    def fit(self, X, y=None):
        return self  # nothing to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.household_ix]
        population_per_household = X[:, self.population_ix] /\
            X[:, self.household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attribute_names].values


class LabelBinarizer_new(BaseEstimator, TransformerMixin):
    """It makes the fit_transform() take only 2 positional argument."""
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        encoder = LabelBinarizer()
        result = encoder.fit_transform(X)
        return result


def display_scores(scores):
    """display the scores calculated from cross-validation"""
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def train_test_split(filename):
    """load the file and create the train and test data sets"""
    # load the data file.
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
    return strat_train_set, strat_test_set


def data_transform(housing):
    """clean the data and transform the text cat into binary"""
    # Separate the text-based and number-based attributes. Text-based attribute
    # needs to be transformed via binarizer.
    # Since pipeline is going to be used, and there is a custom attribute
    # selector.
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
    return housing_prepared
