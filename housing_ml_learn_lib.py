import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer


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
