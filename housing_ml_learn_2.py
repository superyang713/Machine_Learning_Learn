import numpy as np
from sklearn.externals import joblib


grid_search = joblib.load("grid_search.pkl")
cvres = grid_search.cv_results_

# Show the best set of estimators and the list of scores with different
# estimator combinations.
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.ceil(np.sqrt(-mean_score)), params)

# Gain insight by inspecting the best models. Find the importance of each
# attribute (feature) for making accurate predictions.
feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attibs = list(encoder.classes_)

