import numpy as np
from sklearn.externals import joblib


grid_search = joblib.load("grid_search.pkl")
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.ceil(np.sqrt(-mean_score)), params)
