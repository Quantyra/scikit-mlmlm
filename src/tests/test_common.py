import pytest

from sklearn.utils.estimator_checks import check_estimator

from src.sklearn_mlmlm.classifiers import MultiLabelMLMClassifier

import numpy as np

grid_temp = np.arange(3, 6.05, 0.05)
p_grid = 2**grid_temp

# Unfortunately https://github.com/scikit-learn/scikit-learn/blob/7f9bad99d/sklearn/multioutput.py#L440
# Shows that the estimator checks in SKLearn are currently broken for MultiOutput estimators.
@pytest.mark.parametrize("estimator", [MultiLabelMLMClassifier(p_grid)])
def test_all_estimators(estimator):
    return check_estimator(estimator)
