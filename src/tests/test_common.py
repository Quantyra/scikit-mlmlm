import pytest

from sklearn.utils.estimator_checks import check_estimator

from src import TemplateEstimator
from src import TemplateClassifier
from src import TemplateTransformer


@pytest.mark.parametrize(
    "estimator",
    [TemplateEstimator(), TemplateTransformer(), TemplateClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
