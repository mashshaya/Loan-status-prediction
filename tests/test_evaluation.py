import numpy as np

from loan_status_prediction.evaluation import BusinessCosts, business_cost, predict_with_threshold


def test_predict_with_threshold_uses_positive_class_cutoff():
    predictions = predict_with_threshold(np.array([0.2, 0.5, 0.8]), threshold=0.5)

    assert predictions.tolist() == [0, 1, 1]


def test_business_cost_weights_false_positives_and_false_negatives():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    cost = business_cost(y_true, y_pred, BusinessCosts(false_positive=5.0, false_negative=2.0))

    assert cost == 7.0
