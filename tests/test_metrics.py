from soilnet.evaluation import classification_metrics, regression_metrics


def test_regression_metrics_are_separate_and_scaled_by_caller():
    result = regression_metrics([0.0, 100.0], [10.0, 90.0])
    assert result["rmse"] == 10.0
    assert result["mae"] == 10.0
    assert result["me"] == 0.0


def test_classification_metrics():
    result = classification_metrics([0, 1, 1], [0, 1, 0])
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["macro_f1"] <= 1.0
