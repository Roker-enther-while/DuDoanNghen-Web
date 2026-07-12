import numpy as np

from src.models.moving_average import MovingAverageBaseline
from src.models.naive import NaiveLastValueBaseline


def test_naive_predict_shape_range_no_nan():
    X = np.random.default_rng(1).random((5, 4, 3), dtype=np.float32)
    y = np.random.default_rng(2).random(5, dtype=np.float32)
    model = NaiveLastValueBaseline().fit(X, y, ["a", "b", "congestion_score_proxy"])
    pred = model.predict(X)
    assert pred.shape == (5,)
    assert not np.isnan(pred).any()
    assert pred.min() >= 0 and pred.max() <= 1


def test_moving_average_predict_shape_range_no_nan():
    X = np.random.default_rng(3).random((5, 4, 3), dtype=np.float32)
    y = np.random.default_rng(4).random(5, dtype=np.float32)
    model = MovingAverageBaseline(average_steps=2).fit(X, y, ["a", "b", "congestion_score_proxy"])
    pred = model.predict(X)
    assert pred.shape == (5,)
    assert not np.isnan(pred).any()
    assert pred.min() >= 0 and pred.max() <= 1
