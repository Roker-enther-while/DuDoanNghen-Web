import numpy as np

from src.training.data_loader import select_indices


def test_head_sampling():
    np.testing.assert_array_equal(select_indices(10, 3, "head"), np.array([0, 1, 2]))


def test_full_sampling():
    np.testing.assert_array_equal(select_indices(5, 2, "full"), np.arange(5))


def test_evenly_spaced_sampling():
    idx = select_indices(10, 5, "evenly_spaced")
    assert len(idx) == 5
    assert idx[0] == 0
    assert idx[-1] == 9
