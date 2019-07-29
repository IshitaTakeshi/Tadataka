from autograd import numpy as np
from numpy.testing import assert_array_equal
import pytest

from vitamine.observations import Observations


def test_observations():
    observations = Observations(
        np.array([
            [0, 0, 2],
            [1, 2, 3],
            [2, 4, 6],
            [3, 6, 9]
        ]),
        window_size=2
    )

    assert_array_equal(
        observations[0],
        np.array([
            [0, 0, 2],
            [1, 2, 3]
        ])
    )

    assert_array_equal(
        observations[2],
        np.array([
            [2, 4, 6],
            [3, 6, 9]
        ])
    )

    with pytest.raises(IndexError):
        observations.__getitem__(-1)
        observations.__getitem__(3)
