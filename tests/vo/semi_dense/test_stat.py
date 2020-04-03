import numpy as np
from tadataka.vo.semi_dense.stat import is_statically_same, are_statically_same


def test_is_statically_same():
    def equivalent_condition(v1, v2, var, fac):
        return np.abs(v1-v2) <= fac * np.sqrt(var)

    factor = 2.0
    for i in range(100):
        value1 = np.random.uniform(-100, 100)
        value2 = np.random.uniform(-100, 100)
        variance1 = np.random.uniform(0, 50)
        variance2 = np.random.uniform(0, 50)

        c = is_statically_same(value1, value2, variance1, factor)
        assert(c == equivalent_condition(value1, value2, variance1, factor))

        c = are_statically_same(value1, value2, variance1, variance2, factor)
        c1 = equivalent_condition(value1, value2, variance1, factor)
        c2 = equivalent_condition(value1, value2, variance2, factor)
        assert(c == (c1 and c2))
