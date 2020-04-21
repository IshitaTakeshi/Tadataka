def is_statically_same(inv_depth1, inv_depth2, variance, factor):
    # equivalent_condition:
    #   np.abs(inv_depth1-inv_depth2) <= factor * np.sqrt(variance1)
    ds = (inv_depth1 - inv_depth2) * (inv_depth1 - inv_depth2)
    fs = factor * factor
    return ds <= fs * variance


def are_statically_same(inv_depth1, inv_depth2, variance1, variance2, factor):
    c1 = is_statically_same(inv_depth1, inv_depth2, variance1, factor)
    c2 = is_statically_same(inv_depth1, inv_depth2, variance2, factor)
    return c1 and c2
