import numpy as np


def decrease(validity_map, mask, rate=0.98):
    validity_map[mask] = validity_map[mask] * rate
    return validity_map


def increase(validity_map, mask, rate=1.02):
    validity_map[mask] = np.minimum(validity_map[mask] * rate, 1.0)
    return validity_map
