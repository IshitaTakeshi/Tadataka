def safe_invert(v, epsilon=1e-16):
    return 1 / (v + epsilon)
