from enum import IntEnum


class ResultFlag(IntEnum):
    SUCCESS = 0
    KEY_OUT_OF_RANGE = -1
    REF_OUT_OF_RANGE = -2
    EPIPOLAR_TOO_SHORT = -3
    INSUFFICIENT_GRADIENT = -4
    NEGATIVE_PRIOR_DEPTH = -5
    NEGATIVE_REF_DEPTH = -6
    NOT_PROCESSED = -7
