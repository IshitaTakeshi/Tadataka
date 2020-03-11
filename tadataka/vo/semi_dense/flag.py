from enum import IntEnum


class ResultFlag(IntEnum):
    SUCCESS = 0
    KEY_OUT_OF_RANGE = -1
    EPIPOLAR_TOO_SHORT = -2
    INSUFFICIENT_GRADIENT = -3
