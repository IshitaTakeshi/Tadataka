import numpy as np


def allow_1d(which_argument):
    def allow_1d_(function):
        def decorated(*args, **kwargs):
            args = list(args)
            ndim = np.ndim(args[which_argument])

            if ndim == 1:
                args[which_argument] = np.atleast_2d(args[which_argument])
                return function(*args, **kwargs)[0]

            if ndim == 2:
                return function(*args, **kwargs)

            raise ValueError(
                f"Argument number {which_argument} has to be 1d or 2d array"
            )
        return decorated
    return allow_1d_
