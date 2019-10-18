import sys


class InvalidDepthException(Exception):
    pass


class NotEnoughInliersException(Exception):
    pass


def print_error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
