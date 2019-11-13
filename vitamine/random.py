import base64
import os
import sys


str_encoding = sys.getdefaultencoding()


def random_bytes(n_bytes):
    b = base64.b64encode(os.urandom(n_bytes))
    return b.decode(str_encoding)
