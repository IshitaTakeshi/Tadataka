class BaseTransformer(object):
    def __init__(self, X):
        """
        Args:
            X: np.ndarray
                Parameters to be transformed
        """
        self.X = X

    def transform(self, params):
        """
        Args:
            params: np.ndarray
                Parameters to be optimized
        """
        raise NotImplementedError()
