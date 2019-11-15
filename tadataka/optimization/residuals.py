from tadataka.optimization.functions import Function


class BaseResidual(Function):
    def __init__(self, y, transformer):
        """
        y : Target values
        """

        self.y = y
        self.transformer = transformer

    def compute(self, theta):
        """
        Returns:
            calc residual r = y - f(theta)
        """

        return self.y - self.transformer.compute(theta)
