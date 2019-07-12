from vitamine.optimization.functions import Function


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
            residuals
        """

        return self.y - self.transformer.compute(theta)
