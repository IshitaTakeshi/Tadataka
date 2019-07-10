from optimization.functions import Function


class BaseResidual(Function):
    def __init__(self, y, transformation):
        """
        y : Target values
        """

        self.y = y
        self.transformation = transformation

    def compute(self, theta):
        """
        Returns:
            residuals
        """

        # HACK the design of the transformer may not be optimal
        return self.y - self.transformation.compute(theta)
