class Residual(object):
    def __init__(self, transformer, Y):
        """
        Y : Target values
        """

        self.transformer = transformer
        self.Y = Y

    def residuals(self, theta):
        """
        Returns:
            residuals
        """

        # HACK the design of the transformer may not be optimal
        return self.Y - self.transformer.transform(theta)
