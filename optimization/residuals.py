class Residual(object):
    def __init__(self, x, y, transformer):
        """
        y : Target value
        x : Values to be transformed by the transformer
        """

        self.x = x
        self.y = y
        self.transformer = transformer

    def residuals(self, theta):
        """
        Returns:
            residuals of shape (n_points, 2)
        """

        # HACK the design of the transformer is may not be optimal
        return self.y - self.transformer(self.x, theta)
