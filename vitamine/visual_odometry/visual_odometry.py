from vitamine.bundle_adjustment.bundle_adjustment import BundleAdjustment


class VisualOdometry(object):
    def __init__(self, observations, camera_parameters, window_size, start=0, end=None):
        n_observations = observations.shape[0]
        self.observations = observations
        self.camera_parameters = camera_parameters
        self.window_size = window_size
        self.start = start
        self.end = n_observations if end is None else max(n_observations, end)
        assert(self.start < self.end)

    def frames(self):
        for i in range(self.start, self.end-self.window_size+1):
            yield self.estimate(i)

    def estimate(self, i):
        ba = BundleAdjustment(
            self.observations[i:i+self.window_size],
            self.camera_parameters,
            # initial_omegas=omegas,
            # initial_translations=translations,
            # initial_points=points
        )
        return ba.optimize()
