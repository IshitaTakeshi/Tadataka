from vitamine.bundle_adjustment.bundle_adjustment import bundle_adjustment
from vitamine.bundle_adjustment.initializers import (
    PoseInitializer, PointInitializer)
from vitamine.bundle_adjustment.parameters import (
    ParameterMask, from_params, to_params)


class VisualOdometry(object):
    def __init__(self, observations, camera_parameters, window_size, start=0, end=None):
        n_observations = observations.shape[0]
        self.observations = observations
        self.camera_parameters = camera_parameters
        self.window_size = window_size
        self.start = start
        self.end = n_observations if end is None else max(n_observations, end)
        assert(self.start < self.end)

    def sequence(self):
        K = self.camera_parameters.matrix

        initial_omegas = None
        initial_translations = None
        initial_points = None

        for i in range(self.start, self.end-self.window_size+1):
            keypoints = self.observations[i:i+self.window_size]

            if initial_points is None:
                point_initializer = PointInitializer(keypoints, K)
                initial_points = point_initializer.initialize()

            if initial_omegas is None and initial_translations is None:
                pose_initializer = PoseInitializer(keypoints, K)
                initial_omegas, initial_translations =\
                    pose_initializer.initialize(initial_points)

            mask = ParameterMask(initial_omegas, initial_translations,
                                 initial_points)
            omegas, translations, points = mask.get_masked()
            params = to_params(omegas, translations, points)

            keypoints = mask.mask_keypoints(keypoints)

            params = bundle_adjustment(keypoints, params,
                                       mask.n_valid_viewpoints,
                                       mask.n_valid_points,
                                       self.camera_parameters)

            omegas, translations, points =\
                from_params(params, mask.n_valid_viewpoints, mask.n_valid_points)

            initial_omegas, initial_translations, initial_points =\
                mask.fill(omegas, translations, points)
            yield omegas, translations, points
