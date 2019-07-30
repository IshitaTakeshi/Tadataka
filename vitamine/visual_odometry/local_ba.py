from vitamine.bundle_adjustment.parameters import (
    ParameterMask, from_params, to_params)


class LocalBundleAdjustment(object):
    def __init__(self, initial_omegas, initial_translations, initial_points,
                 camera_parameters):
        self.camera_parameters = camera_parameters
        self.K = self.camera_parameters.matrix
        self.mask = ParameterMask(initial_omegas, initial_translations,
                                  initial_points)
        self.initial_params = to_params(*self.mask.get_masked())

    def compute(self, keypoints):
        # extract non-nan elements
        keypoints = self.mask.mask_keypoints(keypoints),
        params = bundle_adjustment_core(keypoints, self.initial_params,
                                        mask.n_valid_viewpoints,
                                        mask.n_valid_points,
                                        self.camera_parameters)

        # decompose 'params' into non-nan poses / points
        omegas, translations, points = from_params(
            params, mask.n_valid_viewpoints, mask.n_valid_points)

        # fill with nans to align the shapes
        return mask.fill(omegas, translations, points)


