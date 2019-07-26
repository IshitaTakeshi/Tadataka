
def check_points(points):
    assert(points.shape[1] == 3)

def check_poses(omegas, translations):
    assert(omegas.shape[0] == translations.shape[0])
    assert(omegas.shape[1] == 3)
    assert(translations.shape[1] == 3)


def check_keypoints(keypoints, omegas, translations, points):
    n_points = keypoints.shape[1]
    n_viewpoints = keypoints.shape[0]
    assert(points.shape[0] == n_points)
    assert(omegas.shape[0] == n_viewpoints)
    assert(translations.shape[0] == n_viewpoints)
