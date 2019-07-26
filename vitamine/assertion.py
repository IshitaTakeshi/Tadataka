
def check_points(points):
    assert(points.shape[1] == 3)

def check_poses(omegas, translations):
    assert(omegas.shape[0] == translations.shape[0])
    assert(omegas.shape[1] == 3)
    assert(translations.shape[1] == 3)
