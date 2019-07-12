from autograd import numpy as np

from bundle_adjustment.triangulation import two_view_reconstruction
from bundle_adjustment.bundle_adjustment import BundleAdjustment
from camera import CameraParameters
from dataset.points import cubic_lattice, corridor
from dataset.bundle_adjustment import generate_observations, generate_translations
from projection.projections import PerspectiveProjection
from rigid.transformation import transform_each
from rigid.rotation import rodrigues


def generate_poses(n_viewpoints):
    omegas = np.zeros((n_viewpoints, 3))  # np.random.uniform(-1, 1, (n_viewpoints, 3))

    translations = np.vstack((
        np.zeros(n_viewpoints),
        np.zeros(n_viewpoints),
        np.arange(0, n_viewpoints) - 1.5
    )).T

    return omegas, translations


window_size = 8
points_true = corridor(width=2, height=4, length=2)
omegas_true, translations_true = generate_poses(n_viewpoints=12)

camera_parameters = CameraParameters(
    focal_length=[1., 1.],
    offset=[0., 0.]
)
projection = PerspectiveProjection(camera_parameters)

observations, masks = generate_observations(
    rodrigues(omegas_true), translations_true, points_true, projection)

observations[~masks] = np.nan

N = observations.shape[0]
for i in range(0, N-window_size+1):
    ba = BundleAdjustment(
        observations[i:i+window_size],
        camera_parameters
    )
    omegas, translations, points = ba.optimize()


from matplotlib import pyplot as plt
from visualizer.visualizers import plot3d
plot3d(points_true)
plot3d(points)
plt.show()
