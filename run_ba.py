from autograd import numpy as np

from bundle_adjustment.triangulation import two_view_reconstruction
from bundle_adjustment.bundle_adjustment import BundleAdjustment, initialize
from camera import CameraParameters
from dataset.points import cubic_lattice
from dataset.generators import generate_observations, generate_translations
from projection.projections import PerspectiveProjection
from rigid.transformation import transform_each
from rigid.rotation import rodrigues


camera_parameters = CameraParameters(
    focal_length=[1., 1.],
    offset=[0., 0.]
)


points_true = cubic_lattice(3)
n_viewpoints = 128
projection = PerspectiveProjection(camera_parameters)


omegas = np.random.uniform(-1, 1, (n_viewpoints, 3))
translations = generate_translations(rodrigues(omegas), points_true)

observations = generate_observations(omegas, translations, points_true, projection)

omegas, translations, points = initialize(observations,
                                          camera_parameters.matrix)

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from visualization import plot3d
plot3d(points_true)
plot3d(points)

ba = BundleAdjustment(observations, projection)
omegas, translations, points = ba.optimize(omegas, translations, points)

plot3d(points)
plt.show()
