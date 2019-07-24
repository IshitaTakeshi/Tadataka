from autograd import numpy as np

from vitamine.dataset.points import donut
from vitamine.bundle_adjustment.triangulation import two_view_reconstruction
from vitamine.bundle_adjustment.bundle_adjustment import bundle_adjustment
from vitamine.camera import CameraParameters
from vitamine.dataset.points import cubic_lattice
from vitamine.dataset.observations import (
    generate_observations, generate_translations)
from vitamine.projection.projections import PerspectiveProjection
from vitamine.optimization.residuals import BaseResidual
from vitamine.rigid.transformation import world_to_camera, camera_to_world
from vitamine.rigid.rotation import rodrigues
from vitamine.visualization.cameras import cameras_poly3d

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from vitamine.visualization.visualizers import plot3d
from vitamine.visualization.visualizers import object_color

camera_parameters = CameraParameters(
    focal_length=[1., 1.],
    offset=[0., 0.]
)

projection = PerspectiveProjection(camera_parameters)

camera_rotations_true, camera_locations_true, points_true =\
    donut(inner_r=8, outer_r=12, height=3, point_density=90, n_viewpoints=13)

ax = plot3d(points_true)
ax.add_collection3d(
    cameras_poly3d(camera_rotations_true, camera_locations_true)
)

rotations, translations =\
    world_to_camera(camera_rotations_true, camera_locations_true)

observations, masks = generate_observations(
    rotations, translations, points_true, projection)
observations[~masks] = np.nan

omegas, translations, points_pred = bundle_adjustment(observations, camera_parameters)

camera_rotations_pred, camera_locations_pred =\
    camera_to_world(rodrigues(omegas), translations)

ax = plot3d(points_pred)

ax.add_collection3d(
    cameras_poly3d(camera_rotations_pred, camera_locations_pred, scale=0.4)
)

plt.show()
