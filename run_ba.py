from autograd import numpy as np

from vitamine.dataset.points import donut
from vitamine.bundle_adjustment.bundle_adjustment import bundle_adjustment_core
from vitamine.camera import CameraParameters
from vitamine.dataset.points import cubic_lattice
from vitamine.dataset.observations import (
    generate_observations, generate_translations)
from vitamine.projection.projections import PerspectiveProjection
from vitamine.optimization.residuals import BaseResidual
from vitamine.rigid.coordinates import world_to_camera, camera_to_world
from vitamine.rigid.rotation import rodrigues
from vitamine.visualization.cameras import cameras_poly3d
from vitamine.visual_odometry.local_ba import LocalBundleAdjustment

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from vitamine.visualization.visualizers import plot3d
from vitamine.visualization.visualizers import object_color

camera_parameters = CameraParameters(
    focal_length=[1., 1.],
    offset=[0., 0.]
)

projection = PerspectiveProjection(camera_parameters)

camera_omegas_true, camera_locations_true, points_true =\
    donut(inner_r=8, outer_r=12, height=3, point_density=30, n_viewpoints=12)

ax = plot3d(points_true)
ax.add_collection3d(
    cameras_poly3d(rodrigues(camera_omegas_true), camera_locations_true)
)

omegas, translations = world_to_camera(camera_omegas_true,
                                       camera_locations_true)

keypoints, masks = generate_observations(
    rodrigues(omegas), translations, points_true, projection)
keypoints[~masks] = np.nan

ba = LocalBundleAdjustment(omegas, translations, points_true,
                           camera_parameters)
omegas_pred, translations_pred, points_pred = ba.compute(keypoints)

ax = plot3d(points_pred)

ax.add_collection3d(
    cameras_poly3d(
        *camera_to_world(rodrigues(omegas_pred), translations_pred),
        scale=0.4
    )
)

plt.show()
