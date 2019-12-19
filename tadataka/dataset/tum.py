import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from tadataka.dataset.match import match_timestamps


def load_poses(path):
    array = np.loadtxt(path)
    timestamps = array[:, 0]
    positions = array[:, 1:4]
    quaternions = array[:, 4:8]
    rotations = Rotation.from_quat(quaternions)
    return timestamps, rotations, positions


def load_image_paths(filepath, prefix):
    timestamps = []
    image_paths = []

    with open(str(filepath), "r") as f:
        reader = csv.reader(f, delimiter=' ')

        for row in reader:
            if row[0].startswith('#'):
                continue
            timestamps.append(float(row[0]))
            filepath = str(Path(prefix, row[1]))
            image_paths.append(filepath)
    return np.array(timestamps), image_paths


def synchronize(timestamps_ref, timestamps1, timestamps2, max_diff=np.inf):
    matches01 = match_timestamps(timestamps_ref, timestamps1, max_diff)
    matches02 = match_timestamps(timestamps_ref, timestamps2, max_diff)
    indices_ref, indices1, indices2 = np.intersect1d(
        matches01[:, 0], matches02[:, 0], return_indices=True
    )
    return np.column_stack((indices_ref,
                            matches01[indices1, 1],
                            matches02[indices2, 1]))
