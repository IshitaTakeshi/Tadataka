import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from tadataka.dataset.match import match_timestamps


def convert_to_tum_poses(rotations: Rotation, positions: np.ndarray):
    assert(len(rotations) == positions.shape[0])

    posevecs = np.empty((len(rotations), 7))
    for i, (position, rot) in enumerate(zip(positions, rotations)):
        posevecs[i] = np.hstack((position, rot.as_quat()))
    return posevecs


def save_in_tum_format(filename, timestamps, rotations, positions):
    assert(len(timestamps) == len(rotations) == positions.shape[0])

    posevecs = convert_to_tum_poses(rotations, positions)

    with open(filename, "w") as f:
        for timestamp, posevec in zip(timestamps, posevecs):
            posestr = " ".join(map(str, posevec.tolist()))
            row = "{} {}".format(timestamp, posestr)

            f.write(row + "\n")


def load_image_paths(filepath, prefix, delimiter=' '):
    # prefix: only subpath of the data file is written in 'filepath'
    # prefix is for concatenating it with the subpath and generate the abs path

    timestamps = []
    image_paths = []

    with open(str(filepath), "r") as f:
        reader = csv.reader(f, delimiter=delimiter)

        for row in reader:
            if row[0].startswith('#'):
                continue
            timestamps.append(float(row[0]))
            filepath = str(Path(prefix, row[1]))
            image_paths.append(filepath)
    return np.array(timestamps), image_paths


def synchronize(timestamps1, timestamps2, timestamps_ref, max_diff=np.inf):
    # match timestamps and return indices
    # timestamps1[matches01[i, 0]] should be the closest to
    # timestamps_ref[matches01[i, 1]]
    matches01 = match_timestamps(timestamps_ref, timestamps1, max_diff)
    matches02 = match_timestamps(timestamps_ref, timestamps2, max_diff)

    # find shared neighbors
    indices_ref, indices1, indices2 = np.intersect1d(
        matches01[:, 0], matches02[:, 0], return_indices=True
    )

    return np.column_stack((matches01[indices1, 1],
                            matches02[indices2, 1],
                            indices_ref))
