from configparser import ConfigParser
from tadataka.camera.model import CameraModel

def parse_(line):
    camera_id, model_params = line.split(' ', maxsplit=1)
    try:
        camera_id = int(camera_id)
    except ValueError:
        raise ValueError("Camera ID must be integer")
    return camera_id, CameraModel.fromstring(model_params)


def load(filename):
    camera_models = dict()
    with open(filename, 'r') as f:
        for line in f:
            camera_id, camera_model = parse_(line)
            camera_models[camera_id] = camera_model
    return camera_models


def save(filename, camera_models):
    # sort by camera_id to make it easy to test
    items = sorted(camera_models.items(), key=lambda v: v[0])

    with open(filename, 'w') as f:
        for camera_id, camera_model in items:
            line = ' '.join([str(camera_id), str(camera_model)])
            f.write(line + '\n')
