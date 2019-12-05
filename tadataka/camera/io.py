from configparser import ConfigParser


def load(filename):
    with open(filename, 'r') as f:
        for line in f:
            camera_id = line.split(' ')[0]


def save(filename, camera_models):
    with open(filename, 'w') as f:
        for camera_id, camera_model in camera_models.items():
            line = ' '.join([str(camera_id), str(camera_model)])
            f.write(line)
