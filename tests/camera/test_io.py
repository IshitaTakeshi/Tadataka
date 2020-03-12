from pathlib import Path

from numpy.testing import assert_array_equal

from tadataka.camera.distortion import FOV
from tadataka.camera.io import load, save
from tadataka.camera.model import CameraModel
from tadataka.camera.parameters import CameraParameters


workspace = Path(__file__).parent  # diectory which this file is placed


def test_load():
    path = Path(workspace, "sample_camera_model.txt")
    camera_models = load(path)

    assert(set(camera_models.keys()) == {1, 2})

    camera_parameters = camera_models[1].camera_parameters
    assert_array_equal(camera_parameters.focal_length, [200.0, 300.1])
    assert_array_equal(camera_parameters.offset, [350.4, 250.1])

    distortion_model = camera_models[1].distortion_model
    assert(isinstance(distortion_model, FOV))
    assert_array_equal(distortion_model.params, [0.01])

    camera_parameters = camera_models[2].camera_parameters
    assert_array_equal(camera_parameters.focal_length, [210.3, 320.2])
    assert_array_equal(camera_parameters.offset, [340.0, 230.0])

    distortion_model = camera_models[2].distortion_model
    assert(isinstance(distortion_model, FOV))
    assert_array_equal(distortion_model.params, [-0.03])


def test_save():

    camera_parameters = CameraParameters(focal_length=[123.4, 200.8],
                                         offset=[250.1, 150.0])
    distortion_model = FOV(0.02)
    camera_model1 = CameraModel(camera_parameters, distortion_model)

    camera_parameters = CameraParameters(focal_length=[400.3, 500.0],
                                         offset=[248.0, 152.0])
    distortion_model = FOV(-0.01)
    camera_model2 = CameraModel(camera_parameters, distortion_model)

    path = Path(workspace, "camera_models.txt")

    camera_models = {1: camera_model1, 2: camera_model2}
    save(path, camera_models)

    expected = ('1 FOV 123.4 200.8 250.1 150.0 0.02\n'
                '2 FOV 400.3 500.0 248.0 152.0 -0.01\n')

    with open(path, 'r') as f:
        s = f.read()

    assert(s == expected)

    path.unlink()  # remove the written file
