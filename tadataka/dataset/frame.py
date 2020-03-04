from collections import namedtuple


Frame = namedtuple(
    "Frame",
    [
        "camera_model",
        "camera_model_depth",
        "pose",
        "image",
        "depth_map",
    ]
)
