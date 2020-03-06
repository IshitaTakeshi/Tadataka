from collections import namedtuple


Frame = namedtuple(
    "Frame",
    [
        "camera_model",
        "pose",
        "image",
        "depth_map",
    ]
)
