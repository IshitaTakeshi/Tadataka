from collections import namedtuple


# HACK we can share APIs between mono and stereo


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
