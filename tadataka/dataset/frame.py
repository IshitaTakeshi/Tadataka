from collections import namedtuple


# HACK we can share APIs between mono and stereo


Frame = namedtuple(
    "Frame",
    [
        "camera_model",
        "pose",
        "image",
        "depth_map",
    ]
)
