from collections import namedtuple


# HACK we can share APIs between mono and stereo


Frame = namedtuple(
    "Frame",
    [
        "camera_model",
        "image",
        "depth_map",
        "rotation",
        "position"
    ]
)
