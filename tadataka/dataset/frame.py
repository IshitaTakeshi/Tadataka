from collections import namedtuple


# HACK we can share APIs between mono and stereo


MonoFrame = namedtuple(
    "MonoFrame",
    [
        "camera_model",
        "image",
        "depth_map",
        "position",
        "rotation"
    ]
)


StereoFrame = namedtuple(
    "StereoFrame",
    [
        "camera_model",
        "image_left",
        "image_right",
        "depth_map_left",
        "depth_map_right",
        "position_left",
        "position_right",
        "rotation"
    ]
)
