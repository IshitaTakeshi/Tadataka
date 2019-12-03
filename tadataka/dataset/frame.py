from collections import namedtuple


# HACK we can share APIs between mono and stereo


MonoFrame = namedtuple(
    "MonoFrame",
    [
        "image",
        "depth_map",
        "rotvec",
        "position"
    ]
)


StereoFrame = namedtuple(
    "StereoFrame",
    [
        "image_left",
        "image_right",
        "depth_map_left",
        "depth_map_right",
        "rotvec",
        "position"
    ]
)
