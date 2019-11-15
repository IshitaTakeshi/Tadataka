from collections import namedtuple


Frame = namedtuple(
    "Frame",
    [
        "timestamp_rgb",
        "timestamp_depth",
        "image",
        "depth_map"
    ]
)

