from collections import namedtuple


Frame = namedtuple(
    "Frame",
    [
        "camera_model",
        "pose",  # pose_wf (transform from frame to world)
        "image",
        "depth_map",
    ]
)
