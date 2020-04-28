class Frame(object):
    def __init__(self, camera_model, image, pose):
        assert(image.ndim == 2)
        self.camera_model = camera_model
        self.image = image
        self.T = pose.T


def relative_frames(refframes, pose_world_key):
    frames = []
    for ref in refframes:
        pose_world_ref = ref.pose
        pose_ref_key = pose_world_ref.inv() * pose_world_key
        frames.append(Frame(ref.camera_model, ref.image, pose_ref_key))
    return frames


class ReferenceSelector(object):
    def __init__(self, refframes, age_map):
        self.refframes = refframes
        self.age_map = age_map

    def __call__(self, u):
        x, y = u
        age = self.age_map[y, x]
        if age == 0:
            return None
        return self.refframes[-age]
