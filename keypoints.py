from skimage.feature import (match_descriptors, corner_peaks, corner_harris,
                             BRIEF)


def extract_keypoints(image1, image2):
    extractor = BRIEF(mode="normal", sigma=0.4)

    def extract_(image):
        keypoints = corner_peaks(corner_harris(image), min_distance=5)
        extractor.extract(image, keypoints)
        keypoints = keypoints[extractor.mask]
        descriptors = extractor.descriptors
        return keypoints, descriptors

    keypoints1, descriptors1 = extract_(image1)
    keypoints2, descriptors2 = extract_(image2)

    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

    return keypoints1, keypoints2, matches
