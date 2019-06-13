from skimage.feature import (match_descriptors, corner_peaks, corner_harris,
                             BRIEF)


def extract_keypoints(image1, image2):
    keypoints1 = corner_peaks(corner_harris(image1), min_distance=5)
    keypoints2 = corner_peaks(corner_harris(image2), min_distance=5)

    extractor = BRIEF(mode="normal", sigma=0.4)

    extractor.extract(image1, keypoints1)
    keypoints1 = keypoints1[extractor.mask]
    descriptors1 = extractor.descriptors

    extractor.extract(image2, keypoints2)
    keypoints2 = keypoints2[extractor.mask]
    descriptors2 = extractor.descriptors

    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)
    return keypoints1, keypoints2, matches
