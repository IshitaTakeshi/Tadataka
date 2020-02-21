from scipy import ndimage


sobel_mode = "reflect"


def grad_x(image):
    return ndimage.sobel(image, axis=1, mode=sobel_mode)


def grad_y(image):
    return ndimage.sobel(image, axis=0, mode=sobel_mode)
