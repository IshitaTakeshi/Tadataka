#include <math.h>
#include "_bilinear.h"


double __interpolation(const double *image, const int width,
                       const double cx, const double cy) {
    double lx = floor(cx);
    double ly = floor(cy);
    int lxi = (int)lx;
    int lyi = (int)ly;

    if(lx == cx && ly == cy) {
        return image[lyi * width + lxi];
    }

    double ux = lx + 1.0;
    double uy = ly + 1.0;
    int uxi = (int)ux;
    int uyi = (int)uy;

    if(lx == cx) {
        return (image[lyi * width + lxi] * (ux - cx) * (uy - cy) +
                image[uyi * width + lxi] * (ux - cx) * (cy - ly));
    }

    if(ly == cy) {
        return (image[lyi * width + lxi] * (ux - cx) * (uy - cy) +
                image[lyi * width + uxi] * (cx - lx) * (uy - cy));
    }

    return (image[lyi * width + lxi] * (ux - cx) * (uy - cy) +
            image[lyi * width + uxi] * (cx - lx) * (uy - cy) +
            image[uyi * width + lxi] * (ux - cx) * (cy - ly) +
            image[uyi * width + uxi] * (cx - lx) * (cy - ly));
}


void _interpolation(
    const double *restrict image, const int image_width,
    const double *restrict coordinates, const int n_coordinates,
    double *restrict intensities) {
    for(int i = 0; i < n_coordinates; i++) {
        intensities[i] = __interpolation(
            image, image_width, coordinates[2*i], coordinates[2*i+1]);
    }
}
