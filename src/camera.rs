use ndarray::{arr1, arr2, Array1, Array2, ArrayView2};

struct CameraParameters {
     focal_length: Array1<f64>,
     offset: Array1<f64>
}

impl CameraParameters {
    fn new(focal_length: (f64, f64), offset: (f64, f64)) -> CameraParameters {
        let (fx, fy) = focal_length;
        let (ox, oy) = offset;
        CameraParameters{ focal_length: arr1(&[fx, fy]), offset: arr1(&[ox, oy]) }
    }
}

trait Normalizer {
    fn normalize(&self, keypoints: ArrayView2<'_, f64>) -> Array2<f64>;
    fn unnormalize(&self, keypoints: ArrayView2<'_, f64>) -> Array2<f64>;
}

impl Normalizer for CameraParameters {
    fn normalize(&self, keypoints: ArrayView2<'_, f64>) -> Array2<f64> {
        let n = keypoints.shape()[0];
        let focal_length = self.focal_length.broadcast((n, 2)).unwrap();
        let offset = self.offset.broadcast((n, 2)).unwrap();
        return (keypoints.to_owned() - offset) / focal_length;
    }

    fn unnormalize(&self, keypoints: ArrayView2<'_, f64>) -> Array2<f64> {
        let n = keypoints.shape()[0];
        let focal_length = self.focal_length.broadcast((n, 2)).unwrap();
        let offset = self.offset.broadcast((n, 2)).unwrap();
        return keypoints.to_owned() * focal_length + offset;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalizer() {
        let camera_params = CameraParameters::new((10., 20.), (2., 4.));

        let normalized = arr2(&[
            [1.0, 1.0],
            [-0.2, -0.2],
            [0.6, 0.3]
        ]);

        let unnormalized = arr2(&[
            [12., 24.],
            [0., 0.],
            [8., 10.]
        ]);

        assert_eq!(camera_params.normalize(unnormalized.view()), normalized);
        assert_eq!(camera_params.unnormalize(normalized.view()), unnormalized);
    }
}
