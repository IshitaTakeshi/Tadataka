use crate::transform::Transform;
use crate::projection::Projection;
use crate::camera::{CameraParameters, Normalizer};
use ndarray::{Array, ArrayBase, Data, Ix1, Ix2};

pub trait Warp<XType, DepthType, D> {
    fn warp(&self, x0: XType, depth0: DepthType) -> Array<f64, D>;
}

impl<S1, S2> Warp<&ArrayBase<S1, Ix1>, f64, Ix1>
for ArrayBase<S2, Ix2>
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    fn warp(&self, x0: &ArrayBase<S1, Ix1>, depth0: f64) -> Array<f64, Ix1> {
        let point0 = Projection::inv_project(x0, depth0);
        let point1 = self.transform(&point0);
        Projection::project(&point1)
    }
}

impl<S1, S2, S3> Warp<&ArrayBase<S2, Ix2>, &ArrayBase<S3, Ix1>, Ix2>
for ArrayBase<S1, Ix2>
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: Data<Elem = f64>,
{
    fn warp(
        &self,
        xs0: &ArrayBase<S2, Ix2>,
        depths0: &ArrayBase<S3, Ix1>
    ) -> Array<f64, Ix2> {
        let points0 = Projection::inv_project(xs0, depths0);
        let points1 = self.transform(&points0);
        Projection::<Ix2, &Array<f64, Ix1>>::project(&points1)
    }
}

struct PerspectiveWarp<'a, T> where T: Data<Elem = f64> {
    camera_params0: &'a CameraParameters,
    camera_params1: &'a CameraParameters,
    transform10: &'a ArrayBase<T, Ix2>,
}

impl<'a, T> PerspectiveWarp<'a, T> where T: Data<Elem = f64> {
    fn new(
        camera_params0: &'a CameraParameters,
        camera_params1: &'a CameraParameters,
        transform10: &'a ArrayBase<T, Ix2>,
    ) -> Self {
        PerspectiveWarp {
            camera_params0: camera_params0,
            camera_params1: camera_params1,
            transform10: transform10,
        }
    }
}

impl<'a, S1, S2> Warp<&ArrayBase<S1, Ix1>, f64, Ix1>
for PerspectiveWarp<'a, S2>
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    fn warp(&self, u0: &ArrayBase<S1, Ix1>, depth0: f64) -> Array<f64, Ix1> {
        let x0 = self.camera_params0.normalize(u0);
        let x1 = Warp::warp(self.transform10, &x0, depth0);
        let u1 = self.camera_params1.unnormalize(&x1);
        u1
    }
}

impl<'a, S1, S2, S3> Warp<&ArrayBase<S1, Ix2>, &ArrayBase<S2, Ix1>, Ix2>
for PerspectiveWarp<'a, S3>
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: Data<Elem = f64>,
{
    fn warp(
        &self,
        us0: &ArrayBase<S1, Ix2>,
        depths0: &ArrayBase<S2, Ix1>
    ) -> Array<f64, Ix2> {
        let xs0 = self.camera_params0.normalize(us0);
        let xs1 = Warp::warp(self.transform10, &xs0, depths0);
        let us1 = self.camera_params1.unnormalize(&xs1);
        us1
    }
}

#[cfg(test)]
mod tests {
    use super::*; // import names from outer scope
    use test::Bencher;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_warp_2d() {
        let transform10 = arr2(&[
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [-1., 0., 0., 4.],
            [0., 0., 0., 1.],
        ]);
        let xs0 = arr2(&[[0., 0.], [2., -1.]]);
        let depths0 = arr1(&[2., 4.]);
        let xs1 = arr2(&[[0.5, 0.0], [-1.0, 1.0]]);
        assert_eq!(Warp::warp(&transform10, &xs0, &depths0), xs1);
    }

    #[test]
    fn test_warp_1d() {
        let transform10 = arr2(&[
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [-1., 0., 0., 4.],
            [0., 0., 0., 1.],
        ]);
        let xs0 = arr1(&[0., 0.]);
        let depth0 = 2.;
        let xs1 = arr1(&[0.5, 0.0]);
        assert_eq!(Warp::warp(&transform10, &xs0, depth0), xs1);
    }

    #[bench]
    fn bench_warp_2d(b: &mut Bencher) {
        let transform10 = arr2(&[
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [-1., 0., 0., 4.],
            [0., 0., 0., 1.],
        ]);
        let xs0 = arr2(&[[0., 0.], [2., -1.]]);
        let depths0 = arr1(&[2., 4.]);
        b.iter(|| Warp::warp(&transform10, &xs0, &depths0))
    }

    #[test]
    fn test_perspective_warp_1d() {
        let camera_params0 = CameraParameters::new((5.0, 5.0), (20.0, 30.0));
        let camera_params1 = CameraParameters::new((20.0, 50.0), (30.0, 20.0));
        let transform10 = arr2(&[
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [-1., 0., 0., 30.],
            [0., 0., 0., 1.],
        ]);
        let depth0 = 10.;
        let u0 = arr1(&[25., 40.]);

        // x0          = [(25. - 20.) / 5., (40. - 30.) / 5.]
        //             = [1., 2.]
        // inv_project(x0, depth0) = [10., 20., 10.]
        // transform10 * (depth0 * x0) = [10., 20., 20.]
        // depth1      = 20.
        // x1          = [0.5, 1.]
        // u1          = [20. * 0.5 + 30., 50. * 1. + 20.]
        //             = [40., 70.]

        let warp10 = PerspectiveWarp::new(
            &camera_params0,
            &camera_params1,
            &transform10
        );

        let u1 = arr1(&[40., 70.]);
        assert_eq!(warp10.warp(&u0, depth0), u1);
    }

    #[test]
    fn test_perspective_warp_2d() {
        let camera_params0 = CameraParameters::new((5.0, 5.0), (20.0, 30.0));
        let camera_params1 = CameraParameters::new((20.0, 50.0), (30.0, 20.0));
        let transform10 = arr2(&[
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [-1., 0., 0., 30.],
            [0., 0., 0., 1.],
        ]);
        let depths0 = arr1(&[10., 5.]);
        let us0 = arr2(&[
            [25., 40.],
            [0., 10.]
        ]);

        // xs0          = [[(25. - 20.) / 5., (40. - 30.) / 5.]
        //                 [( 0. - 20.) / 5., (10. - 30.) / 5.],
        //              = [[1., 2.], [-4. -4.]]
        // inv_project(xs0, depths0) = [[10., 20., 10.], [-20. -20., 5.]]
        // transform10 * (depths0 * xs0) = [[10., 20., 20.], [5., -20., 50.]]
        // depths1      = [20. 40.]
        // x1           = [[0.5, 1.], [0.1, -0.4]]
        // u1          = [[20. * 0.5 + 30., 50. * 1. + 20.],
        //                [20. * (-0.1) + 30., 50 * (-0.4) + 20.]]
        //             = [[40., 70.], [32., 0.]]

        let warp10 = PerspectiveWarp::new(
            &camera_params0,
            &camera_params1,
            &transform10
        );

        let us1 = arr2(&[[40., 70.], [32., 0.]]);
        assert_eq!(warp10.warp(&us0, &depths0), us1);
    }
}
