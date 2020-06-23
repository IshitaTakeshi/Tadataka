use crate::transform::Transform;
use crate::projection::Projection;
use crate::camera::{CameraParameters, Normalizer};
use ndarray::{Array, Array1, Array2, ArrayBase, Data, Ix1, Ix2};

pub trait Warp<XType, DepthType, D> {
    type Output;
    fn warp(&self, x0: XType, depth0: DepthType) -> Self::Output;
}

impl<S1, S2> Warp<&ArrayBase<S1, Ix1>, f64, Ix1>
for ArrayBase<S2, Ix2>
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    type Output = (Array1<f64>, f64);
    fn warp(
        &self,
        x0: &ArrayBase<S1, Ix1>,
        depth0: f64
    ) -> Self::Output {
        let point0 = Projection::inv_project(x0, depth0);
        let point1 = self.transform(&point0);
        let x1 = Projection::project(&point1);
        let depth1 = point1[2];
        (x1, depth1)
    }
}

impl<S1, S2, S3> Warp<&ArrayBase<S2, Ix2>, &ArrayBase<S3, Ix1>, Ix2>
for ArrayBase<S1, Ix2>
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: Data<Elem = f64>,
{
    type Output = (Array2<f64>, Array1<f64>);
    fn warp(
        &self,
        xs0: &ArrayBase<S2, Ix2>,
        depths0: &ArrayBase<S3, Ix1>
    ) -> Self::Output {
        let points0 = Projection::inv_project(xs0, depths0);
        let points1 = self.transform(&points0);
        let xs1 = Projection::<Ix2, &Array1<f64>>::project(&points1);
        let depths1 = points1.slice(s![.., 2]).to_owned();
        (xs1, depths1)
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
    type Output = (Array<f64, Ix1>, f64);
    fn warp(&self, u0: &ArrayBase<S1, Ix1>, depth0: f64) -> Self::Output {
        let x0 = self.camera_params0.normalize(u0);
        let (x1, depth1) = Warp::warp(self.transform10, &x0, depth0);
        let u1 = self.camera_params1.unnormalize(&x1);
        (u1, depth1)
    }
}

impl<'a, S1, S2, S3> Warp<&ArrayBase<S1, Ix2>, &ArrayBase<S2, Ix1>, Ix2>
for PerspectiveWarp<'a, S3>
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: Data<Elem = f64>,
{
    type Output = (Array2<f64>, Array1<f64>);
    fn warp(
        &self,
        us0: &ArrayBase<S1, Ix2>,
        depths0: &ArrayBase<S2, Ix1>
    ) -> Self::Output {
        let xs0 = self.camera_params0.normalize(us0);
        let (xs1, depths1) = Warp::warp(self.transform10, &xs0, depths0);
        let us1 = self.camera_params1.unnormalize(&xs1);
        (us1, depths1)
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

        // ps0 = inv_project(xs0, depths0) = [[0., 0., 2.], [8., -4., 4.]]
        // ps1 = transform10 * ps0 = [[2., 0., 4.], [4., -4., -4.]]
        let xs0 = arr2(&[[0., 0.], [2., -1.]]);
        let depths0 = arr1(&[2., 4.]);
        let xs1 = arr2(&[[0.5, 0.0], [-1.0, 1.0]]);
        let depths1 = arr1(&[4., -4.]);
        let (xs1_, depths1_) = Warp::warp(&transform10, &xs0, &depths0);
        assert_eq!(xs1_, xs1);
        assert_eq!(depths1_, depths1);
    }

    #[test]
    fn test_warp_1d() {
        let transform10 = arr2(&[
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [-1., 0., 0., 4.],
            [0., 0., 0., 1.],
        ]);
        // p0 = [0., 0., 2.]
        // p1 = [2., 0., 4.]
        let x0 = arr1(&[0., 0.]);
        let depth0 = 2.;
        let x1 = arr1(&[0.5, 0.0]);
        let depth1 = 4.;
        let (x1_, depth1_) = Warp::warp(&transform10, &x0, depth0);
        assert_eq!(x1_, x1);
        assert_eq!(depth1_, depth1);
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
        // ps0 = inv_project(x0, depth0) = [10., 20., 10.]
        // ps1 = transform10 * ps1 = [10., 20., 20.]
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
        let depth1 = 20.;
        let (u1_, depth1_) = warp10.warp(&u0, depth0);
        assert_eq!(u1_, u1);
        assert_eq!(depth1_, depth1);
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
        // ps0 = inv_project(xs0, depths0) = [[10., 20., 10.], [-20. -20., 5.]]
        // ps1 = transform10 * ps0 = [[10., 20., 20.], [5., -20., 50.]]
        // depths1      = [20. 50.]
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
        let depths1 = arr1(&[20., 50.]);
        let (us1_, depths1_) = warp10.warp(&us0, &depths0);
        assert_eq!(us1_, us1);
        assert_eq!(depths1_, depths1);
    }
}
