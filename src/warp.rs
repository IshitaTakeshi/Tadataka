use crate::transform::Transform;
use crate::projection::Projection;
use ndarray::{arr1, arr2, Array, ArrayBase, ArrayView1, ArrayView2, Data, Ix1, Ix2};

pub trait Warp<XType, DepthType, D> {
    fn warp(&self, x0: XType, depth0: DepthType) -> Array<f64, D>;
}

impl<S1, S2> Warp<&ArrayBase<S1, Ix1>, f64, Ix1> for ArrayBase<S2, Ix2>
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    fn warp(
        &self,
        x0: &ArrayBase<S1, Ix1>,
        depth0: f64,
    ) -> Array<f64, Ix1> {
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

#[cfg(test)]
mod tests {
    use super::*; // import names from outer scope
    use test::Bencher;

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
}
