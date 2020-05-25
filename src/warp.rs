use crate::transform::Transform;
use crate::projection::Projection;
use ndarray::{arr1, arr2, Array, Array1, Array2,
              ArrayBase, ArrayView1, ArrayView2, Data, Ix1, Ix2};

pub trait Warp<XType, DepthType, D> {
    fn warp(&self, x0: XType, depth0: DepthType) -> Array<f64, D>;
}

impl<S> Warp<ArrayView2<'_, f64>, ArrayView1<'_, f64>, Ix2> for ArrayBase<S, Ix2>
where
    S: Data<Elem = f64>
{
    fn warp(
        &self,
        xs0: ArrayView2<'_, f64>,
        depths0: ArrayView1<'_, f64>
    ) -> Array<f64, Ix2> {
        let points0 = Projection::inv_project(&xs0, depths0);
        let points1 = self.transform(&points0);
        Projection::project(&points1.view())
    }
}

impl<S> Warp<ArrayView1<'_, f64>, f64, Ix1> for ArrayBase<S, Ix2>
where
    S: Data<Elem = f64>
{
    fn warp(
        &self,
        x0: ArrayView1<'_, f64>,
        depth0: f64,
    ) -> Array<f64, Ix1> {
        let point0 = Projection::inv_project(&x0, depth0);
        let point1 = self.transform(&point0);
        Projection::project(&point1.view())
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
        assert_eq!(Warp::warp(&transform10.view(), xs0.view(), depths0.view()), xs1);
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
        assert_eq!(Warp::warp(&transform10.view(), xs0.view(), depth0), xs1);
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
        let transform10_view = transform10.view();
        let xs0_view = xs0.view();
        let depths0_view = depths0.view();
        b.iter(|| Warp::warp(&transform10_view, xs0_view, depths0_view))
    }
}
