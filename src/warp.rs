extern crate test;

use crate::projection::{inv_project_vecs, project_vecs};
use crate::transform::transform;
use ndarray::{arr1, arr2, Array2, ArrayView1, ArrayView2};

pub fn warp(
    transform10: ArrayView2<'_, f64>,
    xs0: ArrayView2<'_, f64>,
    depths0: ArrayView1<'_, f64>,
) -> Array2<f64> {
    let points0 = inv_project_vecs(xs0, depths0);
    let points1 = transform(transform10, points0.view());
    project_vecs(points1.view())
}

#[cfg(test)]
mod tests {
    use super::*; // import names from outer scope
    use test::Bencher;

    #[test]
    fn test_warp() {
        let transform10 = arr2(&[
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [-1., 0., 0., 4.],
            [0., 0., 0., 1.],
        ]);
        let xs0 = arr2(&[[0., 0.], [2., -1.]]);
        let depths0 = arr1(&[2., 4.]);
        let xs1 = arr2(&[[0.5, 0.0], [-1.0, 1.0]]);
        assert_eq!(warp(transform10.view(), xs0.view(), depths0.view()), xs1);
    }

    #[bench]
    fn bench_warp(b: &mut Bencher) {
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
        b.iter(|| warp(transform10_view, xs0_view, depths0_view))
    }
}
