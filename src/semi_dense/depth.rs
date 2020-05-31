use crate::projection::Projection;
use ndarray::{arr1, arr2, Array1, Array2, ArrayView1};
use super::numeric::{Inv, Inverse};
use crate::triangulation::calc_depth0;

pub fn calc_ref_depth(
    transform_rk: &Array2<f64>,
    x_key: &Array1<f64>,
    depth_key: f64,
) -> f64 {
    let p_key = Projection::inv_project(x_key, depth_key);
    let r_rk_z: ArrayView1<'_, f64> = transform_rk.slice(s![2, 0..3]);
    let t_rk_z: f64 = transform_rk[[2, 3]];
    r_rk_z.dot(&p_key) + t_rk_z
}

pub fn calc_key_depth(
    transform_rk: &Array2<f64>,
    x_key: &Array1<f64>,
    x_ref: &ArrayView1<'_, f64>,
) -> f64 {
    calc_depth0(transform_rk, x_key, x_ref)
}

pub fn depth_search_range(inv_depth_range: &(Inv, Inv)) -> (f64, f64) {
    let (min_inv_depth, max_inv_depth) = inv_depth_range;
    let min_depth = max_inv_depth.inv();
    let max_depth = min_inv_depth.inv();
    (min_depth, max_depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calc_inv_depth() {
        let depth_key = 4.0;
        let x_key = arr1(&[0.5, 2.0]);
        let transform_rk = arr2(&[
            [0., 0., 1., 3.],
            [0., 1., 0., 2.],
            [-1., 0., 0., 4.],
            [0., 0., 0., 1.],
        ]);
        assert_eq!(
            calc_ref_depth(&transform_rk, &x_key, depth_key),
            -2.0 + 4.0
        );
    }

    #[test]
    fn test_depth_search_range() {
        let min_inv_depth = Inv::from(0.01);
        let max_inv_depth = Inv::from(0.10);
        let (dmin, dmax) = depth_search_range(&(min_inv_depth, max_inv_depth));
        assert_eq!(dmin, max_inv_depth.inv());
        assert_eq!(dmax, min_inv_depth.inv());
    }

}
