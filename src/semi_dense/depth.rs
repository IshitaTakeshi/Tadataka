extern crate test;
use crate::projection::inv_project_vec;
use ndarray::{arr1, arr2, ArrayView1, ArrayView2};

pub fn calc_ref_depth(
    transform_rk: ArrayView2<'_, f64>,
    x_key: ArrayView1<'_, f64>,
    depth_key: f64,
) -> f64 {
    let p_key = inv_project_vec(x_key, depth_key);
    let r_rk_z: ArrayView1<'_, f64> = transform_rk.slice(s![2, 0..3]);
    let t_rk_z: f64 = transform_rk[[2, 3]];
    r_rk_z.dot(&p_key) + t_rk_z
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
            calc_ref_depth(transform_rk.view(), x_key.view(), depth_key),
            -2.0 + 4.0
        );
    }
}
