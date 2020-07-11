use crate::semi_dense::numeric::{Inv, Inverse};
use crate::semi_dense::flag::Flag;
use ndarray::{Array, Array2, ArrayView2};

fn regularize_patch(
    inv_depth_map: &ArrayView2<'_, Inv>,
    inv_variance_map: &ArrayView2<'_, Inv>,
    flag_map: &ArrayView2<'_, i64>,
) -> Option<Inv> {
    let mut numerator = Inv::from(0.0);
    let mut denominator = Inv::from(0.0);
    for y in 0..3 {
        for x in 0..3 {
            let id = inv_depth_map[[y, x]];
            let iv = inv_variance_map[[y, x]];
            if flag_map[[y, x]] == (Flag::Success as i64) {
                numerator = numerator + id * iv;
                denominator = denominator + iv;
            }
        }
    }
    if denominator == Inv::from(0.0) {
        return None;
    }

    Some(numerator / denominator)
}

pub fn regularize(
    depth_map: &Array2<f64>,
    variance_map: &Array2<f64>,
    flag_map: &Array2<i64>,
) -> Array2<f64> {
    assert_eq!(depth_map.shape(), variance_map.shape());
    assert_eq!(depth_map.shape(), flag_map.shape());
    let shape = depth_map.shape();
    let (height, width) = (shape[0], shape[1]);

    let inv_depth_map = depth_map.map(|&e| e.inv());
    let inv_variance_map = variance_map.map(|&e| e.inv());

    let mut id = Array::zeros((height+2, width+2));
    let mut iv = Array::zeros((height+2, width+2));
    let mut f = Array::from_elem((height+2, width+2), Flag::NotProcessed as i64);
    id.slice_mut(s![1..1+height, 1..1+width]).assign(&inv_depth_map);
    iv.slice_mut(s![1..1+height, 1..1+width]).assign(&inv_variance_map);
    f.slice_mut(s![1..1+height, 1..1+width]).assign(&flag_map);

    let mut regularized: Array2<f64> = Array::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            regularized[[y, x]] = match regularize_patch(
                &id.slice(s![y..y+3, x..x+3]),
                &iv.slice(s![y..y+3, x..x+3]),
                &f.slice(s![y..y+3, x..x+3])
            ) {
                Some(id) => id.inv(),
                None => depth_map[[y, x]]
            }
        }
    }

    regularized
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_regularize_patch() {
        let depth_map = arr2(
            &[[0., 3., 3.],
              [4., 1., 9.],
              [2., 8., 2.]],
        );
        let variance_map = arr2(
            &[[0., 3., 1.],
              [8., 2., 4.],
              [0., 1., 2.]]
        );
        let flag_map = arr2(
            &[[0, 1, 1],
              [1, 0, 1],
              [0, 1, 1]]
        );

        let mut expected = Inv::from(0.);
        let id1 = depth_map[[1, 1]].inv();
        let v1 = variance_map[[1, 1]];
        let mut n_contributions = 0;
        for y in 0..3 {
            for x in 0..3 {
                let id2 = depth_map[[y, x]].inv();
                let v2 = variance_map[[y, x]];
                let flag = flag_map[[y, x]];
                if are_statically_same(id2, id1, v1, v2) &&
                   flag == (Flag::Success as i64) {
                    expected = expected + id2 * v2.inv();
                    n_contributions += 1;
                }
            }
        }
        expected = expected / n_contributions;

        let inv_depth_map = depth_map.map(|&e| e.inv());
        let r = regularize_patch(
            &inv_depth_map.view(),
            &variance_map.view(),
            &flag_map.view()
        );
        assert_eq!(r, expected);
    }

    #[test]
    fn test_regularize() {
        let depth_map = arr2(
            &[[1., 2., 4., 2.],
              [3., 4., 1., 9.],
              [1., 4., 8., 1.]]
        );

        let variance_map = arr2(
            &[[1., 4., 3., 5.],
              [3., 5., 2., 1.],
              [2., 4., 2., 2.]]
        );

        let flag_map = arr2(
            &[[1, 0, 1, 1],
              [1, 1, 0, 1],
              [0, 1, 0, 1]]
        );
        let regularized = regularize(&depth_map, &variance_map, &flag_map);

        let depth_patch = arr2(
            &[[0., 0., 0.],
              [2., 4., 2.],
              [4., 1., 9.]]
        );
        let variance_patch = arr2(
            &[[0., 0., 0.],
              [4., 3., 5.],
              [5., 2., 1.]]
        );
        let flag_patch = arr2(
            &[[0, 0, 0],
              [0, 1, 1],
              [1, 0, 1]]
        );

        assert_eq!(
            regularized[[0, 2]],
            regularize_patch(
                &depth_patch.map(|&e| e.inv()).view(),
                &variance_patch.view(),
                &flag_patch.view(),
            ).inv()
        );
    }
}
