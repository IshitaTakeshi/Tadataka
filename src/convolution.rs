use ndarray::{arr1, arr2, Array, Array2, ArrayBase, Data, Ix1, Ix2, LinalgScalar};

fn convolve2d<S1: Data<Elem = A>, S2: Data<Elem = A>, A: LinalgScalar>(
    map: &ArrayBase<S1, Ix2>,
    kernel: &ArrayBase<S2, Ix2>
) -> Array2<A> {
    let h = map.shape()[0];
    let w = map.shape()[1];
    let kh = kernel.shape()[0];
    let kw = kernel.shape()[1];
    let oh = h-kh+1;
    let ow = w-kw+1;

    let mut out = Array::zeros((oh, ow));
    for y in 0..oh {
        for x in 0..ow {
            out[[y, x]] = (kernel * &map.slice(s![y..y+kh, x..x+kw])).sum();
        }
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convolve2d() {
        let map = arr2(
            &[[1., 2., -1.],
              [0., -1., 1.],
              [3., 2., 1.],
              [2., -1., -1.]]
        );

        let kernel = arr2(
            &[[-2., 2.],
              [0., -1.],
              [1., 1.]]
        );

        let expected = arr2(
            // &[[-2+4+0+1+3+2, -4-2+0-1+2+1],
            //   [0-2+0-2+2-1, +2+2+0-1-1-1]]
            &[[8., -4.],
              [-3., 1.]]
        );
        assert_eq!(convolve2d(&map, &kernel), expected);
    }
}
