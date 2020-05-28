use ndarray::{arr2, Array, Array2, Array4, ArrayBase, Data, Ix2, LinalgScalar};

fn calc_out_size(map_size: usize, kernel_size: usize) -> usize {
    map_size - kernel_size + 1
}

fn im2col<S1: Data<Elem = A>, S2: Data<Elem = A>, A: LinalgScalar>(
    map: &ArrayBase<S1, Ix2>,
    kernel: &ArrayBase<S2, Ix2>,
) -> Array4<A> {
    let h = map.shape()[0];
    let w = map.shape()[1];
    let kh = kernel.shape()[0];
    let kw = kernel.shape()[1];
    let oh = calc_out_size(h, kh);
    let ow = calc_out_size(w, kw);

    let mut col = Array::zeros((kh, kw, oh, ow));
    for y in 0..kh {
        for x in 0..kw {
            col.slice_mut(s![y, x, .., ..]).assign(&map.slice(s![y..y + oh, x..x + ow]));
        }
    }

    col
}

fn convolve2d<S1: Data<Elem = A>, S2: Data<Elem = A>, A: LinalgScalar>(
    map: &ArrayBase<S1, Ix2>,
    kernel: &ArrayBase<S2, Ix2>
) -> Array2<A> {
    let h = map.shape()[0];
    let w = map.shape()[1];
    let kh = kernel.shape()[0];
    let kw = kernel.shape()[1];
    let oh = calc_out_size(h, kh);
    let ow = calc_out_size(w, kw);

    let col = im2col(map, kernel);
    let mut out = Array::zeros((oh, ow));
    for y in 0..oh {
        for x in 0..ow {
            out[[y, x]] = (kernel * &col.slice(s![.., .., y, x])).sum();
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
