use ndarray::{arr2, Array, Array2, Array4, ArrayBase, ArrayView2,
              Data, Ix2, LinalgScalar};

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
            let k = &map.slice(s![y..y + oh, x..x + ow]);
            col.slice_mut(s![y, x, .., ..]).assign(k);
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
    let out_h = calc_out_size(h, kh);
    let out_w = calc_out_size(w, kw);

    assert!(kh % 2 == 1);
    assert!(kw % 2 == 1);

    let pad_h = ((kh - 1) / 2) as usize;
    let pad_w = ((kw - 1) / 2) as usize;
    let col = im2col(map, kernel);

    let mut out = Array::zeros((h, w));
    for y in 0..out_h {
        for x in 0..out_w{
            out[[y+pad_h, x+pad_w]] = (kernel * &col.slice(s![.., .., y, x])).sum();
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convolve2d() {
        fn conv(a: &ArrayView2<'_, i64>, b: &Array2<i64>) -> i64 {
            (a * b).sum()
        }

        let map = arr2(
            &[[1, 2, -1, 0],
              [0, -1, -1, 1],
              [3, -2, 1, -1],
              [-2, -1, 1, 2],
              [0, 1, -1, 2],
              [3, 4, 1, -1]]
        );

        let kernel = arr2(
            &[[-2, 2, 1],
              [0, -1, 2],
              [1, 1, 1],
              [1, 0, -1],
              [2, 3, 1]]
        );

        let mut expected = Array::zeros((map.shape()[0], map.shape()[1]));
        expected[[2, 1]] = conv(&map.slice(s![0..5, 0..3]), &kernel);
        expected[[2, 2]] = conv(&map.slice(s![0..5, 1..4]), &kernel);
        expected[[3, 1]] = conv(&map.slice(s![1..6, 0..3]), &kernel);
        expected[[3, 2]] = conv(&map.slice(s![1..6, 1..4]), &kernel);
        assert_eq!(convolve2d(&map, &kernel), expected);
    }
}
