use ndarray::{arr1, Array, Array1, ArrayView1};
use ndarray_linalg::Norm;
use crate::gradient::gradient1d;

fn calc_error(a: &ArrayView1<f64>, b: &Array1<f64>) -> f64 {
    let d = a.to_owned() - b;
    d.dot(&d)
}

fn search_(sequence: &Array1<f64>, kernel: &Array1<f64>) -> usize {
    let n = sequence.shape()[0];
    let k = kernel.shape()[0];

    let mut min_error = f64::INFINITY;
    let mut argmin = 0;
    for i in 0..n - k + 1 {
        let e = calc_error(&sequence.slice(s![i..i + k]), &kernel);
        if e < min_error {
            min_error = e;
            argmin = i;
        }
    }

    argmin as usize
}

pub fn search(sequence: &Array1<f64>, kernel: &Array1<f64>) -> usize {
    let argmin = search_(&sequence, &kernel);
    let k = kernel.shape()[0];
    let offset = (k / 2) as usize;
    argmin + offset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intensity_search() {
        // errors = [25 + 16 + 0, 4 + 9 + 4, 1 + 25 + 9, 9 + 0 + 1, 4 + 16 + 1]
        //        = [41, 17, 35, 10, 21]
        // argmin(errors) == 3
        // offset == 1 ( == len(intensities_key) // 2)
        // expected = argmin(errors) + offset == 4
        let intensities_ref = arr1(&[-4., 3., 2., 4., -1., 3., 1.]);
        let intensities_key = arr1(&[1., -1., 2.]);
        let index = search(&intensities_ref, &intensities_key);
        assert_eq!(index, 4);

        // argmin(errors) == 2
        // offset == 1 ( == len(intensities_key) // 2)
        // expected = argmin(errors) + offset == 3
        let intensities_ref = arr1(&[-4., 3., 1., -1.]);
        let intensities_key = arr1(&[1., -1.]);
        let index = search(&intensities_ref, &intensities_key);
        assert_eq!(index, 3);

        // argmin(errors) == 0
        // offset == 1 ( == len(intensities_key) // 2)
        // expected = argmin(errors) + offset == 1
        let intensities_ref = arr1(&[1., -1., -4., 3.]);
        let intensities_key = arr1(&[1., -1.]);
        let index = search(&intensities_ref, &intensities_key);
        assert_eq!(index, 1);
    }
}
