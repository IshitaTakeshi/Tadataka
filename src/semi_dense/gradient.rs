use ndarray::{arr1, arr2, Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use crate::gradient::{sobel_x, sobel_y};
use crate::interpolation::Interpolation;

pub struct ImageGradient {
    gx: Array2<f64>,
    gy: Array2<f64>
}

impl ImageGradient {
    pub fn new<S: Data<Elem = f64>>(image: &ArrayBase<S, Ix2>) -> Self {
        let gx = sobel_x(image);
        let gy = sobel_y(image);
        ImageGradient { gx: gx, gy: gy }
    }

    pub fn get<S: Data<Elem = f64>>(
        &self,
        coordinate: &ArrayBase<S, Ix1>
    ) -> Array1<f64> {
        let gx = self.gx.interpolate(coordinate);
        let gy = self.gy.interpolate(coordinate);
        arr1(&[gx, gy])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(grad: &ImageGradient, c: &Array1<f64>) {
        let expected = arr1(&[grad.gx.interpolate(c), grad.gy.interpolate(c)]);
        assert_eq!(grad.get(&c), expected);
    }

    #[test]
    fn test_get() {
        let gx = arr2(
            &[[0., 2., 1.],
              [0., 1., 1.],
              [1., 2., 0.]]
        );
        let gy = arr2(
            &[[1., 1., 0.],
              [0., 2., 1.],
              [1., 0., 1.]]
        );
        let c = arr1(&[0.3, 1.2]);
        let grad = ImageGradient { gx: gx, gy: gy };

        run(&grad, &c);
    }
}
