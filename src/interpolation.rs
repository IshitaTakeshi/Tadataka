use ndarray::{Array, Array1, ArrayBase, ArrayView1, ArrayView2, Data, Ix1, Ix2};
use num::NumCast;
use num_traits::float::Float;

pub trait Interpolation<CoordinateType, OutputType> {
    fn interpolate(&self, c: &CoordinateType) -> OutputType;
}

fn interpolate<A: Float>(
    image: &ArrayView2<'_, A>,
    coordinate: &ArrayView1<'_, A>
) -> A {
    let cx = coordinate[0];
    let cy = coordinate[1];
    let lx = cx.floor();
    let ly = cy.floor();
    let lxi: usize= NumCast::from(lx).unwrap();
    let lyi: usize= NumCast::from(ly).unwrap();

    if lx == cx && ly == cy {
        return image[[lyi, lxi]];
    }

    let ux = lx + NumCast::from(1.).unwrap();
    let uy = ly + NumCast::from(1.).unwrap();
    let uxi: usize = NumCast::from(ux).unwrap();
    let uyi: usize = NumCast::from(uy).unwrap();

    if lx == cx {
        return image[[lyi, lxi]] * (ux - cx) * (uy - cy)
             + image[[uyi, lxi]] * (ux - cx) * (cy - ly);
    }

    if ly == cy {
        return image[[lyi, lxi]] * (ux - cx) * (uy - cy)
             + image[[lyi, uxi]] * (cx - lx) * (uy - cy);
    }

    image[[lyi, lxi]] * (ux - cx) * (uy - cy) +
    image[[lyi, uxi]] * (cx - lx) * (uy - cy) +
    image[[uyi, lxi]] * (ux - cx) * (cy - ly) +
    image[[uyi, uxi]] * (cx - lx) * (cy - ly)
}

impl<A, S1, S2> Interpolation<ArrayBase<S1, Ix1>, A> for ArrayBase<S2, Ix2>
where
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
    A: Float,
{
    fn interpolate(&self, coordinate: &ArrayBase<S1, Ix1>) -> A {
        assert!(coordinate.shape()[0] == 2);
        interpolate(&self.view(), &coordinate.view())
    }
}

impl<A, S1, S2> Interpolation<ArrayBase<S1, Ix2>, Array1<A>> for ArrayBase<S2, Ix2>
where
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
    A: Float,
{
    fn interpolate(&self, coordinates: &ArrayBase<S1, Ix2>) -> Array1<A> {
        assert!(coordinates.shape()[1] == 2);

        let n = coordinates.shape()[0];
        let mut intensities = Array::zeros(n);
        for i in 0..n {
            intensities[i] = interpolate(&self.view(), &coordinates.row(i));
        }
        intensities
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_interpolate_1d_input() {
        let image = arr2(&[[0., 1., 5.],
                           [0., 0., 2.],
                           [4., 3., 2.],
                           [5., 6., 1.]]);

        let expected = image[[2, 1]] * (2.0 - 1.3) * (3.0 - 2.6)
                     + image[[2, 2]] * (1.3 - 1.0) * (3.0 - 2.6)
                     + image[[3, 1]] * (2.0 - 1.3) * (2.6 - 2.0)
                     + image[[3, 2]] * (1.3 - 1.0) * (2.6 - 2.0);

        let c = arr1(&[1.3, 2.6]);
        assert_eq!(image.interpolate(&c), expected);

        // minimum coordinate
        let c = arr1(&[0.0, 0.0]);
        assert_eq!(image.interpolate(&c), image[[0, 0]]);

        // minimum x
        let c = arr1(&[0.0, 0.1]);
        let expected = image[[0, 0]] * (1.0 - 0.0) * (1.0 - 0.1)
                     + image[[1, 0]] * (1.0 - 0.0) * (0.1 - 0.0);
        assert_eq!(image.interpolate(&c), expected);

        // minimum y
        let c = arr1(&[0.1, 0.0]);
        let expected = image[[0, 0]] * (1.0 - 0.1) * (1.0 - 0.0)
                     + image[[0, 1]] * (0.1 - 0.0) * (1.0 - 0.0);
        assert_eq!(image.interpolate(&c), expected);

        // maximum x
        let c = arr1(&[2.0, 2.9]);
        let expected = image[[2, 2]] * (3.0 - 2.0) * (3.0 - 2.9)
                     + image[[3, 2]] * (3.0 - 2.0) * (2.9 - 2.0);
        assert_eq!(image.interpolate(&c), expected);

        // maximum y
        let c = arr1(&[1.9, 3.0]);
        let expected = image[[3, 1]] * (2.0 - 1.9) * (4.0 - 3.0)
                     + image[[3, 2]] * (1.9 - 1.0) * (4.0 - 3.0);
        assert_eq!(image.interpolate(&c), expected);

        // maximum c
        let c = arr1(&[2.0, 3.0]);
        assert_eq!(image.interpolate(&c), image[[3, 2]]);

        // TODO How about invalid input?
        // let c = arr(&[[3.0, 2.01]]);
        // let c = arr(&[[3.01, 2.0]]);
        // let c = arr(&[[-0.01, 0.0]]);
        // let c = arr(&[[0.0, -0.01]]);
    }

    #[test]
    fn test_interpolate_2d_input() {
        let image = arr2(&[[0., 1., 5.],
                           [0., 0., 2.],
                           [4., 3., 2.],
                           [5., 6., 1.]]);
        let c = arr2(&[[0.0, 0.1],
                       [0.1, 0.0]]);

        let expected = arr1(
            &[image[[0, 0]] * (1.0 - 0.0) * (1.0 - 0.1) +
              image[[1, 0]] * (1.0 - 0.0) * (0.1 - 0.0),
              image[[0, 0]] * (1.0 - 0.1) * (1.0 - 0.0) +
              image[[0, 1]] * (0.1 - 0.0) * (1.0 - 0.0)]
        );
        assert_eq!(image.interpolate(&c), expected);
    }
}
