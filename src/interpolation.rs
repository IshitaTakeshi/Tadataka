use ndarray::{arr1, arr2, ArrayView1, ArrayView2};

pub fn interpolation(image: ArrayView2<'_, f64>, coordinate: ArrayView1<'_, f64>) -> f64 {
    let cx = coordinate[0];
    let cy = coordinate[1];
    let lx = cx.floor();
    let ly = cy.floor();
    let lxi = lx as usize;
    let lyi = ly as usize;

    if lx == cx && ly == cy {
        return image[[lyi, lxi]];
    }

    let ux = lx + 1.0;
    let uy = ly + 1.0;
    let uxi = ux as usize;
    let uyi = uy as usize;

    if lx == cx {
        return image[[lyi, lxi]] * (ux - cx) * (uy - cy)
            + image[[uyi, lxi]] * (ux - cx) * (cy - ly);
    }

    if ly == cy {
        return image[[lyi, lxi]] * (ux - cx) * (uy - cy)
            + image[[lyi, uxi]] * (cx - lx) * (uy - cy);
    }

    image[[lyi, lxi]] * (ux - cx) * (uy - cy)
        + image[[lyi, uxi]] * (cx - lx) * (uy - cy)
        + image[[uyi, lxi]] * (ux - cx) * (cy - ly)
        + image[[uyi, uxi]] * (cx - lx) * (cy - ly)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolation() {
        let image = arr2(
            &[[0., 1., 5.],
              [0., 0., 2.],
              [4., 3., 2.],
              [5., 6., 1.]]
        );

        let expected = image[[2, 1]] * (2.0 - 1.3) * (3.0 - 2.6) +
                       image[[2, 2]] * (1.3 - 1.0) * (3.0 - 2.6) +
                       image[[3, 1]] * (2.0 - 1.3) * (2.6 - 2.0) +
                       image[[3, 2]] * (1.3 - 1.0) * (2.6 - 2.0);

        let c = arr1(&[1.3, 2.6]);
        assert_eq!(interpolation(image.view(), c.view()), expected);

        // minimum coordinate
        let c = arr1(&[0.0, 0.0]);
        assert_eq!(interpolation(image.view(), c.view()), image[[0, 0]]);

        // minimum x
        let c = arr1(&[0.0, 0.1]);
        let expected = image[[0, 0]] * (1.0 - 0.0) * (1.0 - 0.1) +
                       image[[1, 0]] * (1.0 - 0.0) * (0.1 - 0.0);
        assert_eq!(interpolation(image.view(), c.view()), expected);

        // minimum y
        let c = arr1(&[0.1, 0.0]);
        let expected = image[[0, 0]] * (1.0 - 0.1) * (1.0 - 0.0) +
                       image[[0, 1]] * (0.1 - 0.0) * (1.0 - 0.0);
        assert_eq!(interpolation(image.view(), c.view()), expected);

        // maximum x
        let c = arr1(&[2.0, 2.9]);
        let expected = image[[2, 2]] * (3.0 - 2.0) * (3.0 - 2.9) +
                       image[[3, 2]] * (3.0 - 2.0) * (2.9 - 2.0);
        assert_eq!(interpolation(image.view(), c.view()), expected);

        // maximum y
        let c = arr1(&[1.9, 3.0]);
        let expected = image[[3, 1]] * (2.0 - 1.9) * (4.0 - 3.0) +
                       image[[3, 2]] * (1.9 - 1.0) * (4.0 - 3.0);
        assert_eq!(interpolation(image.view(), c.view()), expected);

        // maximum c
        let c = arr1(&[2.0, 3.0]);
        assert_eq!(interpolation(image.view(), c.view()), image[[3, 2]]);

        // HACK How about invalid input?
        // interpolation(image, [[3.0, 2.01]])
        // interpolation(image, [[3.01, 2.0]])
        // interpolation(image, [[-0.01, 0.0]])
        // interpolation(image, [[0.0, -0.01]])
    }
}
