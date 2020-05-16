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
