use ndarray::ArrayView2;

pub fn interpolation(image: ArrayView2<'_, f64>, cx: f64, cy: f64) -> f64 {
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
