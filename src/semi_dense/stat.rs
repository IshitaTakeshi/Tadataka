use crate::semi_dense::numeric::Inv;

static FACTOR: f64 = 2.0;

pub fn is_statically_same(
    inv_depth1: Inv,
    inv_depth2: Inv,
    variance: f64,
) -> bool {
    // equivalent_condition:
    //   np.abs(inv_depth1-inv_depth2) <= factor * np.sqrt(variance1)
    let ds = (inv_depth1 - inv_depth2) * (inv_depth1 - inv_depth2);
    let fs = FACTOR * FACTOR;
    f64::from(ds) <= fs * variance
}

pub fn are_statically_same(
    inv_depth1: Inv,
    inv_depth2: Inv,
    variance1: f64,
    variance2: f64,
) -> bool {
    let c1 = is_statically_same(inv_depth1, inv_depth2, variance1);
    let c2 = is_statically_same(inv_depth1, inv_depth2, variance2);
    c1 && c2
}

#[cfg(tests)]
mod tests {
    use super::*;
    use rand;

    #[test]
    fn test_is_statically_same() {
        fn equivalent_condition(v1: f64, v2: f64, var: f64) {
            f64::abs(v1-v2) <= FACTOR * f64::sqrt(var)
        }

        let factor = 2.0;
        for i in 0..100 {
            let mut rng = rand::thread_rng();
            let value1 = rng.gen_range(-100., 100.);
            let value2 = rng.gen_range(-100., 100.);
            let variance1 = rng.gen_range(0., 50.);
            let variance2 = rng.gen_range(0., 50.);

            let c = is_statically_same(value1, value2, variance1);
            assert_eq!(c, equivalent_condition(value1, value2, variance1));

            let c = are_statically_same(value1, value2, variance1, variance2);
            let c1 = equivalent_condition(value1, value2, variance1);
            let c2 = equivalent_condition(value1, value2, variance2);
            assert_eq!(c, (c1 && c2))
        }
    }
}
