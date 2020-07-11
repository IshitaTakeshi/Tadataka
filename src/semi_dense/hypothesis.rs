use crate::cmp::clamp;
use super::flag::Flag;
use super::numeric::Inv;

static VARINACE_FACTOR: f64 = 2.0;

#[derive(Debug)]
pub struct Hypothesis {
    pub inv_depth: Inv,
    pub variance: f64,
    valid_min: Inv,
    valid_max: Inv,
}

fn range(inv_depth: Inv, variance: f64) -> (f64, f64) {
    let min = f64::from(inv_depth) - VARINACE_FACTOR * variance;
    let max = f64::from(inv_depth) + VARINACE_FACTOR * variance;
    (min, max)
}

pub fn check_args(
    inv_depth: Inv,
    variance: f64,
    inv_depth_range: (Inv, Inv)
) -> Result<(), Flag> {
    if f64::from(inv_depth) <= 0. {
        return Err(Flag::NegativePriorDepth);
    }

    let (valid_min, valid_max) = inv_depth_range;
    let (min, max) = range(inv_depth, variance);
    if max <= f64::from(valid_min) || f64::from(valid_max) <= min {
        return Err(Flag::HypothesisOutOfSerchRange);
    }

    Ok(())
}

impl Hypothesis {
    pub fn new(
        inv_depth: Inv,
        variance: f64,
        inv_depth_range: (Inv, Inv),
    ) -> Hypothesis {
        let (valid_min, valid_max) = inv_depth_range;
        Hypothesis {
            inv_depth: inv_depth,
            variance: variance,
            valid_min: valid_min,
            valid_max: valid_max
        }
    }

    pub fn range(&self) -> (Inv, Inv) {
        let (min, max) = range(self.inv_depth, self.variance);
        let valid_min = f64::from(self.valid_min);
        let valid_max = f64::from(self.valid_max);
        let min = Inv::from(clamp(min, valid_min, valid_max));
        let max = Inv::from(clamp(max, valid_min, valid_max));
        (min, max)
    }
}

pub fn try_make_hypothesis(
    inv_depth: Inv,
    variance: f64,
    inv_depth_range: (Inv, Inv)
) -> Result<Hypothesis, Flag> {
    check_args(inv_depth, variance, inv_depth_range)?;
    Ok(Hypothesis::new(inv_depth, variance, inv_depth_range))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_new() {
        // predicted range fits in [min, max]
        let range = (Inv::from(0.4), Inv::from(1.0));
        let (min, max) = Hypothesis::new(Inv::from(0.7), 0.1, range).range();
        approx::assert_abs_diff_eq!(f64::from(min), 0.5);
        approx::assert_abs_diff_eq!(f64::from(max), 0.9);

        // inv_depth - factor * variance < min < inv_depth + factor * variance
        let (min, max) = Hypothesis::new(Inv::from(0.3), 0.1, range).range();
        approx::assert_abs_diff_eq!(f64::from(min), 0.4);
        approx::assert_abs_diff_eq!(f64::from(max), 0.5);

        // inv_depth - factor * variance < max < inv_depth + factor * variance
        let (min, max) = Hypothesis::new(Inv::from(0.9), 0.1, range).range();
        approx::assert_abs_diff_eq!(f64::from(min), 0.7);
        approx::assert_abs_diff_eq!(f64::from(max), 1.0);

        let flag = Flag::HypothesisOutOfSerchRange;

        // inv_depth - factor * variance < inv_depth + factor * variance < min
        assert_eq!(check_args(Inv::from(0.1), 0.1, range).unwrap_err(), flag);
        // max < inv_depth - factor * variance < inv_depth + factor * variance
        assert_eq!(check_args(Inv::from(1.5), 0.1, range).unwrap_err(), flag);

        // inv_depth - factor * variance < inv_depth + factor * variance == min
        assert_eq!(check_args(Inv::from(0.2), 0.1, range).unwrap_err(), flag);
        // max == inv_depth - factor * variance < inv_depth + factor * variance
        assert_eq!(check_args(Inv::from(1.2), 0.1, range).unwrap_err(), flag);
    }
}
