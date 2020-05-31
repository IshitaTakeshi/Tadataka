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

impl Hypothesis {
    pub fn new(
        inv_depth: Inv,
        variance: f64,
        valid_range: (Inv, Inv)
    ) -> Result<Hypothesis, Flag> {
        if f64::from(inv_depth) <= 0. {
            return Err(Flag::NegativePriorDepth);
        }

        let (valid_min, valid_max) = valid_range;

        let h = Hypothesis {
            inv_depth: inv_depth,
            variance: variance,
            valid_min: valid_min,
            valid_max: valid_max
        };

        let (min, max) = h._range();
        if max <= f64::from(valid_min) || f64::from(valid_max) <= min {
            return Err(Flag::HypothesisOutOfSerchRange);
        }

        Ok(h)
    }

    fn _range(&self) -> (f64, f64) {
        let min = f64::from(self.inv_depth) - VARINACE_FACTOR * self.variance;
        let max = f64::from(self.inv_depth) + VARINACE_FACTOR * self.variance;
        (min, max)
    }

    pub fn range(&self) -> (Inv, Inv) {
        let (min, max) = self._range();
        let valid_min = f64::from(self.valid_min);
        let valid_max = f64::from(self.valid_max);
        let min = Inv::from(clamp(min, valid_min, valid_max));
        let max = Inv::from(clamp(max, valid_min, valid_max));
        (min, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_new() {
        // predicted range fits in [min, max]
        let range = (Inv::from(0.4), Inv::from(1.0));
        let (min, max) = Hypothesis::new(Inv::from(0.7), 0.1, range).unwrap().range();
        approx::assert_abs_diff_eq!(f64::from(min), 0.5);
        approx::assert_abs_diff_eq!(f64::from(max), 0.9);

        // inv_depth - factor * variance < min < inv_depth + factor * variance
        let (min, max) = Hypothesis::new(Inv::from(0.3), 0.1, range).unwrap().range();
        approx::assert_abs_diff_eq!(f64::from(min), 0.4);
        approx::assert_abs_diff_eq!(f64::from(max), 0.5);

        // inv_depth - factor * variance < max < inv_depth + factor * variance
        let (min, max) = Hypothesis::new(Inv::from(0.9), 0.1, range).unwrap().range();
        approx::assert_abs_diff_eq!(f64::from(min), 0.7);
        approx::assert_abs_diff_eq!(f64::from(max), 1.0);

        let flag = Flag::HypothesisOutOfSerchRange;

        // inv_depth - factor * variance < inv_depth + factor * variance < min
        assert_eq!(Hypothesis::new(Inv::from(0.1), 0.1, range).unwrap_err(), flag);
        // max < inv_depth - factor * variance < inv_depth + factor * variance
        assert_eq!(Hypothesis::new(Inv::from(1.5), 0.1, range).unwrap_err(), flag);

        // inv_depth - factor * variance < inv_depth + factor * variance == min
        assert_eq!(Hypothesis::new(Inv::from(0.2), 0.1, range).unwrap_err(), flag);
        // max == inv_depth - factor * variance < inv_depth + factor * variance
        assert_eq!(Hypothesis::new(Inv::from(1.2), 0.1, range).unwrap_err(), flag);
    }
}
