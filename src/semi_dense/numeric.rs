extern crate derive_more;

use crate::numeric;
use derive_more::{Neg, From, Into};
use num_derive::{Float, Num, NumCast, NumOps, One, ToPrimitive, Zero};

#[derive(
    Copy, Clone, Debug, Float, From, Into, Neg, Num, NumCast, NumOps,
    One, PartialEq, PartialOrd, ToPrimitive, Zero
)]
pub struct Inv(f64);

pub trait Inverse<A> {
    fn inv(self) -> A;
}

impl Inverse<Inv> for f64 {
    fn inv(self) -> Inv {
        Inv::from(numeric::safe_invert(self))
    }
}

impl Inverse<f64> for Inv {
    fn inv(self) -> f64 {
        numeric::safe_invert(f64::from(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inv() {
        assert_eq!(Inv::from(10.0).inv(), 0.1);
        assert_eq!((10.0).inv(), Inv::from(0.1));
    }
}
