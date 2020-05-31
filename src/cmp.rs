use std::cmp::PartialOrd;

pub fn clamp<T: PartialOrd>(v: T, min: T, max: T) -> T {
    assert!(min <= max);
    if v < min {
        return min;
    }
    if v > max {
        return max;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(2., 1., 3.), 2.);
        assert_eq!(clamp(0., 1., 3.), 1.);
        assert_eq!(clamp(4., 1., 3.), 3.);
    }
}
