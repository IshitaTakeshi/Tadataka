static EPSILON: f64 = f64::EPSILON;

pub fn safe_invert(v: f64) -> f64 {
    1. / (v + EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_invert() {
        assert_eq!(safe_invert(10.0), 0.1);
        assert_eq!(safe_invert(0.0), 1. / EPSILON);
    }
}
