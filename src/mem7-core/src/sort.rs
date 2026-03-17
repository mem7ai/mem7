use std::cmp::Ordering;

/// Sort a slice in-place by a score function, descending (highest first).
/// NaN values are treated as less than any number.
pub fn sort_by_score_desc<T>(items: &mut [T], score_fn: impl Fn(&T) -> f32) {
    items.sort_by(|a, b| {
        let sa = score_fn(a);
        let sb = score_fn(b);
        sb.partial_cmp(&sa).unwrap_or_else(|| {
            // NaN values sink to the end
            match (sa.is_nan(), sb.is_nan()) {
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                _ => Ordering::Equal,
            }
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_descending() {
        let mut v = vec![1.0_f32, 3.0, 2.0];
        sort_by_score_desc(&mut v, |x| *x);
        assert_eq!(v, vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn handles_nan() {
        let mut v = vec![f32::NAN, 1.0, 2.0];
        sort_by_score_desc(&mut v, |x| *x);
        assert_eq!(v[0], 2.0);
        assert_eq!(v[1], 1.0);
    }
}
