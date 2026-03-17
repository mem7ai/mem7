/// Distance metric for vector similarity.
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Cosine,
    DotProduct,
    Euclidean,
}

impl DistanceMetric {
    /// Compute similarity between two vectors. Higher = more similar.
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::Cosine => cosine_similarity(a, b),
            Self::DotProduct => dot_product(a, b),
            Self::Euclidean => {
                let d = euclidean_distance(a, b);
                1.0 / (1.0 + d)
            }
        }
    }
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = norm_a * norm_b;
    if denom == 0.0 { 0.0 } else { dot / denom }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let sim = DistanceMetric::Cosine.similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = DistanceMetric::Cosine.similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }
}
