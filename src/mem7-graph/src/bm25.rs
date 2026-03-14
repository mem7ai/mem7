use std::collections::HashMap;

use crate::types::GraphSearchResult;

const K1: f64 = 1.2;
const B: f64 = 0.75;

fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(String::from)
        .collect()
}

fn triple_to_text(r: &GraphSearchResult) -> String {
    format!("{} {} {}", r.source, r.relationship, r.destination)
}

/// Rerank graph search results using BM25 scoring against the query.
pub fn rerank(results: &[GraphSearchResult], query: &str, top_n: usize) -> Vec<GraphSearchResult> {
    if results.is_empty() {
        return Vec::new();
    }

    let query_tokens = tokenize(query);
    if query_tokens.is_empty() {
        return results.iter().take(top_n).cloned().collect();
    }

    let docs: Vec<Vec<String>> = results
        .iter()
        .map(|r| tokenize(&triple_to_text(r)))
        .collect();
    let n = docs.len() as f64;
    let avg_dl: f64 = docs.iter().map(|d| d.len() as f64).sum::<f64>() / n;

    // Document frequency for each query term
    let mut df: HashMap<&str, usize> = HashMap::new();
    for qt in &query_tokens {
        let count = docs.iter().filter(|d| d.iter().any(|t| t == qt)).count();
        df.insert(qt.as_str(), count);
    }

    let mut scored: Vec<(usize, f64)> = results
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let doc = &docs[i];
            let dl = doc.len() as f64;
            let mut score = 0.0;

            for qt in &query_tokens {
                let tf = doc.iter().filter(|t| *t == qt).count() as f64;
                let doc_freq = *df.get(qt.as_str()).unwrap_or(&0) as f64;
                let idf = ((n - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).ln();
                let tf_norm = (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * dl / avg_dl));
                score += idf * tf_norm;
            }

            (i, score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scored
        .into_iter()
        .take(top_n)
        .map(|(i, bm25_score)| {
            let mut r = results[i].clone();
            r.score = Some(bm25_score as f32);
            r
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(src: &str, rel: &str, dst: &str) -> GraphSearchResult {
        GraphSearchResult {
            source: src.into(),
            relationship: rel.into(),
            destination: dst.into(),
            score: None,
        }
    }

    #[test]
    fn rerank_basic() {
        let results = vec![
            result("Alice", "lives_in", "London"),
            result("Bob", "works_at", "Google"),
            result("Alice", "works_at", "Microsoft"),
        ];

        let ranked = rerank(&results, "Alice works", 2);
        assert_eq!(ranked.len(), 2);
        // "Alice works_at Microsoft" matches both query tokens
        assert_eq!(ranked[0].source, "Alice");
        assert_eq!(ranked[0].relationship, "works_at");
    }

    #[test]
    fn rerank_empty() {
        let ranked = rerank(&[], "query", 5);
        assert!(ranked.is_empty());
    }

    #[test]
    fn rerank_respects_top_n() {
        let results = vec![
            result("A", "r1", "B"),
            result("C", "r2", "D"),
            result("E", "r3", "F"),
        ];
        let ranked = rerank(&results, "A B C D E F", 2);
        assert_eq!(ranked.len(), 2);
    }
}
