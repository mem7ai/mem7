use serde::de::DeserializeOwned;

/// Strip markdown code fences from an LLM response and deserialize as JSON.
///
/// Returns `Err(String)` with a descriptive message on failure so callers can
/// wrap it into their own `Mem7Error` variant.
pub fn parse_json_response<T: DeserializeOwned>(raw: &str) -> Result<T, String> {
    let trimmed = raw.trim();
    let cleaned = if trimmed.starts_with("```json") {
        trimmed
            .trim_start_matches("```json")
            .trim_end_matches("```")
            .trim()
    } else if trimmed.starts_with("```") {
        trimmed
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
    } else {
        trimmed
    };

    serde_json::from_str(cleaned).map_err(|e| format!("JSON parse error: {e}\nRaw: {raw}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    struct Sample {
        value: String,
    }

    #[test]
    fn plain_json() {
        let r: Sample = parse_json_response(r#"{"value": "ok"}"#).unwrap();
        assert_eq!(r.value, "ok");
    }

    #[test]
    fn json_with_code_fence() {
        let r: Sample = parse_json_response("```json\n{\"value\": \"fenced\"}\n```").unwrap();
        assert_eq!(r.value, "fenced");
    }

    #[test]
    fn json_with_generic_fence() {
        let r: Sample = parse_json_response("```\n{\"value\": \"generic\"}\n```").unwrap();
        assert_eq!(r.value, "generic");
    }

    #[test]
    fn bad_json_returns_err() {
        let r = parse_json_response::<Sample>("not json");
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("JSON parse error"));
    }
}
