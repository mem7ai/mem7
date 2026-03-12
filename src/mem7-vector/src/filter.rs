use mem7_core::MemoryFilter;
use serde_json::Value;

/// Check if a payload matches the given filter criteria.
///
/// First checks the top-level `user_id`, `agent_id`, `run_id` fields, then
/// evaluates the optional metadata filter DSL against `payload["metadata"]`.
pub fn matches_filter(payload: &Value, filter: &MemoryFilter) -> bool {
    if let Some(ref uid) = filter.user_id
        && payload.get("user_id").and_then(|v| v.as_str()) != Some(uid.as_str())
    {
        return false;
    }
    if let Some(ref aid) = filter.agent_id
        && payload.get("agent_id").and_then(|v| v.as_str()) != Some(aid.as_str())
    {
        return false;
    }
    if let Some(ref rid) = filter.run_id
        && payload.get("run_id").and_then(|v| v.as_str()) != Some(rid.as_str())
    {
        return false;
    }

    if let Some(ref meta_filter) = filter.metadata {
        let meta = payload.get("metadata").unwrap_or(&Value::Null);
        if !eval_filter_object(meta, meta_filter) {
            return false;
        }
    }

    true
}

/// Evaluate a filter JSON object against a metadata JSON object.
///
/// The filter object can contain:
/// - `{"key": "value"}` — simple equality
/// - `{"key": {"eq": v, "gt": v, ...}}` — operator-based
/// - `{"AND": [f1, f2, ...]}` — all must match
/// - `{"OR":  [f1, f2, ...]}` — any must match
/// - `{"NOT": [f1]}` — negation
fn eval_filter_object(metadata: &Value, filter: &Value) -> bool {
    let obj = match filter.as_object() {
        Some(o) => o,
        None => return true,
    };

    for (key, condition) in obj {
        match key.as_str() {
            "AND" => {
                if let Some(arr) = condition.as_array()
                    && !arr.iter().all(|f| eval_filter_object(metadata, f))
                {
                    return false;
                }
            }
            "OR" => {
                if let Some(arr) = condition.as_array()
                    && !arr.iter().any(|f| eval_filter_object(metadata, f))
                {
                    return false;
                }
            }
            "NOT" => {
                if let Some(arr) = condition.as_array()
                    && arr.iter().any(|f| eval_filter_object(metadata, f))
                {
                    return false;
                }
            }
            field => {
                let payload_val = metadata.get(field).unwrap_or(&Value::Null);
                if !eval_condition(payload_val, condition) {
                    return false;
                }
            }
        }
    }

    true
}

/// Evaluate a single condition against a payload value.
///
/// If `condition` is a plain value (string, number, bool, null), performs
/// equality check. If it's an object with operator keys, dispatches to the
/// appropriate comparator.
fn eval_condition(payload_val: &Value, condition: &Value) -> bool {
    match condition {
        Value::Object(ops) => {
            for (op, expected) in ops {
                let matched = match op.as_str() {
                    "eq" => values_equal(payload_val, expected),
                    "ne" => !values_equal(payload_val, expected),
                    "gt" => {
                        compare_numbers(payload_val, expected) == Some(std::cmp::Ordering::Greater)
                    }
                    "gte" => matches!(
                        compare_numbers(payload_val, expected),
                        Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                    ),
                    "lt" => {
                        compare_numbers(payload_val, expected) == Some(std::cmp::Ordering::Less)
                    }
                    "lte" => matches!(
                        compare_numbers(payload_val, expected),
                        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                    ),
                    "in" => match expected.as_array() {
                        Some(arr) => arr.iter().any(|v| values_equal(payload_val, v)),
                        None => false,
                    },
                    "nin" => match expected.as_array() {
                        Some(arr) => !arr.iter().any(|v| values_equal(payload_val, v)),
                        None => true,
                    },
                    "contains" => match (payload_val.as_str(), expected.as_str()) {
                        (Some(hay), Some(needle)) => hay.contains(needle),
                        _ => false,
                    },
                    "icontains" => match (payload_val.as_str(), expected.as_str()) {
                        (Some(hay), Some(needle)) => {
                            hay.to_lowercase().contains(&needle.to_lowercase())
                        }
                        _ => false,
                    },
                    _ => true,
                };
                if !matched {
                    return false;
                }
            }
            true
        }
        _ => values_equal(payload_val, condition),
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    if a == b {
        return true;
    }
    match (to_f64(a), to_f64(b)) {
        (Some(fa), Some(fb)) => (fa - fb).abs() < f64::EPSILON,
        _ => false,
    }
}

fn to_f64(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_i64().map(|i| i as f64))
}

fn compare_numbers(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    let fa = to_f64(a)?;
    let fb = to_f64(b)?;
    fa.partial_cmp(&fb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_payload(metadata: Value) -> Value {
        json!({
            "text": "test",
            "user_id": "alice",
            "metadata": metadata,
        })
    }

    #[test]
    fn test_legacy_user_id_filter() {
        let payload = make_payload(json!({}));
        let filter = MemoryFilter {
            user_id: Some("alice".into()),
            ..Default::default()
        };
        assert!(matches_filter(&payload, &filter));

        let filter2 = MemoryFilter {
            user_id: Some("bob".into()),
            ..Default::default()
        };
        assert!(!matches_filter(&payload, &filter2));
    }

    #[test]
    fn test_simple_equality() {
        let payload = make_payload(json!({"status": "active", "priority": 5}));
        let filter = MemoryFilter {
            metadata: Some(json!({"status": "active"})),
            ..Default::default()
        };
        assert!(matches_filter(&payload, &filter));

        let filter2 = MemoryFilter {
            metadata: Some(json!({"status": "inactive"})),
            ..Default::default()
        };
        assert!(!matches_filter(&payload, &filter2));
    }

    #[test]
    fn test_eq_operator() {
        let payload = make_payload(json!({"count": 10}));
        let filter = MemoryFilter {
            metadata: Some(json!({"count": {"eq": 10}})),
            ..Default::default()
        };
        assert!(matches_filter(&payload, &filter));

        let filter2 = MemoryFilter {
            metadata: Some(json!({"count": {"eq": 5}})),
            ..Default::default()
        };
        assert!(!matches_filter(&payload, &filter2));
    }

    #[test]
    fn test_ne_operator() {
        let payload = make_payload(json!({"status": "active"}));
        let filter = MemoryFilter {
            metadata: Some(json!({"status": {"ne": "deleted"}})),
            ..Default::default()
        };
        assert!(matches_filter(&payload, &filter));

        let filter2 = MemoryFilter {
            metadata: Some(json!({"status": {"ne": "active"}})),
            ..Default::default()
        };
        assert!(!matches_filter(&payload, &filter2));
    }

    #[test]
    fn test_gt_gte_lt_lte() {
        let payload = make_payload(json!({"score": 75}));

        assert!(matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"score": {"gt": 50}})),
                ..Default::default()
            }
        ));
        assert!(!matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"score": {"gt": 75}})),
                ..Default::default()
            }
        ));
        assert!(matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"score": {"gte": 75}})),
                ..Default::default()
            }
        ));
        assert!(matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"score": {"lt": 100}})),
                ..Default::default()
            }
        ));
        assert!(!matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"score": {"lt": 75}})),
                ..Default::default()
            }
        ));
        assert!(matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"score": {"lte": 75}})),
                ..Default::default()
            }
        ));
    }

    #[test]
    fn test_in_nin() {
        let payload = make_payload(json!({"tag": "rust"}));

        assert!(matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"tag": {"in": ["rust", "python"]}})),
                ..Default::default()
            }
        ));
        assert!(!matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"tag": {"in": ["go", "java"]}})),
                ..Default::default()
            }
        ));
        assert!(matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"tag": {"nin": ["go", "java"]}})),
                ..Default::default()
            }
        ));
        assert!(!matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"tag": {"nin": ["rust", "python"]}})),
                ..Default::default()
            }
        ));
    }

    #[test]
    fn test_contains() {
        let payload = make_payload(json!({"description": "Alice plays tennis every week"}));

        assert!(matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"description": {"contains": "tennis"}})),
                ..Default::default()
            }
        ));
        assert!(!matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"description": {"contains": "basketball"}})),
                ..Default::default()
            }
        ));
    }

    #[test]
    fn test_icontains() {
        let payload = make_payload(json!({"name": "Alice Smith"}));

        assert!(matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"name": {"icontains": "alice"}})),
                ..Default::default()
            }
        ));
        assert!(matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"name": {"icontains": "SMITH"}})),
                ..Default::default()
            }
        ));
    }

    #[test]
    fn test_and_combinator() {
        let payload = make_payload(json!({"status": "active", "priority": 5}));

        assert!(matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"AND": [
                    {"status": "active"},
                    {"priority": {"gte": 3}}
                ]})),
                ..Default::default()
            }
        ));
        assert!(!matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"AND": [
                    {"status": "active"},
                    {"priority": {"gt": 10}}
                ]})),
                ..Default::default()
            }
        ));
    }

    #[test]
    fn test_or_combinator() {
        let payload = make_payload(json!({"status": "pending"}));

        assert!(matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"OR": [
                    {"status": "active"},
                    {"status": "pending"}
                ]})),
                ..Default::default()
            }
        ));
        assert!(!matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"OR": [
                    {"status": "active"},
                    {"status": "deleted"}
                ]})),
                ..Default::default()
            }
        ));
    }

    #[test]
    fn test_not_combinator() {
        let payload = make_payload(json!({"status": "active"}));

        assert!(matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"NOT": [{"status": "deleted"}]})),
                ..Default::default()
            }
        ));
        assert!(!matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"NOT": [{"status": "active"}]})),
                ..Default::default()
            }
        ));
    }

    #[test]
    fn test_nested_combinators() {
        let payload = make_payload(json!({"status": "active", "priority": 5, "tag": "rust"}));

        let filter = MemoryFilter {
            metadata: Some(json!({"AND": [
                {"status": "active"},
                {"OR": [
                    {"tag": "rust"},
                    {"tag": "python"}
                ]}
            ]})),
            ..Default::default()
        };
        assert!(matches_filter(&payload, &filter));
    }

    #[test]
    fn test_missing_key_no_match() {
        let payload = make_payload(json!({"status": "active"}));

        assert!(!matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"nonexistent": "value"})),
                ..Default::default()
            }
        ));
    }

    #[test]
    fn test_empty_metadata_filter() {
        let payload = make_payload(json!({"status": "active"}));

        assert!(matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({})),
                ..Default::default()
            }
        ));
    }

    #[test]
    fn test_null_metadata_in_payload() {
        let payload = json!({"text": "test", "user_id": "alice"});

        assert!(matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({})),
                ..Default::default()
            }
        ));
        assert!(!matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"status": "active"})),
                ..Default::default()
            }
        ));
    }

    #[test]
    fn test_numeric_int_float_equality() {
        let payload = make_payload(json!({"score": 10}));

        assert!(matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"score": 10.0})),
                ..Default::default()
            }
        ));
        assert!(matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"score": {"eq": 10.0}})),
                ..Default::default()
            }
        ));
    }

    #[test]
    fn test_multiple_operators_on_same_field() {
        let payload = make_payload(json!({"score": 75}));

        assert!(matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"score": {"gte": 50, "lte": 100}})),
                ..Default::default()
            }
        ));
        assert!(!matches_filter(
            &payload,
            &MemoryFilter {
                metadata: Some(json!({"score": {"gte": 80, "lte": 100}})),
                ..Default::default()
            }
        ));
    }

    #[test]
    fn test_combined_first_class_and_metadata() {
        let payload = json!({
            "text": "test",
            "user_id": "alice",
            "metadata": {"status": "active"},
        });

        let filter = MemoryFilter {
            user_id: Some("alice".into()),
            metadata: Some(json!({"status": "active"})),
            ..Default::default()
        };
        assert!(matches_filter(&payload, &filter));

        let filter2 = MemoryFilter {
            user_id: Some("bob".into()),
            metadata: Some(json!({"status": "active"})),
            ..Default::default()
        };
        assert!(!matches_filter(&payload, &filter2));
    }
}
