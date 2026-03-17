use std::time::{SystemTime, UNIX_EPOCH};

use mem7_config::DecayConfig;

/// Compute how many seconds have elapsed since the given ISO 8601 timestamp.
/// Returns 0.0 if parsing fails or the timestamp is in the future.
fn age_from_iso(ts: &str) -> f64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    mem7_datetime::iso_to_epoch(ts).map_or(0.0, |epoch| (now - epoch).max(0.0))
}

/// Compute the retention factor for a memory given its age and access count.
///
/// ```text
/// stability = base_half_life * (1 + rehearsal_factor * ln(1 + access_count))
/// retention = exp(-((age / stability) ^ decay_shape))
/// effective = min_retention + (1 - min_retention) * retention
/// ```
pub fn retention(age_secs: f64, access_count: u32, config: &DecayConfig) -> f64 {
    if age_secs <= 0.0 {
        return 1.0;
    }

    let stability = config.base_half_life_secs
        * (1.0 + config.rehearsal_factor * (1.0 + access_count as f64).ln());

    let ratio = age_secs / stability;
    let decay = (-(ratio.powf(config.decay_shape))).exp();

    config.min_retention + (1.0 - config.min_retention) * decay
}

/// Apply time-based decay to a raw similarity/rerank score.
pub fn apply_decay(raw_score: f32, age_secs: f64, access_count: u32, config: &DecayConfig) -> f32 {
    (raw_score as f64 * retention(age_secs, access_count, config)) as f32
}

/// Compute age in seconds from a payload JSON value, falling back through
/// `last_accessed_at` → `updated_at` → `created_at`.
pub fn age_from_payload(payload: &serde_json::Value) -> f64 {
    let ts = payload
        .get("last_accessed_at")
        .and_then(|v| v.as_str())
        .or_else(|| payload.get("updated_at").and_then(|v| v.as_str()))
        .or_else(|| payload.get("created_at").and_then(|v| v.as_str()))
        .unwrap_or("");
    age_from_iso(ts)
}

/// Extract `access_count` from a payload, defaulting to 0 for old records.
pub fn access_count_from_payload(payload: &serde_json::Value) -> u32 {
    payload
        .get("access_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32
}

/// Compute age in seconds for a `MemoryItem`, falling back through
/// `last_accessed_at` → `updated_at` → `created_at`.
pub fn age_from_memory_item(
    last_accessed_at: Option<&str>,
    updated_at: &str,
    created_at: &str,
) -> f64 {
    let ts = last_accessed_at
        .filter(|s| !s.is_empty())
        .unwrap_or(if !updated_at.is_empty() {
            updated_at
        } else {
            created_at
        });
    age_from_iso(ts)
}

/// Compute age from an optional ISO timestamp string (for graph results).
pub fn age_from_option(ts: Option<&str>) -> f64 {
    ts.map_or(0.0, age_from_iso)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> DecayConfig {
        DecayConfig {
            enabled: true,
            base_half_life_secs: 604800.0,
            decay_shape: 0.8,
            min_retention: 0.1,
            rehearsal_factor: 0.5,
        }
    }

    #[test]
    fn zero_age_returns_full_retention() {
        let r = retention(0.0, 0, &test_config());
        assert!((r - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn retention_decreases_over_time() {
        let cfg = test_config();
        let r1 = retention(86400.0, 0, &cfg); // 1 day
        let r7 = retention(604800.0, 0, &cfg); // 7 days
        let r30 = retention(2592000.0, 0, &cfg); // 30 days

        assert!(r1 > r7, "1d={r1} should be > 7d={r7}");
        assert!(r7 > r30, "7d={r7} should be > 30d={r30}");
        assert!(
            r30 >= cfg.min_retention,
            "30d={r30} should be >= floor={}",
            cfg.min_retention
        );
    }

    #[test]
    fn rehearsal_increases_stability() {
        let cfg = test_config();
        let age = 604800.0; // 7 days
        let r0 = retention(age, 0, &cfg);
        let r5 = retention(age, 5, &cfg);
        let r20 = retention(age, 20, &cfg);

        assert!(r5 > r0, "5 accesses should retain more than 0");
        assert!(r20 > r5, "20 accesses should retain more than 5");
    }

    #[test]
    fn floor_prevents_total_decay() {
        let cfg = test_config();
        let r = retention(86400.0 * 365.0, 0, &cfg); // 1 year, no rehearsal
        assert!(r >= cfg.min_retention);
        assert!(r < cfg.min_retention + 0.01);
    }

    #[test]
    fn apply_decay_scales_score() {
        let cfg = test_config();
        let raw = 0.9_f32;
        let decayed = apply_decay(raw, 604800.0, 0, &cfg);
        assert!(decayed < raw);
        assert!(decayed > 0.0);
    }

    #[test]
    fn parse_iso_round_trips() {
        let epoch = mem7_datetime::iso_to_epoch("2025-01-01T00:00:00Z");
        assert!(epoch.is_some());
        assert!((epoch.unwrap() - 1735689600.0).abs() < 1.0);
    }

    #[test]
    fn age_from_payload_fallback_chain() {
        let p1 = serde_json::json!({"last_accessed_at": "2025-01-01T00:00:00Z"});
        assert!(age_from_payload(&p1) > 0.0);

        let p2 = serde_json::json!({"updated_at": "2025-01-01T00:00:00Z"});
        assert!(age_from_payload(&p2) > 0.0);

        let p3 = serde_json::json!({"created_at": "2025-01-01T00:00:00Z"});
        assert!(age_from_payload(&p3) > 0.0);

        let p4 = serde_json::json!({});
        assert!((age_from_payload(&p4) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn access_count_defaults_to_zero() {
        let p = serde_json::json!({});
        assert_eq!(access_count_from_payload(&p), 0);

        let p = serde_json::json!({"access_count": 7});
        assert_eq!(access_count_from_payload(&p), 7);
    }
}
