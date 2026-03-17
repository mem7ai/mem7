use mem7_core::MemoryItem;
use uuid::Uuid;

/// Build the JSON payload for a new memory record.
pub fn build_memory_payload(
    text: &str,
    user_id: Option<&str>,
    agent_id: Option<&str>,
    run_id: Option<&str>,
    metadata: Option<&serde_json::Value>,
    now: &str,
) -> serde_json::Value {
    let mut payload = serde_json::json!({
        "text": text,
        "user_id": user_id,
        "agent_id": agent_id,
        "run_id": run_id,
        "created_at": now,
        "updated_at": now,
        "last_accessed_at": now,
        "access_count": 0,
    });
    if let Some(meta) = metadata {
        payload["metadata"] = meta.clone();
    }
    payload
}

/// Build the JSON payload for a raw (non-inferred) memory, including the role.
pub fn build_raw_memory_payload(
    text: &str,
    role: &str,
    user_id: Option<&str>,
    agent_id: Option<&str>,
    run_id: Option<&str>,
    metadata: Option<&serde_json::Value>,
    now: &str,
) -> serde_json::Value {
    let mut payload = build_memory_payload(text, user_id, agent_id, run_id, metadata, now);
    payload["role"] = serde_json::Value::String(role.to_string());
    payload
}

/// Build the JSON payload for an updated memory record (no `created_at`).
pub fn build_update_payload(
    text: &str,
    user_id: Option<&str>,
    agent_id: Option<&str>,
    run_id: Option<&str>,
    metadata: Option<&serde_json::Value>,
    now: &str,
    access_count: u32,
) -> serde_json::Value {
    let mut payload = serde_json::json!({
        "text": text,
        "user_id": user_id,
        "agent_id": agent_id,
        "run_id": run_id,
        "updated_at": now,
        "last_accessed_at": now,
        "access_count": access_count,
    });
    if let Some(meta) = metadata {
        payload["metadata"] = meta.clone();
    }
    payload
}

/// Reconstitute a `MemoryItem` from vector store payload.
pub fn payload_to_memory_item(
    id: Uuid,
    payload: &serde_json::Value,
    score: Option<f32>,
) -> MemoryItem {
    MemoryItem {
        id,
        text: payload
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        user_id: payload
            .get("user_id")
            .and_then(|v| v.as_str())
            .map(String::from),
        agent_id: payload
            .get("agent_id")
            .and_then(|v| v.as_str())
            .map(String::from),
        run_id: payload
            .get("run_id")
            .and_then(|v| v.as_str())
            .map(String::from),
        metadata: payload
            .get("metadata")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
        created_at: payload
            .get("created_at")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        updated_at: payload
            .get("updated_at")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        score,
        last_accessed_at: payload
            .get("last_accessed_at")
            .and_then(|v| v.as_str())
            .map(String::from),
        access_count: payload
            .get("access_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_memory_payload_basic() {
        let p = build_memory_payload(
            "hello",
            Some("u1"),
            None,
            None,
            None,
            "2025-01-01T00:00:00Z",
        );
        assert_eq!(p["text"], "hello");
        assert_eq!(p["user_id"], "u1");
        assert_eq!(p["access_count"], 0);
        assert!(p.get("metadata").is_none());
    }

    #[test]
    fn build_memory_payload_with_metadata() {
        let meta = serde_json::json!({"key": "value"});
        let p = build_memory_payload("hi", None, None, None, Some(&meta), "now");
        assert_eq!(p["metadata"]["key"], "value");
    }

    #[test]
    fn raw_payload_includes_role() {
        let p = build_raw_memory_payload("hi", "user", None, None, None, None, "now");
        assert_eq!(p["role"], "user");
    }

    #[test]
    fn update_payload_has_no_created_at() {
        let p = build_update_payload("updated", None, None, None, None, "now", 3);
        assert!(p.get("created_at").is_none());
        assert_eq!(p["access_count"], 3);
    }

    #[test]
    fn payload_round_trip() {
        let now = "2025-06-15T10:30:00Z";
        let original = build_memory_payload("test text", Some("u1"), Some("a1"), None, None, now);
        let id = mem7_core::new_memory_id();
        let item = payload_to_memory_item(id, &original, Some(0.95));

        assert_eq!(item.id, id);
        assert_eq!(item.text, "test text");
        assert_eq!(item.user_id.as_deref(), Some("u1"));
        assert_eq!(item.agent_id.as_deref(), Some("a1"));
        assert!(item.run_id.is_none());
        assert_eq!(item.score, Some(0.95));
        assert_eq!(item.created_at, now);
        assert_eq!(item.access_count, 0);
    }

    #[test]
    fn payload_to_memory_item_handles_empty() {
        let payload = serde_json::json!({});
        let item = payload_to_memory_item(mem7_core::new_memory_id(), &payload, None);
        assert_eq!(item.text, "");
        assert!(item.user_id.is_none());
        assert!(item.score.is_none());
        assert_eq!(item.access_count, 0);
    }
}
