use mem7_core::{MemoryEventMetadata, MemoryItem};
use uuid::Uuid;

fn metadata_str_field(metadata: &serde_json::Value, key: &str) -> Option<String> {
    metadata.get(key).and_then(|v| v.as_str()).map(String::from)
}

struct PromotedMetadata {
    metadata: serde_json::Value,
    user_id: Option<String>,
    agent_id: Option<String>,
    run_id: Option<String>,
    actor_id: Option<String>,
    role: Option<String>,
}

fn split_promoted_metadata(payload: &serde_json::Value) -> PromotedMetadata {
    let mut metadata = payload
        .get("metadata")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    let mut metadata_user_id = None;
    let mut metadata_agent_id = None;
    let mut metadata_run_id = None;
    let mut actor_id = None;
    let mut role = None;

    if let Some(obj) = metadata.as_object_mut() {
        metadata_user_id = obj
            .remove("user_id")
            .and_then(|v| v.as_str().map(String::from));
        metadata_agent_id = obj
            .remove("agent_id")
            .and_then(|v| v.as_str().map(String::from));
        metadata_run_id = obj
            .remove("run_id")
            .and_then(|v| v.as_str().map(String::from));
        actor_id = obj
            .remove("actor_id")
            .and_then(|v| v.as_str().map(String::from));
        role = obj
            .remove("role")
            .and_then(|v| v.as_str().map(String::from));

        if obj.is_empty() {
            metadata = serde_json::Value::Null;
        }
    }

    PromotedMetadata {
        metadata,
        user_id: metadata_user_id,
        agent_id: metadata_agent_id,
        run_id: metadata_run_id,
        actor_id,
        role,
    }
}

/// Build the JSON payload for a new memory record.
pub fn build_memory_payload(
    text: &str,
    user_id: Option<&str>,
    agent_id: Option<&str>,
    run_id: Option<&str>,
    metadata: Option<&serde_json::Value>,
    now: &str,
    memory_type: Option<&str>,
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
    if let Some(mt) = memory_type {
        payload["memory_type"] = serde_json::Value::String(mt.to_string());
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
    let mut payload = build_memory_payload(text, user_id, agent_id, run_id, metadata, now, None);
    payload["role"] = serde_json::Value::String(role.to_string());
    payload
}

/// Build the JSON payload for an updated memory record (no `created_at`).
#[allow(clippy::too_many_arguments)]
pub fn build_update_payload(
    text: &str,
    user_id: Option<&str>,
    agent_id: Option<&str>,
    run_id: Option<&str>,
    metadata: Option<&serde_json::Value>,
    created_at: Option<&str>,
    now: &str,
    access_count: u32,
    memory_type: Option<&str>,
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
    if let Some(created_at) = created_at {
        payload["created_at"] = serde_json::Value::String(created_at.to_string());
    }
    if let Some(mt) = memory_type {
        payload["memory_type"] = serde_json::Value::String(mt.to_string());
    }
    payload
}

/// Reconstitute a `MemoryItem` from vector store payload.
pub fn payload_to_memory_item(
    id: Uuid,
    payload: &serde_json::Value,
    score: Option<f32>,
) -> MemoryItem {
    let promoted = split_promoted_metadata(payload);

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
            .map(String::from)
            .or(promoted.user_id),
        agent_id: payload
            .get("agent_id")
            .and_then(|v| v.as_str())
            .map(String::from)
            .or(promoted.agent_id),
        run_id: payload
            .get("run_id")
            .and_then(|v| v.as_str())
            .map(String::from)
            .or(promoted.run_id),
        actor_id: payload
            .get("actor_id")
            .and_then(|v| v.as_str())
            .map(String::from)
            .or(promoted.actor_id),
        role: payload
            .get("role")
            .and_then(|v| v.as_str())
            .map(String::from)
            .or(promoted.role),
        metadata: promoted.metadata,
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
        memory_type: payload
            .get("memory_type")
            .and_then(|v| v.as_str())
            .map(String::from),
    }
}

/// Extract audit metadata from a stored payload for history records.
pub fn payload_to_event_metadata(payload: &serde_json::Value) -> MemoryEventMetadata {
    let metadata = payload.get("metadata");

    MemoryEventMetadata {
        created_at: payload
            .get("created_at")
            .and_then(|v| v.as_str())
            .map(String::from),
        updated_at: payload
            .get("updated_at")
            .and_then(|v| v.as_str())
            .map(String::from),
        is_deleted: false,
        actor_id: payload
            .get("actor_id")
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| metadata.and_then(|m| metadata_str_field(m, "actor_id"))),
        role: payload
            .get("role")
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| metadata.and_then(|m| metadata_str_field(m, "role"))),
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
            None,
        );
        assert_eq!(p["text"], "hello");
        assert_eq!(p["user_id"], "u1");
        assert_eq!(p["access_count"], 0);
        assert!(p.get("metadata").is_none());
        assert!(p.get("memory_type").is_none());
    }

    #[test]
    fn build_memory_payload_with_memory_type() {
        let p = build_memory_payload("hi", None, None, None, None, "now", Some("preference"));
        assert_eq!(p["memory_type"], "preference");
    }

    #[test]
    fn build_memory_payload_with_metadata() {
        let meta = serde_json::json!({"key": "value"});
        let p = build_memory_payload("hi", None, None, None, Some(&meta), "now", None);
        assert_eq!(p["metadata"]["key"], "value");
    }

    #[test]
    fn raw_payload_includes_role() {
        let p = build_raw_memory_payload("hi", "user", None, None, None, None, "now");
        assert_eq!(p["role"], "user");
    }

    #[test]
    fn update_payload_has_no_created_at() {
        let p = build_update_payload("updated", None, None, None, None, None, "now", 3, None);
        assert!(p.get("created_at").is_none());
        assert_eq!(p["access_count"], 3);
    }

    #[test]
    fn update_payload_preserves_created_at_when_provided() {
        let p = build_update_payload(
            "updated",
            None,
            None,
            None,
            None,
            Some("2025-01-01T00:00:00Z"),
            "now",
            3,
            None,
        );
        assert_eq!(p["created_at"], "2025-01-01T00:00:00Z");
    }

    #[test]
    fn payload_round_trip() {
        let now = "2025-06-15T10:30:00Z";
        let original = build_memory_payload(
            "test text",
            Some("u1"),
            Some("a1"),
            None,
            None,
            now,
            Some("procedural"),
        );
        let id = mem7_core::new_memory_id();
        let item = payload_to_memory_item(id, &original, Some(0.95));

        assert_eq!(item.id, id);
        assert_eq!(item.text, "test text");
        assert_eq!(item.user_id.as_deref(), Some("u1"));
        assert_eq!(item.agent_id.as_deref(), Some("a1"));
        assert!(item.run_id.is_none());
        assert!(item.actor_id.is_none());
        assert!(item.role.is_none());
        assert_eq!(item.score, Some(0.95));
        assert_eq!(item.created_at, now);
        assert_eq!(item.access_count, 0);
        assert_eq!(item.memory_type.as_deref(), Some("procedural"));
    }

    #[test]
    fn payload_to_memory_item_handles_empty() {
        let payload = serde_json::json!({});
        let item = payload_to_memory_item(mem7_core::new_memory_id(), &payload, None);
        assert_eq!(item.text, "");
        assert!(item.user_id.is_none());
        assert!(item.score.is_none());
        assert_eq!(item.access_count, 0);
        assert!(item.memory_type.is_none());
        assert!(item.actor_id.is_none());
        assert!(item.role.is_none());
    }

    #[test]
    fn payload_to_memory_item_promotes_metadata_fields() {
        let payload = serde_json::json!({
            "text": "remember me",
            "metadata": {
                "user_id": "u1",
                "agent_id": "a1",
                "run_id": "r1",
                "actor_id": "actor-1",
                "role": "assistant",
                "topic": "ops"
            }
        });

        let item = payload_to_memory_item(mem7_core::new_memory_id(), &payload, None);
        assert_eq!(item.user_id.as_deref(), Some("u1"));
        assert_eq!(item.agent_id.as_deref(), Some("a1"));
        assert_eq!(item.run_id.as_deref(), Some("r1"));
        assert_eq!(item.actor_id.as_deref(), Some("actor-1"));
        assert_eq!(item.role.as_deref(), Some("assistant"));
        assert_eq!(item.metadata["topic"], "ops");
        assert!(item.metadata.get("actor_id").is_none());
    }

    #[test]
    fn payload_to_event_metadata_promotes_nested_fields() {
        let payload = serde_json::json!({
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-02T00:00:00Z",
            "role": "assistant",
            "metadata": {
                "actor_id": "actor-1"
            }
        });

        let event = payload_to_event_metadata(&payload);
        assert_eq!(event.created_at.as_deref(), Some("2026-01-01T00:00:00Z"));
        assert_eq!(event.updated_at.as_deref(), Some("2026-01-02T00:00:00Z"));
        assert_eq!(event.actor_id.as_deref(), Some("actor-1"));
        assert_eq!(event.role.as_deref(), Some("assistant"));
    }
}
