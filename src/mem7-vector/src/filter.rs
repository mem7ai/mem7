use mem7_core::MemoryFilter;

/// Check if a payload matches the given filter criteria.
pub fn matches_filter(payload: &serde_json::Value, filter: &MemoryFilter) -> bool {
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
    true
}
