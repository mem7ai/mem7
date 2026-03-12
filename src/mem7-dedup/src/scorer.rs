use std::collections::HashMap;

use mem7_error::{Mem7Error, Result};
use mem7_core::MemoryAction;
use serde::Deserialize;
use uuid::Uuid;

/// A single memory decision from the LLM's response.
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryDecision {
    pub id: String,
    pub text: String,
    pub event: MemoryAction,
    pub old_memory: Option<String>,
}

/// Parsed LLM memory-update response.
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryUpdateResponse {
    pub memory: Vec<MemoryDecision>,
}

/// Maps integer IDs (used by LLM) back to real UUIDs.
/// The LLM receives integer IDs to avoid hallucinating UUIDs.
pub struct IdMapping {
    int_to_uuid: HashMap<String, Uuid>,
}

impl IdMapping {
    pub fn new() -> Self {
        Self {
            int_to_uuid: HashMap::new(),
        }
    }

    pub fn add(&mut self, int_id: String, uuid: Uuid) {
        self.int_to_uuid.insert(int_id, uuid);
    }

    pub fn resolve(&self, int_id: &str) -> Option<Uuid> {
        self.int_to_uuid.get(int_id).copied()
    }
}

impl Default for IdMapping {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the "existing memory" dict that gets sent to the LLM for comparison.
/// Maps each existing memory to an integer ID for stability.
pub fn build_existing_memory_dict(
    existing: &[(Uuid, String)],
) -> (serde_json::Value, IdMapping) {
    let mut mapping = IdMapping::new();
    let mut entries = Vec::new();

    for (idx, (uuid, text)) in existing.iter().enumerate() {
        let int_id = idx.to_string();
        mapping.add(int_id.clone(), *uuid);
        entries.push(serde_json::json!({
            "id": int_id,
            "text": text,
        }));
    }

    (serde_json::Value::Array(entries), mapping)
}

/// Parse the LLM's JSON response for memory update decisions.
pub fn parse_memory_update_response(json_str: &str) -> Result<MemoryUpdateResponse> {
    // Try parsing directly
    if let Ok(resp) = serde_json::from_str::<MemoryUpdateResponse>(json_str) {
        return Ok(resp);
    }

    // Try extracting JSON from markdown code blocks
    let trimmed = json_str.trim();
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

    serde_json::from_str(cleaned).map_err(|e| {
        Mem7Error::Serialization(format!("Failed to parse memory update response: {e}\nRaw: {json_str}"))
    })
}

/// Deduplicate a list of existing memories retrieved for multiple facts.
/// Returns a deduplicated list of (uuid, text) pairs.
pub fn deduplicate_memories(memories: Vec<(Uuid, String, f32)>) -> Vec<(Uuid, String)> {
    let mut seen = HashMap::new();
    for (uuid, text, score) in memories {
        seen.entry(uuid)
            .and_modify(|(existing_text, existing_score): &mut (String, f32)| {
                if score > *existing_score {
                    *existing_text = text.clone();
                    *existing_score = score;
                }
            })
            .or_insert((text, score));
    }
    seen.into_iter()
        .map(|(uuid, (text, _))| (uuid, text))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_response() {
        let json = r#"{"memory": [{"id": "0", "text": "Loves pizza", "event": "ADD"}]}"#;
        let resp = parse_memory_update_response(json).unwrap();
        assert_eq!(resp.memory.len(), 1);
        assert_eq!(resp.memory[0].text, "Loves pizza");
        assert_eq!(resp.memory[0].event, MemoryAction::Add);
    }

    #[test]
    fn parse_code_block_response() {
        let json = "```json\n{\"memory\": [{\"id\": \"0\", \"text\": \"test\", \"event\": \"NONE\"}]}\n```";
        let resp = parse_memory_update_response(json).unwrap();
        assert_eq!(resp.memory.len(), 1);
    }

    #[test]
    fn dedup_keeps_highest_score() {
        let id = Uuid::now_v7();
        let memories = vec![
            (id, "low score".into(), 0.5),
            (id, "high score".into(), 0.9),
        ];
        let result = deduplicate_memories(memories);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, "high score");
    }
}
