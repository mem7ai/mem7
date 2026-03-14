use mem7_core::{ChatMessage, Fact};
use mem7_dedup::{
    MemoryUpdateResponse, build_existing_memory_dict, deduplicate_memories,
    parse_memory_update_response,
};
use mem7_error::{Mem7Error, Result};
use mem7_llm::{LlmClient, LlmMessage, ResponseFormat};
use serde::Deserialize;
use tracing::debug;

use crate::prompts::{AGENT_FACT_EXTRACTION_PROMPT, USER_FACT_EXTRACTION_PROMPT};

#[derive(Debug, Deserialize)]
struct FactExtractionOutput {
    facts: Vec<String>,
}

/// Determine whether to use agent memory extraction.
/// Returns `true` when `agent_id` is set AND messages contain an `assistant` role.
fn should_use_agent_extraction(agent_id: Option<&str>, messages: &[ChatMessage]) -> bool {
    agent_id.is_some() && messages.iter().any(|m| m.role == "assistant")
}

/// Extract facts from a conversation using the LLM.
///
/// When no `custom_prompt` is provided, the prompt is selected automatically:
/// - If `agent_id` is set and the conversation contains assistant messages,
///   uses `AGENT_FACT_EXTRACTION_PROMPT` (extracts facts from assistant only).
/// - Otherwise uses `USER_FACT_EXTRACTION_PROMPT` (extracts facts from user only).
pub async fn extract_facts(
    llm: &dyn LlmClient,
    messages: &[ChatMessage],
    agent_id: Option<&str>,
    custom_prompt: Option<&str>,
) -> Result<Vec<Fact>> {
    let prompt = custom_prompt.unwrap_or_else(|| {
        if should_use_agent_extraction(agent_id, messages) {
            AGENT_FACT_EXTRACTION_PROMPT
        } else {
            USER_FACT_EXTRACTION_PROMPT
        }
    });

    let today = today_date();

    let conversation = messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let system_prompt = format!("{prompt}\nToday's date is {today}.");

    let llm_messages = vec![
        LlmMessage::system(system_prompt),
        LlmMessage::user(format!("Input:\n{conversation}")),
    ];

    let response = llm
        .chat_completion(&llm_messages, Some(&ResponseFormat::json()))
        .await?;

    debug!(raw_response = %response.content, "fact extraction response");

    let output: FactExtractionOutput = parse_json_response(&response.content)?;

    Ok(output.facts.into_iter().map(|text| Fact { text }).collect())
}

fn today_date() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let days = d.as_secs() / 86400;
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

/// Ask the LLM to decide how to update memory given new facts and existing memories.
pub async fn decide_memory_updates(
    llm: &dyn LlmClient,
    new_facts: &[Fact],
    existing_memories: Vec<(uuid::Uuid, String, f32)>,
    custom_prompt: Option<&str>,
) -> Result<(MemoryUpdateResponse, mem7_dedup::IdMapping)> {
    let deduped = deduplicate_memories(existing_memories);
    let (existing_json, id_mapping) = build_existing_memory_dict(&deduped);
    let existing_str = serde_json::to_string_pretty(&existing_json)?;

    let facts_list: Vec<String> = new_facts.iter().map(|f| f.text.clone()).collect();
    let facts_str = serde_json::to_string(&facts_list)?;

    let prompt =
        crate::prompts::build_update_memory_prompt(custom_prompt, &existing_str, &facts_str);

    let llm_messages = vec![LlmMessage::user(prompt)];

    let response = llm
        .chat_completion(&llm_messages, Some(&ResponseFormat::json()))
        .await?;

    debug!(raw_response = %response.content, "memory update response");

    let update_resp = parse_memory_update_response(&response.content)?;

    Ok((update_resp, id_mapping))
}

fn parse_json_response<T: serde::de::DeserializeOwned>(raw: &str) -> Result<T> {
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

    serde_json::from_str(cleaned)
        .map_err(|e| Mem7Error::Serialization(format!("JSON parse error: {e}\nRaw: {raw}")))
}
