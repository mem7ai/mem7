use mem7_core::{ChatMessage, Fact};
use mem7_dedup::{
    MemoryUpdateResponse, build_existing_memory_dict, deduplicate_memories,
    parse_memory_update_response,
};
use mem7_error::{Mem7Error, Result};
use mem7_llm::{LlmClient, LlmMessage, ResponseFormat};
use serde::Deserialize;
use tracing::debug;

use crate::prompts::DEFAULT_FACT_EXTRACTION_PROMPT;

#[derive(Debug, Deserialize)]
struct FactExtractionOutput {
    facts: Vec<String>,
}

/// Extract facts from a conversation using the LLM.
pub async fn extract_facts(
    llm: &dyn LlmClient,
    messages: &[ChatMessage],
    custom_prompt: Option<&str>,
) -> Result<Vec<Fact>> {
    let prompt = custom_prompt.unwrap_or(DEFAULT_FACT_EXTRACTION_PROMPT);

    let conversation = messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let llm_messages = vec![LlmMessage::system(prompt), LlmMessage::user(conversation)];

    let response = llm
        .chat_completion(&llm_messages, Some(&ResponseFormat::json()))
        .await?;

    debug!(raw_response = %response.content, "fact extraction response");

    let output: FactExtractionOutput = parse_json_response(&response.content)?;

    Ok(output.facts.into_iter().map(|text| Fact { text }).collect())
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
