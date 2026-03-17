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

    let today = mem7_datetime::today_date();

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

    let output: FactExtractionOutput =
        mem7_core::parse_json_response(&response.content).map_err(Mem7Error::Serialization)?;

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
