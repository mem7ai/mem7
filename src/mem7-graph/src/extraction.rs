use mem7_error::{Mem7Error, Result};
use mem7_llm::{LlmClient, LlmMessage, ResponseFormat};
use serde::Deserialize;
use tracing::debug;

use crate::prompts::{
    DELETE_RELATIONS_PROMPT, ENTITY_EXTRACTION_PROMPT, RELATION_EXTRACTION_PROMPT,
};
use crate::types::{Entity, GraphSearchResult, Relation};

#[derive(Debug, Deserialize)]
struct EntityExtractionOutput {
    #[serde(default)]
    entities: Vec<RawEntity>,
}

#[derive(Debug, Deserialize)]
struct RawEntity {
    entity: String,
    entity_type: String,
}

#[derive(Debug, Deserialize)]
struct RelationExtractionOutput {
    #[serde(default)]
    relations: Vec<RawRelation>,
}

#[derive(Debug, Deserialize)]
struct RawRelation {
    source: String,
    relationship: String,
    destination: String,
}

/// Extract entities from a conversation using the LLM.
pub async fn extract_entities(
    llm: &dyn LlmClient,
    conversation: &str,
    custom_prompt: Option<&str>,
) -> Result<Vec<Entity>> {
    let prompt = custom_prompt.unwrap_or(ENTITY_EXTRACTION_PROMPT);

    let messages = vec![LlmMessage::system(prompt), LlmMessage::user(conversation)];

    let response = llm
        .chat_completion(&messages, Some(&ResponseFormat::json()))
        .await?;

    debug!(raw = %response.content, "entity extraction response");

    let output: EntityExtractionOutput = parse_json_response(&response.content)?;

    Ok(output
        .entities
        .into_iter()
        .map(|e| Entity {
            name: e.entity,
            entity_type: e.entity_type,
            embedding: None,
            created_at: None,
            mentions: 0,
        })
        .collect())
}

/// Extract relations between entities from a conversation using the LLM.
pub async fn extract_relations(
    llm: &dyn LlmClient,
    conversation: &str,
    entities: &[Entity],
    custom_prompt: Option<&str>,
) -> Result<Vec<Relation>> {
    if entities.is_empty() {
        return Ok(Vec::new());
    }

    let prompt = custom_prompt.unwrap_or(RELATION_EXTRACTION_PROMPT);

    let entity_names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
    let user_input = format!(
        "Entities: {}\nText: {}",
        entity_names.join(", "),
        conversation
    );

    let messages = vec![LlmMessage::system(prompt), LlmMessage::user(user_input)];

    let response = llm
        .chat_completion(&messages, Some(&ResponseFormat::json()))
        .await?;

    debug!(raw = %response.content, "relation extraction response");

    let output: RelationExtractionOutput = parse_json_response(&response.content)?;

    Ok(output
        .relations
        .into_iter()
        .map(|r| Relation {
            source: r.source,
            relationship: r.relationship,
            destination: r.destination,
            created_at: None,
            mentions: 0,
            valid: true,
        })
        .collect())
}

#[derive(Debug, Deserialize)]
struct DeletionOutput {
    #[serde(default)]
    deletions: Vec<RawRelation>,
}

/// Ask the LLM which existing relations should be invalidated given new data.
/// Returns triples `(source, relationship, destination)` to soft-delete.
pub async fn extract_deletions(
    llm: &dyn LlmClient,
    existing: &[GraphSearchResult],
    new_data: &str,
) -> Result<Vec<(String, String, String)>> {
    if existing.is_empty() {
        return Ok(Vec::new());
    }

    let existing_str = existing
        .iter()
        .map(|r| format!("{} -- {} -- {}", r.source, r.relationship, r.destination))
        .collect::<Vec<_>>()
        .join("\n");

    let user_msg =
        format!("Here are the existing memories:\n{existing_str}\n\nNew Information:\n{new_data}");

    let messages = vec![
        LlmMessage::system(DELETE_RELATIONS_PROMPT),
        LlmMessage::user(user_msg),
    ];

    let response = llm
        .chat_completion(&messages, Some(&ResponseFormat::json()))
        .await?;

    debug!(raw = %response.content, "deletion extraction response");

    let output: DeletionOutput = parse_json_response(&response.content)?;

    Ok(output
        .deletions
        .into_iter()
        .map(|d| (d.source, d.relationship, d.destination))
        .collect())
}

fn parse_json_response<T: serde::de::DeserializeOwned>(raw: &str) -> Result<T> {
    mem7_core::parse_json_response(raw).map_err(Mem7Error::Graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_entity_response() {
        let raw = r#"{"entities": [{"entity": "Alice", "entity_type": "Person"}, {"entity": "tennis", "entity_type": "Activity"}]}"#;
        let output: EntityExtractionOutput = parse_json_response(raw).unwrap();
        assert_eq!(output.entities.len(), 2);
        assert_eq!(output.entities[0].entity, "Alice");
        assert_eq!(output.entities[1].entity_type, "Activity");
    }

    #[test]
    fn parse_entity_response_with_code_fence() {
        let raw =
            "```json\n{\"entities\": [{\"entity\": \"Bob\", \"entity_type\": \"Person\"}]}\n```";
        let output: EntityExtractionOutput = parse_json_response(raw).unwrap();
        assert_eq!(output.entities.len(), 1);
        assert_eq!(output.entities[0].entity, "Bob");
    }

    #[test]
    fn parse_empty_entities() {
        let raw = r#"{"entities": []}"#;
        let output: EntityExtractionOutput = parse_json_response(raw).unwrap();
        assert!(output.entities.is_empty());
    }

    #[test]
    fn parse_relation_response() {
        let raw = r#"{"relations": [{"source": "USER", "relationship": "works_at", "destination": "Google"}]}"#;
        let output: RelationExtractionOutput = parse_json_response(raw).unwrap();
        assert_eq!(output.relations.len(), 1);
        assert_eq!(output.relations[0].source, "USER");
        assert_eq!(output.relations[0].relationship, "works_at");
        assert_eq!(output.relations[0].destination, "Google");
    }

    #[test]
    fn parse_empty_relations() {
        let raw = r#"{"relations": []}"#;
        let output: RelationExtractionOutput = parse_json_response(raw).unwrap();
        assert!(output.relations.is_empty());
    }
}
