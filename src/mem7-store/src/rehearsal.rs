use std::sync::Arc;

use mem7_core::MemoryFilter;
use mem7_datetime::now_iso;
use mem7_graph::GraphStore;
use mem7_vector::VectorIndex;
use uuid::Uuid;

/// Spawn a fire-and-forget task to update `last_accessed_at` and `access_count`
/// for retrieved memories and relations, strengthening them against decay.
pub fn spawn_rehearsal(
    vi: Arc<dyn VectorIndex>,
    graph: Option<Arc<dyn GraphStore>>,
    mem_ids: Vec<Uuid>,
    rel_triples: Vec<(String, String, String)>,
    filter: MemoryFilter,
) {
    tokio::spawn(async move {
        let now = now_iso();
        for mid in mem_ids {
            if let Ok(Some((_, payload))) = vi.get(&mid).await {
                let ac = payload
                    .get("access_count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let mut p = payload;
                p["last_accessed_at"] = serde_json::Value::String(now.clone());
                p["access_count"] = serde_json::json!(ac + 1);
                if let Err(e) = vi.update(&mid, None, Some(p)).await {
                    tracing::warn!(memory_id = %mid, "rehearsal update failed: {e}");
                }
            }
        }
        if let Some(g) = graph
            && !rel_triples.is_empty()
            && let Err(e) = g.rehearse_relations(&rel_triples, &filter, &now).await
        {
            tracing::warn!("graph rehearsal failed: {e}");
        }
    });
}
