mod sqlite;

use async_trait::async_trait;
use mem7_core::{MemoryAction, MemoryEvent};
use mem7_error::Result;
use uuid::Uuid;

pub use sqlite::SqliteHistory;

/// Abstract audit-trail store for memory operations.
///
/// Implement this trait to plug in a different backend (e.g. Postgres, S3 log).
#[async_trait]
pub trait HistoryStore: Send + Sync {
    async fn add_event(
        &self,
        memory_id: Uuid,
        old_value: Option<&str>,
        new_value: Option<&str>,
        action: MemoryAction,
    ) -> Result<MemoryEvent>;

    async fn get_history(&self, memory_id: Uuid) -> Result<Vec<MemoryEvent>>;

    async fn reset(&self) -> Result<()>;
}
