use async_trait::async_trait;
use mem7_core::{MemoryAction, MemoryEvent, MemoryEventMetadata, new_memory_id};
use mem7_datetime::now_iso;
use mem7_error::Result;
use rusqlite::params;
use tokio_rusqlite::Connection;
use uuid::Uuid;

use crate::HistoryStore;

/// Async SQLite-backed audit trail for memory operations.
pub struct SqliteHistory {
    conn: Connection,
}

impl SqliteHistory {
    pub async fn new(path: &str) -> Result<Self> {
        let conn = if path == ":memory:" {
            Connection::open_in_memory().await?
        } else {
            Connection::open(path).await?
        };

        conn.call(|conn| {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS memory_history (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    action TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    updated_at TEXT,
                    is_deleted INTEGER NOT NULL DEFAULT 0,
                    actor_id TEXT,
                    role TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_history_memory_id ON memory_history(memory_id);",
            )?;
            ensure_history_column(
                conn,
                "updated_at",
                "ALTER TABLE memory_history ADD COLUMN updated_at TEXT",
            )?;
            ensure_history_column(
                conn,
                "is_deleted",
                "ALTER TABLE memory_history ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0",
            )?;
            ensure_history_column(
                conn,
                "actor_id",
                "ALTER TABLE memory_history ADD COLUMN actor_id TEXT",
            )?;
            ensure_history_column(
                conn,
                "role",
                "ALTER TABLE memory_history ADD COLUMN role TEXT",
            )?;
            Ok(())
        })
        .await?;

        Ok(Self { conn })
    }
}

#[async_trait]
impl HistoryStore for SqliteHistory {
    async fn add_event(
        &self,
        memory_id: Uuid,
        old_value: Option<&str>,
        new_value: Option<&str>,
        action: MemoryAction,
        metadata: MemoryEventMetadata,
    ) -> Result<MemoryEvent> {
        let event_id = new_memory_id();
        let action_str = action.to_string();
        let old = old_value.map(String::from);
        let new = new_value.map(String::from);
        let created_at = metadata.created_at.unwrap_or_else(now_iso);
        let updated_at = metadata.updated_at;
        let is_deleted = if metadata.is_deleted { 1 } else { 0 };
        let actor_id = metadata.actor_id;
        let role = metadata.role;
        let created_at_for_insert = created_at.clone();
        let updated_at_for_insert = updated_at.clone();
        let actor_id_for_insert = actor_id.clone();
        let role_for_insert = role.clone();

        self.conn
            .call(move |conn| {
                conn.execute(
                    "INSERT INTO memory_history (
                        id, memory_id, old_value, new_value, action, created_at, updated_at, is_deleted, actor_id, role
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                    params![
                        event_id.to_string(),
                        memory_id.to_string(),
                        old,
                        new,
                        action_str,
                        created_at_for_insert,
                        updated_at_for_insert,
                        is_deleted,
                        actor_id_for_insert,
                        role_for_insert,
                    ],
                )?;
                Ok(())
            })
            .await?;

        Ok(MemoryEvent {
            id: event_id,
            memory_id,
            old_value: old_value.map(String::from),
            new_value: new_value.map(String::from),
            action,
            created_at,
            updated_at,
            is_deleted: is_deleted != 0,
            actor_id,
            role,
        })
    }

    async fn get_history(&self, memory_id: Uuid) -> Result<Vec<MemoryEvent>> {
        let events = self
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT id, memory_id, old_value, new_value, action, created_at, updated_at, is_deleted, actor_id, role
                     FROM memory_history
                     WHERE memory_id = ?1
                     ORDER BY created_at ASC, COALESCE(updated_at, created_at) ASC",
                )?;
                let rows = stmt.query_map(params![memory_id.to_string()], |row| {
                    let id_str: String = row.get(0)?;
                    let mid_str: String = row.get(1)?;
                    let old_val: Option<String> = row.get(2)?;
                    let new_val: Option<String> = row.get(3)?;
                    let action_str: String = row.get(4)?;
                    let created_at: String = row.get(5)?;
                    let updated_at: Option<String> = row.get(6)?;
                    let is_deleted: i64 = row.get(7)?;
                    let actor_id: Option<String> = row.get(8)?;
                    let role: Option<String> = row.get(9)?;
                    Ok((
                        id_str,
                        mid_str,
                        old_val,
                        new_val,
                        action_str,
                        created_at,
                        updated_at,
                        is_deleted,
                        actor_id,
                        role,
                    ))
                })?;
                let mut events = Vec::new();
                for row in rows {
                    let (
                        id_str,
                        mid_str,
                        old_val,
                        new_val,
                        action_str,
                        created_at,
                        updated_at,
                        is_deleted,
                        actor_id,
                        role,
                    ) = row?;
                    events.push(MemoryEvent {
                        id: Uuid::parse_str(&id_str).unwrap_or_default(),
                        memory_id: Uuid::parse_str(&mid_str).unwrap_or_default(),
                        old_value: old_val,
                        new_value: new_val,
                        action: match action_str.as_str() {
                            "ADD" => MemoryAction::Add,
                            "UPDATE" => MemoryAction::Update,
                            "DELETE" => MemoryAction::Delete,
                            _ => MemoryAction::None,
                        },
                        created_at,
                        updated_at,
                        is_deleted: is_deleted != 0,
                        actor_id,
                        role,
                    });
                }
                Ok(events)
            })
            .await?;
        Ok(events)
    }

    async fn reset(&self) -> Result<()> {
        self.conn
            .call(|conn| {
                conn.execute("DELETE FROM memory_history", [])?;
                Ok(())
            })
            .await?;
        Ok(())
    }
}

fn ensure_history_column(
    conn: &rusqlite::Connection,
    column_name: &str,
    alter_sql: &str,
) -> rusqlite::Result<()> {
    let mut stmt = conn.prepare("PRAGMA table_info(memory_history)")?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let existing: String = row.get(1)?;
        if existing == column_name {
            return Ok(());
        }
    }

    conn.execute(alter_sql, [])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn in_memory_history() -> SqliteHistory {
        SqliteHistory::new(":memory:").await.unwrap()
    }

    #[tokio::test]
    async fn add_and_retrieve_event() {
        let h = in_memory_history().await;
        let mid = mem7_core::new_memory_id();

        let event = h
            .add_event(
                mid,
                None,
                Some("hello world"),
                MemoryAction::Add,
                MemoryEventMetadata::default(),
            )
            .await
            .unwrap();

        assert_eq!(event.memory_id, mid);
        assert_eq!(event.action, MemoryAction::Add);
        assert_eq!(event.new_value.as_deref(), Some("hello world"));
        assert!(event.old_value.is_none());
        assert!(!event.created_at.is_empty());
        assert!(event.updated_at.is_none());
        assert!(!event.is_deleted);
    }

    #[tokio::test]
    async fn get_history_returns_ordered() {
        let h = in_memory_history().await;
        let mid = mem7_core::new_memory_id();

        h.add_event(
            mid,
            None,
            Some("v1"),
            MemoryAction::Add,
            MemoryEventMetadata::default(),
        )
        .await
        .unwrap();
        h.add_event(
            mid,
            Some("v1"),
            Some("v2"),
            MemoryAction::Update,
            MemoryEventMetadata::default(),
        )
        .await
        .unwrap();
        h.add_event(
            mid,
            Some("v2"),
            None,
            MemoryAction::Delete,
            MemoryEventMetadata {
                is_deleted: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let events = h.get_history(mid).await.unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].action, MemoryAction::Add);
        assert_eq!(events[1].action, MemoryAction::Update);
        assert_eq!(events[2].action, MemoryAction::Delete);
    }

    #[tokio::test]
    async fn get_history_empty_for_unknown_id() {
        let h = in_memory_history().await;
        let events = h.get_history(mem7_core::new_memory_id()).await.unwrap();
        assert!(events.is_empty());
    }

    #[tokio::test]
    async fn reset_clears_all() {
        let h = in_memory_history().await;
        let mid = mem7_core::new_memory_id();
        h.add_event(
            mid,
            None,
            Some("data"),
            MemoryAction::Add,
            MemoryEventMetadata::default(),
        )
        .await
        .unwrap();

        h.reset().await.unwrap();

        let events = h.get_history(mid).await.unwrap();
        assert!(events.is_empty());
    }

    #[tokio::test]
    async fn stores_extended_audit_fields() {
        let h = in_memory_history().await;
        let mid = mem7_core::new_memory_id();
        let event = h
            .add_event(
                mid,
                Some("old"),
                Some("new"),
                MemoryAction::Update,
                MemoryEventMetadata {
                    created_at: Some("2026-01-01T00:00:00Z".into()),
                    updated_at: Some("2026-01-02T00:00:00Z".into()),
                    is_deleted: false,
                    actor_id: Some("actor-1".into()),
                    role: Some("assistant".into()),
                },
            )
            .await
            .unwrap();

        assert_eq!(event.created_at, "2026-01-01T00:00:00Z");

        let history = h.get_history(mid).await.unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(
            history[0].updated_at.as_deref(),
            Some("2026-01-02T00:00:00Z")
        );
        assert_eq!(history[0].actor_id.as_deref(), Some("actor-1"));
        assert_eq!(history[0].role.as_deref(), Some("assistant"));
    }
}
