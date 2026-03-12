use mem7_error::Result;
use mem7_core::{MemoryAction, MemoryEvent, new_memory_id};
use rusqlite::params;
use tokio_rusqlite::Connection;
use uuid::Uuid;

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
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
                );
                CREATE INDEX IF NOT EXISTS idx_history_memory_id ON memory_history(memory_id);",
            )?;
            Ok(())
        })
        .await?;

        Ok(Self { conn })
    }

    pub async fn add_event(
        &self,
        memory_id: Uuid,
        old_value: Option<&str>,
        new_value: Option<&str>,
        action: MemoryAction,
    ) -> Result<MemoryEvent> {
        let event_id = new_memory_id();
        let action_str = action.to_string();
        let old = old_value.map(String::from);
        let new = new_value.map(String::from);

        let created_at = self
            .conn
            .call(move |conn| {
                conn.execute(
                    "INSERT INTO memory_history (id, memory_id, old_value, new_value, action)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        event_id.to_string(),
                        memory_id.to_string(),
                        old,
                        new,
                        action_str,
                    ],
                )?;
                let created_at: String = conn.query_row(
                    "SELECT created_at FROM memory_history WHERE id = ?1",
                    params![event_id.to_string()],
                    |row| row.get(0),
                )?;
                Ok(created_at)
            })
            .await?;

        Ok(MemoryEvent {
            id: event_id,
            memory_id,
            old_value: old_value.map(String::from),
            new_value: new_value.map(String::from),
            action,
            created_at,
        })
    }

    pub async fn get_history(&self, memory_id: Uuid) -> Result<Vec<MemoryEvent>> {
        let events = self
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT id, memory_id, old_value, new_value, action, created_at
                     FROM memory_history
                     WHERE memory_id = ?1
                     ORDER BY created_at ASC",
                )?;
                let rows = stmt.query_map(params![memory_id.to_string()], |row| {
                    let id_str: String = row.get(0)?;
                    let mid_str: String = row.get(1)?;
                    let old_val: Option<String> = row.get(2)?;
                    let new_val: Option<String> = row.get(3)?;
                    let action_str: String = row.get(4)?;
                    let created_at: String = row.get(5)?;
                    Ok((id_str, mid_str, old_val, new_val, action_str, created_at))
                })?;
                let mut events = Vec::new();
                for row in rows {
                    let (id_str, mid_str, old_val, new_val, action_str, created_at) = row?;
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
                    });
                }
                Ok(events)
            })
            .await?;
        Ok(events)
    }

    pub async fn reset(&self) -> Result<()> {
        self.conn
            .call(|conn| {
                conn.execute("DELETE FROM memory_history", [])?;
                Ok(())
            })
            .await?;
        Ok(())
    }
}
