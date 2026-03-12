use uuid::Uuid;

/// Generate a new UUIDv7 memory ID (time-sortable, RFC 9562).
pub fn new_memory_id() -> Uuid {
    Uuid::now_v7()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuidv7_is_time_sorted() {
        let id1 = new_memory_id();
        let id2 = new_memory_id();
        assert!(id2 > id1, "UUIDv7 should be monotonically increasing");
    }

    #[test]
    fn uuidv7_has_timestamp() {
        let id = new_memory_id();
        let ts = id.get_timestamp();
        assert!(ts.is_some(), "UUIDv7 should contain a timestamp");
    }
}
