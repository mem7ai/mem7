use std::time::{SystemTime, UNIX_EPOCH};

/// Current UTC time as ISO 8601 string (`YYYY-MM-DDThh:mm:ssZ`).
pub fn now_iso() -> String {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

/// Current UTC date as `YYYY-MM-DD`.
pub fn today_date() -> String {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let days = d.as_secs() / 86400;
    let (y, m, d) = days_to_ymd(days);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Parse ISO 8601 `YYYY-MM-DDThh:mm:ssZ` into epoch seconds.
pub fn iso_to_epoch(ts: &str) -> Option<f64> {
    let ts = ts.trim().trim_end_matches('Z');
    let (date, time) = ts.split_once('T')?;
    let mut date_parts = date.split('-');
    let y: u64 = date_parts.next()?.parse().ok()?;
    let m: u64 = date_parts.next()?.parse().ok()?;
    let d: u64 = date_parts.next()?.parse().ok()?;

    let mut time_parts = time.split(':');
    let h: u64 = time_parts.next()?.parse().ok()?;
    let min: u64 = time_parts.next()?.parse().ok()?;
    let sec: u64 = time_parts.next()?.parse().ok()?;

    let days = ymd_to_days(y, m, d);
    Some((days * 86400 + h * 3600 + min * 60 + sec) as f64)
}

/// Convert days since Unix epoch → (year, month, day) using the civil calendar algorithm.
fn days_to_ymd(days_since_epoch: u64) -> (u64, u64, u64) {
    let z = days_since_epoch + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Convert (year, month, day) → days since Unix epoch.
fn ymd_to_days(y: u64, m: u64, d: u64) -> u64 {
    let y = if m <= 2 { y.wrapping_sub(1) } else { y };
    let era = y / 400;
    let yoe = y - era * 400;
    let m_adj = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * m_adj + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_iso_is_well_formed() {
        let s = now_iso();
        assert!(s.ends_with('Z'));
        assert!(s.contains('T'));
        assert_eq!(s.len(), 20);
    }

    #[test]
    fn today_date_is_well_formed() {
        let s = today_date();
        assert_eq!(s.len(), 10);
        assert_eq!(&s[4..5], "-");
        assert_eq!(&s[7..8], "-");
    }

    #[test]
    fn iso_round_trip() {
        let epoch = iso_to_epoch("2025-01-01T00:00:00Z");
        assert!(epoch.is_some());
        assert!((epoch.unwrap() - 1735689600.0).abs() < 1.0);
    }

    #[test]
    fn iso_to_epoch_rejects_invalid() {
        assert!(iso_to_epoch("not-a-date").is_none());
        assert!(iso_to_epoch("").is_none());
    }
}
