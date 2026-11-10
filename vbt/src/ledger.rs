//! The JSONL ledger and canonical output.

use std::io::Write;

use serde_json::{Map, Value, json};
use time::macros::format_description;

use crate::tracker::{Event, ScoreMap, kind};

/// The ledger's `at` value: RFC 3339 UTC without fractional seconds.
pub fn ledger_at() -> String {
    const FORMAT: &[time::format_description::FormatItem<'_>] =
        format_description!("[year]-[month]-[day]T[hour]:[minute]:[second]Z");
    time::OffsetDateTime::now_utc()
        .format(FORMAT)
        .unwrap_or_default()
}

/// The shape every ledger line shares, so one jq query works across stream lifecycle
/// and tracked-event lines.
pub fn ledger_base(kind: &str, second: i64) -> Map<String, Value> {
    let mut record = Map::new();
    record.insert("at".into(), Value::String(ledger_at()));
    record.insert("second".into(), json!(second));
    record.insert("kind".into(), Value::String(kind.to_string()));
    record.insert("map".into(), Value::String(String::new()));
    record.insert("map_number".into(), json!(0));
    record.insert("match_id".into(), Value::String(String::new()));
    record.insert("score".into(), Value::Null);
    record.insert("winner".into(), Value::String(String::new()));
    record.insert("source".into(), Value::String(String::new()));
    record.insert("held".into(), json!(false));
    record
}

/// Turns one event into its ledger record.
pub fn event_record(event: &Event) -> Map<String, Value> {
    let mut record = ledger_base(&event.kind, event.second);
    record.insert("map".into(), Value::String(event.map.clone()));
    record.insert("map_number".into(), json!(event.map_number));
    record.insert("match_id".into(), Value::String(event.match_id.clone()));
    record.insert("score".into(), json!(event.score));
    record.insert("winner".into(), Value::String(event.winner.clone()));
    record.insert("source".into(), Value::String(event.source.clone()));
    record.insert("held".into(), json!(event.held));
    if let Some(exact) = event.exact {
        record.insert("exact".into(), json!(exact));
    }
    if let Some(read_score) = &event.read_score
        && !read_score.is_empty()
    {
        record.insert("read_score".into(), json!(read_score));
    }
    if let Some(previous_score) = &event.previous_score {
        record.insert("previous_score".into(), json!(previous_score));
    }
    if !event.pause.is_empty() {
        record.insert("pause".into(), Value::String(event.pause.clone()));
        record.insert("pause_since".into(), json!(event.pause_since));
        if !event.reason.is_empty() {
            record.insert("reason".into(), Value::String(event.reason.clone()));
        }
    }
    if !event.riot_state.is_empty() || !event.riot_blue.is_empty() {
        record.insert("riot_state".into(), Value::String(event.riot_state.clone()));
        record.insert("riot_blue".into(), Value::String(event.riot_blue.clone()));
        record.insert("riot_red".into(), Value::String(event.riot_red.clone()));
    }
    if event.kind == kind::FRAME_OBSERVATION
        || event.left_score.is_some()
        || !event.visual_left.is_empty()
    {
        record.insert(
            "visual_left".into(),
            Value::String(event.visual_left.clone()),
        );
        record.insert(
            "visual_right".into(),
            Value::String(event.visual_right.clone()),
        );
        record.insert(
            "left_score".into(),
            event.left_score.map(|v| json!(v)).unwrap_or(Value::Null),
        );
        record.insert(
            "right_score".into(),
            event.right_score.map(|v| json!(v)).unwrap_or(Value::Null),
        );
        record.insert("video_score".into(), json!(event.video_score));
    }
    if event.kind == kind::FRAME_OBSERVATION {
        record.insert("plate_ok".into(), json!(event.plate_ok));
        record.insert("scene".into(), Value::String(event.scene.clone()));
        record.insert("left_confidence".into(), json!(event.left_confidence));
        record.insert("right_confidence".into(), json!(event.right_confidence));
    }
    if let Some(scoreboard) = &event.scoreboard {
        record.insert(
            "scoreboard".into(),
            serde_json::to_value(scoreboard).unwrap_or(Value::Null),
        );
    }
    record
}

/// Turns a stream lifecycle line into its ledger record.
pub fn lifecycle_record(kind: &str, url: &str, second: i64) -> Map<String, Value> {
    let mut record = ledger_base(kind, second);
    record.insert("stream".into(), Value::String(url.to_string()));
    record
}

/// Appends one record as a JSON line, written unbuffered so a lifecycle event
/// reaches disk before the process can stall or reconnect.
pub fn append(file: &mut std::fs::File, record: &Map<String, Value>, path: &str) {
    if let Err(error) = write_line(file, record) {
        tracing::warn!(path, kind = ?record.get("kind"), error = %error, "ledger write failed");
    }
}

/// Writes one record as a JSON line.
pub fn write_line(writer: &mut impl Write, record: &Map<String, Value>) -> std::io::Result<()> {
    let mut line = serde_json::to_vec(record)?;
    line.push(b'\n');
    writer.write_all(&line)
}

/// Writes one stable line per event, for diffing two runs.
pub fn print_canonical(writer: &mut impl Write, event: &Event) {
    let mut suffix = String::new();
    if event.held {
        suffix.push_str(" held=true");
    }
    if let Some(read) = &event.read_score
        && !read.is_empty()
    {
        suffix.push_str(" read=");
        suffix.push_str(&format_canonical_score(Some(read)));
    }
    if !event.pause.is_empty() {
        suffix.push_str(" pause=");
        suffix.push_str(&event.pause);
        if !event.reason.is_empty() {
            suffix.push_str(" reason=");
            suffix.push_str(&event.reason);
        }
    }
    let _ = writeln!(
        writer,
        "sec={} kind={} map={} score={} winner={} source={}{}",
        event.second,
        event.kind,
        event.map,
        format_canonical_score(event.score.as_ref()),
        event.winner,
        event.source,
        suffix
    );
}

/// Renders a score map as `A=1,B=2`, sorted by code. An empty or absent score is `""`.
pub fn format_canonical_score(score: Option<&ScoreMap>) -> String {
    let Some(score) = score else {
        return String::new();
    };
    let parts: Vec<String> = score
        .iter()
        .map(|(code, value)| format!("{code}={value}"))
        .collect();
    parts.join(",")
}

/// The events-file ledger: one append-only, unbuffered file.
pub struct Ledger {
    file: std::fs::File,
    path: String,
}

impl Ledger {
    /// Opens the ledger with append/create semantics and mode 0600.
    pub fn open(path: &str) -> Result<Ledger, std::io::Error> {
        use std::os::unix::fs::OpenOptionsExt;
        let file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .mode(0o600)
            .open(path)?;
        Ok(Ledger {
            file,
            path: path.to_string(),
        })
    }

    /// Appends one event record.
    pub fn record(&mut self, event: &Event) {
        let record = event_record(event);
        append(&mut self.file, &record, &self.path);
    }

    /// Appends one stream lifecycle record. Lifecycle lines never reach stdout.
    pub fn lifecycle(&mut self, kind: &str, url: &str, second: i64) {
        let record = lifecycle_record(kind, url, second);
        append(&mut self.file, &record, &self.path);
    }
}
