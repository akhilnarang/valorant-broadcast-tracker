//! Ledger and canonical-output tests.

use serde_json::{Value, json};
use vbt::ledger;
use vbt::tracker::{Event, ScoreMap};

fn scores(entries: &[(&str, i32)]) -> Option<ScoreMap> {
    Some(
        entries
            .iter()
            .map(|(code, value)| ((*code).to_string(), *value))
            .collect(),
    )
}

fn event_value(event: &Event) -> Value {
    let record = ledger::event_record(event);
    let at = record
        .get("at")
        .and_then(|value| value.as_str())
        .expect("every ledger line carries at");
    time::OffsetDateTime::parse(at, &time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|e| panic!("at = {at:?}, want an RFC3339 wall clock: {e}"));
    let mut value = serde_json::to_value(record).unwrap();
    value.as_object_mut().unwrap().remove("at");
    value
}

#[test]
fn test_canonical_output_includes_held_and_rejected_reads() {
    let cases: [(Event, &str); 4] = [
        (
            Event {
                kind: "round".into(),
                second: 10,
                map: "Ascent".into(),
                score: scores(&[("LOUD", 12)]),
                read_score: scores(&[("LOUD", 13)]),
                winner: "LOUD".into(),
                held: true,
                ..Default::default()
            },
            "sec=10 kind=round map=Ascent score=LOUD=12 winner=LOUD source= held=true read=LOUD=13\n",
        ),
        (
            Event {
                kind: "reject".into(),
                second: 11,
                map: "Ascent".into(),
                score: scores(&[("LOUD", 12)]),
                read_score: scores(&[("LOUD", 7)]),
                ..Default::default()
            },
            "sec=11 kind=reject map=Ascent score=LOUD=12 winner= source= read=LOUD=7\n",
        ),
        (
            Event {
                kind: "map_start".into(),
                second: 5,
                map: "Ascent".into(),
                score: scores(&[("LOUD", 0), ("EDG", 0)]),
                ..Default::default()
            },
            "sec=5 kind=map_start map=Ascent score=EDG=0,LOUD=0 winner= source=\n",
        ),
        (
            Event {
                kind: "pause_start".into(),
                second: 12,
                pause: "tech_pause".into(),
                reason: "GEAR".into(),
                ..Default::default()
            },
            "sec=12 kind=pause_start map= score= winner= source= pause=tech_pause reason=GEAR\n",
        ),
    ];
    for (event, want) in cases {
        let mut output = Vec::new();
        ledger::print_canonical(&mut output, &event);
        assert_eq!(String::from_utf8(output).unwrap(), want);
    }
}

// The ledger is the record a live run leaves behind: every tracked transition must
// land in it as one JSON line carrying the run identity and the read's evidence.
#[test]
fn test_ledger_writes_every_event_kind_with_its_fields() {
    let cases: [(&str, Event, Value); 14] = [
        (
            "shadow frame evidence",
            Event {
                kind: "frame_observation".into(),
                second: 40,
                match_id: "match-1".into(),
                map: "Abyss".into(),
                map_number: 2,
                visual_left: "TL".into(),
                visual_right: "G2".into(),
                left_score: Some(3),
                right_score: Some(1),
                plate_ok: true,
                scene: "gameplay".into(),
                video_score: scores(&[("TL", 3), ("G2", 1)]),
                riot_blue: "G2".into(),
                riot_red: "TL".into(),
                left_confidence: 0.99,
                right_confidence: 0.98,
                ..Default::default()
            },
            json!({
                "second": 40, "kind": "frame_observation", "map": "Abyss", "map_number": 2,
                "match_id": "match-1", "score": null, "winner": "", "source": "", "held": false,
                "visual_left": "TL", "visual_right": "G2", "left_score": 3, "right_score": 1,
                "video_score": {"TL": 3, "G2": 1}, "plate_ok": true, "scene": "gameplay",
                "left_confidence": 0.99, "right_confidence": 0.98,
                "riot_state": "", "riot_blue": "G2", "riot_red": "TL",
            }),
        ),
        (
            "map start",
            Event {
                kind: "map_start".into(),
                second: 5,
                map: "Ascent".into(),
                map_number: 1,
                match_id: "match-1".into(),
                score: scores(&[("LOUD", 0), ("EDG", 0)]),
                riot_state: "unstarted".into(),
                riot_blue: "LOUD".into(),
                riot_red: "EDG".into(),
                ..Default::default()
            },
            json!({
                "second": 5, "kind": "map_start", "map": "Ascent", "map_number": 1,
                "match_id": "match-1", "score": {"LOUD": 0, "EDG": 0}, "winner": "",
                "source": "", "held": false,
                "riot_state": "unstarted", "riot_blue": "LOUD", "riot_red": "EDG",
            }),
        ),
        (
            "held round",
            Event {
                kind: "round".into(),
                second: 40,
                map: "Ascent".into(),
                map_number: 1,
                match_id: "match-1".into(),
                score: scores(&[("LOUD", 12), ("EDG", 10)]),
                winner: "LOUD".into(),
                held: true,
                ..Default::default()
            },
            json!({
                "second": 40, "kind": "round", "map": "Ascent", "map_number": 1,
                "match_id": "match-1", "score": {"LOUD": 12, "EDG": 10}, "winner": "LOUD",
                "source": "", "held": true,
            }),
        ),
        (
            "sync",
            Event {
                kind: "sync".into(),
                second: 70,
                map: "Ascent".into(),
                map_number: 1,
                match_id: "match-1".into(),
                score: scores(&[("LOUD", 8), ("EDG", 6)]),
                ..Default::default()
            },
            json!({
                "second": 70, "kind": "sync", "map": "Ascent", "map_number": 1,
                "match_id": "match-1", "score": {"LOUD": 8, "EDG": 6}, "winner": "",
                "source": "", "held": false,
            }),
        ),
        (
            "reject",
            Event {
                kind: "reject".into(),
                second: 20,
                map: "Ascent".into(),
                map_number: 1,
                match_id: "match-1".into(),
                score: scores(&[("LOUD", 4), ("EDG", 3)]),
                read_score: scores(&[("LOUD", 3), ("EDG", 3)]),
                ..Default::default()
            },
            json!({
                "second": 20, "kind": "reject", "map": "Ascent", "map_number": 1,
                "match_id": "match-1", "score": {"LOUD": 4, "EDG": 3},
                "read_score": {"LOUD": 3, "EDG": 3}, "winner": "", "source": "", "held": false,
            }),
        ),
        (
            "score held",
            Event {
                kind: "score_held".into(),
                second: 45,
                map: "Ascent".into(),
                map_number: 1,
                match_id: "match-1".into(),
                score: scores(&[("LOUD", 12), ("EDG", 10)]),
                read_score: scores(&[("LOUD", 13), ("EDG", 10)]),
                held: true,
                ..Default::default()
            },
            json!({
                "second": 45, "kind": "score_held", "map": "Ascent", "map_number": 1,
                "match_id": "match-1", "score": {"LOUD": 12, "EDG": 10},
                "read_score": {"LOUD": 13, "EDG": 10}, "winner": "", "source": "", "held": true,
            }),
        ),
        (
            "score corroborated",
            Event {
                kind: "score_corroborated".into(),
                second: 50,
                map: "Ascent".into(),
                map_number: 1,
                match_id: "match-1".into(),
                score: scores(&[("LOUD", 13), ("EDG", 10)]),
                read_score: scores(&[("LOUD", 13), ("EDG", 10)]),
                ..Default::default()
            },
            json!({
                "second": 50, "kind": "score_corroborated", "map": "Ascent", "map_number": 1,
                "match_id": "match-1", "score": {"LOUD": 13, "EDG": 10},
                "read_score": {"LOUD": 13, "EDG": 10}, "winner": "", "source": "", "held": false,
            }),
        ),
        (
            "score rollback",
            Event {
                kind: "score_rollback".into(),
                second: 55,
                map: "Ascent".into(),
                map_number: 1,
                match_id: "match-1".into(),
                score: scores(&[("LOUD", 12), ("EDG", 10)]),
                read_score: scores(&[("LOUD", 12), ("EDG", 10)]),
                source: "confirmed_held_score".into(),
                ..Default::default()
            },
            json!({
                "second": 55, "kind": "score_rollback", "map": "Ascent", "map_number": 1,
                "match_id": "match-1", "score": {"LOUD": 12, "EDG": 10},
                "read_score": {"LOUD": 12, "EDG": 10}, "winner": "",
                "source": "confirmed_held_score", "held": false,
            }),
        ),
        (
            "hidden map end",
            Event {
                kind: "map_end".into(),
                second: 60,
                map: "Ascent".into(),
                map_number: 1,
                match_id: "match-1".into(),
                score: scores(&[("LOUD", 13), ("EDG", 10)]),
                winner: "LOUD".into(),
                source: "riot".into(),
                exact: Some(true),
                ..Default::default()
            },
            json!({
                "second": 60, "kind": "map_end", "map": "Ascent", "map_number": 1,
                "match_id": "match-1", "score": {"LOUD": 13, "EDG": 10}, "winner": "LOUD",
                "source": "riot", "held": false, "exact": true,
            }),
        ),
        (
            "sides corrected",
            Event {
                kind: "sides_corrected".into(),
                second: 30,
                map: "Ascent".into(),
                map_number: 1,
                match_id: "match-1".into(),
                score: scores(&[("LOUD", 3), ("EDG", 2)]),
                riot_state: "inProgress".into(),
                riot_blue: "EDG".into(),
                riot_red: "LOUD".into(),
                ..Default::default()
            },
            json!({
                "second": 30, "kind": "sides_corrected", "map": "Ascent", "map_number": 1,
                "match_id": "match-1", "score": {"LOUD": 3, "EDG": 2}, "winner": "",
                "source": "", "held": false,
                "riot_state": "inProgress", "riot_blue": "EDG", "riot_red": "LOUD",
            }),
        ),
        (
            "pending end",
            Event {
                kind: "pending_end".into(),
                second: 65,
                map: "Ascent".into(),
                map_number: 1,
                match_id: "match-1".into(),
                score: scores(&[("LOUD", 12), ("EDG", 10)]),
                winner: "LOUD".into(),
                ..Default::default()
            },
            json!({
                "second": 65, "kind": "pending_end", "map": "Ascent", "map_number": 1,
                "match_id": "match-1", "score": {"LOUD": 12, "EDG": 10}, "winner": "LOUD",
                "source": "", "held": false,
            }),
        ),
        (
            "pending cleared",
            Event {
                kind: "pending_cleared".into(),
                second: 68,
                map: "Ascent".into(),
                map_number: 1,
                match_id: "match-1".into(),
                score: scores(&[("LOUD", 12), ("EDG", 10)]),
                ..Default::default()
            },
            json!({
                "second": 68, "kind": "pending_cleared", "map": "Ascent", "map_number": 1,
                "match_id": "match-1", "score": {"LOUD": 12, "EDG": 10}, "winner": "",
                "source": "", "held": false,
            }),
        ),
        (
            // A pause can precede the first map, so no run identity is attached yet.
            "pause start",
            Event {
                kind: "pause_start".into(),
                second: 12,
                pause: "tech_pause".into(),
                reason: "GEAR".into(),
                pause_since: 10,
                ..Default::default()
            },
            json!({
                "second": 12, "kind": "pause_start", "map": "", "map_number": 0,
                "match_id": "", "score": null, "winner": "", "source": "", "held": false,
                "pause": "tech_pause", "reason": "GEAR", "pause_since": 10,
            }),
        ),
        (
            "pause end",
            Event {
                kind: "pause_end".into(),
                second: 90,
                pause: "tech_pause".into(),
                reason: "GEAR".into(),
                pause_since: 10,
                source: "overlay_gone".into(),
                ..Default::default()
            },
            json!({
                "second": 90, "kind": "pause_end", "map": "", "map_number": 0,
                "match_id": "", "score": null, "winner": "", "source": "overlay_gone",
                "held": false, "pause": "tech_pause", "reason": "GEAR", "pause_since": 10,
            }),
        ),
    ];

    for (name, event, want) in cases {
        let got = event_value(&event);
        assert_eq!(got, want, "{name}");
    }
}

// A stream connection is a lifecycle line in the same ledger.
#[test]
fn test_ledger_records_stream_lifecycle() {
    let path = std::env::temp_dir().join(format!(
        "vbt-ledger-lifecycle-{}-{}.jsonl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut ledger = ledger::Ledger::open(path.to_str().unwrap()).expect("open ledger");
    ledger.lifecycle("stream_connect", "https://example.invalid/stream", 3);
    drop(ledger);
    let data = std::fs::read_to_string(&path).expect("read ledger");
    let _ = std::fs::remove_file(&path);
    let got: Value = serde_json::from_str(data.trim()).expect("decode lifecycle line");
    assert_eq!(got["kind"], "stream_connect");
    assert_eq!(got["second"], 3);
    assert_eq!(got["stream"], "https://example.invalid/stream");
    let at = got["at"].as_str().expect("at");
    time::OffsetDateTime::parse(at, &time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|e| panic!("at = {at:?}: {e}"));
}
