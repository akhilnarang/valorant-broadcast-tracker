//! Riot poller tests, against a local TCP responder.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use vbt::riot::{API_KEY, LiveRiot, Poller, RiotState};
use vbt::tracker::Riot as RiotSource;

const LIVE_BODY: &str = r#"{"data":{"schedule":{"events":[
    {"id":"event-1","type":"match","match":{"id":"264451"}}
]}}}"#;

const DETAILS_BODY: &str = r#"{"data":{"event":{"match":{
    "id":"264451",
    "teams":[
        {"id":"1120","code":"LOUD","name":"LOUD","result":{"gameWins":1}},
        {"id":474,"code":"EDG","name":"EDward Gaming","result":{"gameWins":0}}
    ],
    "games":[
        {"number":1,"state":"completed","teams":[{"id":"1120","side":"blue"},{"id":474,"side":"red"}]},
        {"number":2,"state":"inProgress","teams":[{"id":474,"side":"blue"},{"id":"1120","side":"red"}]},
        {"number":3,"state":"unstarted"}
    ]
}}}}"#;

#[derive(Default)]
struct Recorded {
    paths: Mutex<Vec<String>>,
    queries: Mutex<Vec<String>>,
    headers: Mutex<Vec<BTreeMap<String, String>>>,
}

type Handler = Arc<dyn Fn(&str, &str) -> (u16, String) + Send + Sync>;

/// Serves HTTP/1.1 over a local TCP port, returning the base URL.
fn start_server(recorded: Arc<Recorded>, handler: Handler) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else {
                continue;
            };
            let handler = handler.clone();
            let recorded = recorded.clone();
            std::thread::spawn(move || {
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                loop {
                    let n = match stream.read(&mut chunk) {
                        Ok(0) | Err(_) => return,
                        Ok(n) => n,
                    };
                    buf.extend_from_slice(&chunk[..n]);
                    if find_subslice(&buf, b"\r\n\r\n").is_some() {
                        break;
                    }
                }
                let head = String::from_utf8_lossy(&buf).to_string();
                let mut lines = head.lines();
                let request_line = lines.next().unwrap_or("");
                let mut parts = request_line.split_whitespace();
                let _method = parts.next();
                let target = parts.next().unwrap_or("/");
                let (path, query) = match target.split_once('?') {
                    Some((path, query)) => (path.to_string(), query.to_string()),
                    None => (target.to_string(), String::new()),
                };
                let mut headers = BTreeMap::new();
                for line in lines {
                    if let Some((key, value)) = line.split_once(':') {
                        headers.insert(key.trim().to_ascii_lowercase(), value.trim().to_string());
                    }
                }
                {
                    let mut paths = recorded.paths.lock().unwrap();
                    paths.push(path.clone());
                    let mut queries = recorded.queries.lock().unwrap();
                    queries.push(query.clone());
                    let mut recorded_headers = recorded.headers.lock().unwrap();
                    recorded_headers.push(headers);
                }
                let (status, body) = handler(&path, &query);
                let reason = match status {
                    200 => "OK",
                    500 => "Internal Server Error",
                    _ => "Error",
                };
                let response = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
            });
        }
    });
    format!("http://{addr}/persisted/val")
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn test_poller(base: &str, state: &Arc<Mutex<RiotState>>) -> Poller {
    Poller::with_base_url(base, state.clone()).unwrap()
}

// The poll finds the live match, fetches its details, and every accessor reads them.
#[tokio::test]
async fn test_poll_parses_live_match() {
    let recorded = Arc::new(Recorded::default());
    let base = start_server(
        recorded.clone(),
        Arc::new(|path, _query| match path {
            "/persisted/val/getLiveDetails" => (200, LIVE_BODY.to_string()),
            "/persisted/val/getEventDetails" => (200, DETAILS_BODY.to_string()),
            _ => (404, "not found".to_string()),
        }),
    );
    let state = Arc::new(Mutex::new(RiotState::default()));
    let poller = test_poller(&base, &state);
    let riot = LiveRiot::from_state(state.clone());

    // Before the first poll: no match, so the placeholders stand in.
    assert!(!riot.live_now());
    assert!(!riot.has_event());
    assert_eq!(riot.match_id(), "");
    assert_eq!(
        riot.teams()
            .iter()
            .map(|t| t.code.clone())
            .collect::<Vec<_>>(),
        ["LEFT", "RIGHT"]
    );
    assert_eq!(
        riot.listed_order(""),
        vbt::tracker::Sides {
            blue: "LEFT".into(),
            red: "RIGHT".into()
        }
    );
    assert_eq!(riot.map_number(0), 1);
    assert!(riot.game_wins().is_empty());
    assert_eq!(riot.game_state(1), None);

    poller.poll().await.expect("poll");

    assert!(riot.live_now());
    assert!(riot.has_event());
    assert_eq!(riot.match_id(), "264451");
    let teams: Vec<(String, String, String)> = riot
        .teams()
        .into_iter()
        .map(|team| (team.id, team.code, team.name))
        .collect();
    assert_eq!(
        teams,
        [
            ("1120".to_string(), "LOUD".to_string(), "LOUD".to_string()),
            (
                "474".to_string(),
                "EDG".to_string(),
                "EDward Gaming".to_string()
            ),
        ]
    );
    assert_eq!(
        riot.listed_order("264451"),
        vbt::tracker::Sides {
            blue: "LOUD".into(),
            red: "EDG".into()
        }
    );
    assert_eq!(
        riot.listed_order("another-match"),
        vbt::tracker::Sides {
            blue: "LEFT".into(),
            red: "RIGHT".into()
        }
    );
    assert_eq!(
        riot.sides("264451", 1),
        Some(vbt::tracker::Sides {
            blue: "LOUD".into(),
            red: "EDG".into()
        })
    );
    assert_eq!(
        riot.sides("264451", 2),
        Some(vbt::tracker::Sides {
            blue: "EDG".into(),
            red: "LOUD".into()
        })
    );
    assert_eq!(
        riot.sides("264451", 3),
        None,
        "Riot lists no sides for game 3"
    );
    assert_eq!(riot.sides("264451", 9), None);
    assert_eq!(riot.sides("another-match", 2), None);
    assert_eq!(riot.map_number(0), 2, "Riot's in-progress game");
    assert_eq!(riot.map_number(5), 2);
    assert_eq!(riot.game_state(1).as_deref(), Some("completed"));
    assert_eq!(riot.game_state(2).as_deref(), Some("inProgress"));
    assert_eq!(riot.game_state(3).as_deref(), Some("unstarted"));
    assert_eq!(riot.game_state(9), None);
    assert_eq!(
        riot.game_wins(),
        vbt::tracker::ScoreMap::from([("LOUD".to_string(), 1), ("EDG".to_string(), 0)])
    );

    let headers = recorded.headers.lock().unwrap();
    for header in headers.iter() {
        assert_eq!(header.get("x-api-key").map(String::as_str), Some(API_KEY));
    }
    let queries = recorded.queries.lock().unwrap();
    assert!(
        queries
            .iter()
            .all(|q| q.contains("hl=en-US") && q.contains("sport=val"))
    );
    assert!(
        queries.iter().any(|q| q.contains("id=264451")),
        "getEventDetails id = {queries:?}"
    );
}

// What counts as a live match, and where its id comes from.
#[tokio::test]
async fn test_poll_finds_only_live_matches() {
    let cases: [(&str, &str, bool, &str); 6] = [
        (
            "type match",
            r#"[{"id":"111","type":"match"}]"#,
            true,
            "111",
        ),
        (
            "match object",
            r#"[{"id":"222","match":{"id":"m-222"}}]"#,
            true,
            "m-222",
        ),
        (
            "numeric id",
            r#"[{"id":264451000,"type":"match"}]"#,
            true,
            "264451000",
        ),
        (
            "large numeric match id",
            r#"[{"type":"match","match":{"id":109996848902791075}}]"#,
            true,
            "109996848902791075",
        ),
        (
            "not a match",
            r#"[{"id":"333","type":"show","match":null}]"#,
            false,
            "",
        ),
        ("no events", "[]", false, ""),
    ];
    for (name, events, live, match_id) in cases {
        let details_calls = Arc::new(AtomicUsize::new(0));
        let counter = details_calls.clone();
        let body = format!(r#"{{"data":{{"schedule":{{"events":{events}}}}}}}"#);
        let base = start_server(
            Arc::new(Recorded::default()),
            Arc::new(move |path, _query| {
                if path == "/persisted/val/getEventDetails" {
                    counter.fetch_add(1, Ordering::SeqCst);
                    return (200, DETAILS_BODY.to_string());
                }
                (200, body.clone())
            }),
        );
        let state = Arc::new(Mutex::new(RiotState::default()));
        let poller = test_poller(&base, &state);
        let riot = LiveRiot::from_state(state.clone());
        poller
            .poll()
            .await
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(riot.live_now(), live, "{name}: LiveNow");
        assert_eq!(riot.match_id(), match_id, "{name}: MatchID");
        if !live {
            assert_eq!(
                details_calls.load(Ordering::SeqCst),
                0,
                "{name}: fetched event details without a live match"
            );
        }
    }

    // With two live matches the poller keeps the one it follows, and its details,
    // even when the details refresh fails. A fresh poller takes the first listed.
    let two =
        r#"{"data":{"schedule":{"events":[{"id":"A","type":"match"},{"id":"B","type":"match"}]}}}"#;
    let base = start_server(
        Arc::new(Recorded::default()),
        Arc::new(move |path, _query| {
            if path == "/persisted/val/getEventDetails" {
                return (500, String::new());
            }
            (200, two.to_string())
        }),
    );
    let state = Arc::new(Mutex::new(RiotState {
        live_now: true,
        match_id: "B".to_string(),
        event: Some(Default::default()),
    }));
    let riot = LiveRiot::from_state(state.clone());
    assert!(test_poller(&base, &state).poll().await.is_err());
    assert_eq!(riot.match_id(), "B");
    assert!(
        riot.has_event(),
        "two live matches dropped the followed match's details"
    );

    let state = Arc::new(Mutex::new(RiotState::default()));
    let riot = LiveRiot::from_state(state.clone());
    assert!(test_poller(&base, &state).poll().await.is_err());
    assert_eq!(riot.match_id(), "A");
}

// An API error changes nothing, and the next poll recovers.
#[tokio::test]
async fn test_failed_poll_keeps_last_details() {
    let failing = Arc::new(AtomicBool::new(false));
    let flag = failing.clone();
    let idle = Arc::new(AtomicBool::new(false));
    let idle_flag = idle.clone();
    let base = start_server(
        Arc::new(Recorded::default()),
        Arc::new(move |path, _query| {
            if flag.load(Ordering::SeqCst) {
                return (500, "boom".to_string());
            }
            if idle_flag.load(Ordering::SeqCst) && path.ends_with("getLiveDetails") {
                return (200, r#"{"data":{"schedule":{"events":null}}}"#.to_string());
            }
            if path == "/persisted/val/getEventDetails" {
                return (200, DETAILS_BODY.to_string());
            }
            (200, LIVE_BODY.to_string())
        }),
    );
    let state = Arc::new(Mutex::new(RiotState::default()));
    let poller = test_poller(&base, &state);
    let riot = LiveRiot::from_state(state.clone());

    poller.poll().await.expect("first poll");
    let want_teams = riot.teams();
    failing.store(true, Ordering::SeqCst);
    let error = poller.poll().await.expect_err("a failing API succeeded");
    assert!(
        error.to_string().contains("500"),
        "error {error} does not name the status"
    );
    assert!(riot.live_now());
    assert_eq!(riot.match_id(), "264451");
    assert_eq!(riot.teams(), want_teams);
    assert_eq!(
        riot.sides("264451", 2),
        Some(vbt::tracker::Sides {
            blue: "EDG".into(),
            red: "LOUD".into()
        })
    );

    failing.store(false, Ordering::SeqCst);
    poller.poll().await.expect("recovery poll");
    // Riot explicitly emits null while idle: unlike a failed poll, it must stop
    // the live stream rather than leaving the previous match active.
    idle.store(true, Ordering::SeqCst);
    poller.poll().await.expect("null schedule poll");
    assert!(
        !riot.live_now(),
        "null events kept the previous stream live"
    );
}

// The background loop polls at once, keeps going after failed polls, and returns
// on cancellation.
#[tokio::test]
async fn test_start_polls_and_stop_returns() {
    let polled = Arc::new(AtomicUsize::new(0));
    let counter = polled.clone();
    let base = start_server(
        Arc::new(Recorded::default()),
        Arc::new(move |_path, _query| {
            counter.fetch_add(1, Ordering::SeqCst);
            (500, "boom".to_string())
        }),
    );
    let state = Arc::new(Mutex::new(RiotState::default()));
    let poller = test_poller(&base, &state);
    let cancel: vbt::cancel::CancellationFlag = Arc::new(AtomicBool::new(false));
    let task = tokio::spawn(poller.run(cancel.clone()));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while polled.load(Ordering::SeqCst) == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "poller made no request"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    cancel.store(true, Ordering::SeqCst);
    tokio::time::timeout(std::time::Duration::from_secs(5), task)
        .await
        .expect("run did not return")
        .expect("task panicked");
}
