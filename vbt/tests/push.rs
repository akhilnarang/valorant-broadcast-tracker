//! Score publisher tests, against a hand-rolled Unix-socket HTTP/1.1 responder.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use base64::Engine as _;
use vbt::push::{Config, Context, Pusher, TOKEN_BYTES};
use vbt::tracker::{Pause, ScoreMap, Status, Team};

/// One request the fake backend saw.
#[derive(Debug, Clone)]
struct Captured {
    method: String,
    path: String,
    raw_query: String,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
}

type Handler = Arc<dyn Fn(Captured) -> (u16, &'static str, Vec<u8>) + Send + Sync>;

/// Serves the handler over the Unix socket at `path`.
fn unix_server(path: &Path, handler: Handler) {
    let listener = UnixListener::bind(path).expect("bind unix socket");
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else {
                continue;
            };
            let handler = handler.clone();
            std::thread::spawn(move || {
                if let Some(request) = read_request(&mut stream) {
                    let (status, reason, body) = handler(request);
                    let head = format!(
                        "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(head.as_bytes());
                    let _ = stream.write_all(&body);
                }
            });
        }
    });
}

fn read_request(stream: &mut UnixStream) -> Option<Captured> {
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .ok()?;
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
        let Some(head_end) = find_subslice(&buf, b"\r\n\r\n") else {
            continue;
        };
        let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
        let mut lines = head.split("\r\n");
        let request_line = lines.next()?;
        let mut parts = request_line.split_whitespace();
        let method = parts.next()?.to_string();
        let target = parts.next()?.to_string();
        let (path, raw_query) = match target.split_once('?') {
            Some((path, query)) => (path.to_string(), query.to_string()),
            None => (target, String::new()),
        };
        let mut headers = BTreeMap::new();
        for line in lines {
            if let Some((key, value)) = line.split_once(':') {
                headers.insert(key.trim().to_ascii_lowercase(), value.trim().to_string());
            }
        }
        let content_length: usize = headers
            .get("content-length")
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        if buf.len() >= head_end + 4 + content_length {
            let body = buf[head_end + 4..head_end + 4 + content_length].to_vec();
            return Some(Captured {
                method,
                path,
                raw_query,
                headers,
                body,
            });
        }
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Creates a short per-test directory. `temp_dir()` is ~50 bytes on macOS and a
/// Unix socket path must stay under 104 bytes (`sun_path`), so use `/tmp` and a
/// counter instead of a nanosecond timestamp.
fn temp_dir(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let dir = PathBuf::from("/tmp").join(format!(
        "vbt-p-{tag}-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn score(entries: &[(&str, i32)]) -> ScoreMap {
    entries
        .iter()
        .map(|(code, value)| ((*code).to_string(), *value))
        .collect()
}

// The token file holds the token and has mode 0600.
#[test]
fn test_write_token_creates_and_forces_0600() {
    for existing in [false, true] {
        let name = if existing { "existing" } else { "created" };
        let dir = temp_dir("token");
        let path = dir.join("video-token");
        if existing {
            std::fs::write(&path, b"stale").unwrap();
        }
        let mut pusher = Pusher::new(Config {
            token_file: path.to_string_lossy().into_owned(),
            ..Default::default()
        })
        .unwrap();
        pusher
            .write_token()
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let metadata = std::fs::metadata(&path).unwrap();
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(metadata.permissions().mode() & 0o777, 0o600, "{name}: mode");
        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, pusher.token, "{name}: token file content");
        let raw = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(&pusher.token)
            .unwrap_or_else(|e| panic!("{name}: token is not unpadded URL-safe base64: {e}"));
        assert_eq!(raw.len(), TOKEN_BYTES, "{name}: token entropy");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

// A second live writer must fail on the token lock.
#[test]
fn test_second_live_writer_fails_on_lock() {
    let dir = temp_dir("lock");
    let path = dir.join("video-token");
    let config = Config {
        token_file: path.to_string_lossy().into_owned(),
        ..Default::default()
    };
    let mut first = Pusher::new(config.clone()).unwrap();
    first.write_token().expect("first writer owns the token");
    let mut second = Pusher::new(config).unwrap();
    let error = second.write_token().expect_err("second writer must fail");
    assert!(
        matches!(error, vbt::push::Error::Locked(_)),
        "second writer error = {error}"
    );
    // The first writer's token file is untouched.
    assert_eq!(std::fs::read_to_string(&path).unwrap(), first.token);
    let _ = std::fs::remove_dir_all(&dir);
}

// The score request's method, path, headers and body.
#[tokio::test]
async fn test_publish_writes_score_over_unix_socket() {
    let dir = temp_dir("publish");
    let socket = dir.join("s.sock");
    let (tx, rx) = std::sync::mpsc::channel::<Captured>();
    unix_server(
        &socket,
        Arc::new(move |request| {
            let _ = tx.send(request);
            (204, "No Content", Vec::new())
        }),
    );

    let mut pusher = Pusher::new(Config {
        socket: socket.to_string_lossy().into_owned(),
        token_file: dir.join("video-token").to_string_lossy().into_owned(),
        ..Default::default()
    })
    .unwrap();
    assert!(
        pusher.config.enabled(),
        "enabled with socket and token file"
    );
    pusher.write_token().unwrap();

    let teams = vec![
        Team {
            id: "1120".into(),
            code: "LOUD".into(),
            name: "LOUD".into(),
        },
        Team {
            id: "474".into(),
            code: "EDG".into(),
            name: "EDward Gaming".into(),
        },
    ];
    let before = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    pusher
        .publish(
            2,
            &score(&[("LOUD", 8), ("EDG", 5)]),
            &teams,
            "LOUD",
            Status::Ok,
            None,
        )
        .await
        .expect("publish");
    let after = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    let got = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("backend got no request");
    assert_eq!(got.method, "PUT");
    assert_eq!(got.path, vbt::push::SCORE_PATH);
    assert_eq!(
        got.headers.get("content-type").map(String::as_str),
        Some("application/json")
    );
    assert_eq!(
        got.headers.get("x-video-token").map(String::as_str),
        Some(pusher.token.as_str())
    );

    let body: serde_json::Value = serde_json::from_slice(&got.body).expect("body is JSON");
    let keys: Vec<&str> = body
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        ["map_number", "observed_at", "status", "teams"],
        "exactly four fields"
    );
    assert_eq!(body["status"], "ok");
    assert_eq!(body["map_number"], 2);
    let observed_at = body["observed_at"].as_i64().unwrap();
    assert!(observed_at >= before && observed_at <= after, "observed_at");
    assert_eq!(
        body["teams"],
        serde_json::json!([
            {"code": "EDG", "name": "EDward Gaming", "score": 5, "side": "red"},
            {"code": "LOUD", "name": "LOUD", "score": 8, "side": "blue"},
        ])
    );

    // A pause in progress rides the same body as a fifth, optional field.
    pusher
        .publish(
            2,
            &score(&[("LOUD", 8), ("EDG", 5)]),
            &teams,
            "LOUD",
            Status::Ok,
            Some(&Pause {
                kind: "tech_pause".into(),
                reason: "GEAR".into(),
                since: 120,
            }),
        )
        .await
        .expect("publish with pause");
    let got = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("paused request");
    let body: serde_json::Value = serde_json::from_slice(&got.body).unwrap();
    assert_eq!(
        body["pause"],
        serde_json::json!({"kind": "tech_pause", "reason": "GEAR", "since": 120})
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// A backend that refuses the score fails the publish.
#[tokio::test]
async fn test_publish_rejects_non_204() {
    let dir = temp_dir("reject");
    let socket = dir.join("s.sock");
    // A huge error body that never finishes must return after its first 200
    // bytes, not consume the whole response or wait for the request timeout.
    let listener = UnixListener::bind(&socket).unwrap();
    let (release, held) = std::sync::mpsc::channel::<()>();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        read_request(&mut stream).unwrap();
        stream.write_all(b"HTTP/1.1 422 Unprocessable Entity\r\nContent-Length: 1073741824\r\nConnection: close\r\n\r\n").unwrap();
        let mut prefix = br#"{"detail":"invalid body"}"#.to_vec();
        prefix.resize(200, b'x');
        stream.write_all(&prefix).unwrap();
        let _ = held.recv_timeout(Duration::from_secs(5));
    });
    let pusher = Pusher::new(Config {
        socket: socket.to_string_lossy().into_owned(),
        token_file: dir.join("video-token").to_string_lossy().into_owned(),
        ..Default::default()
    })
    .unwrap();
    let error = tokio::time::timeout(
        Duration::from_secs(1),
        pusher.publish(
            1,
            &score(&[("LOUD", 3)]),
            &[Team {
                id: String::new(),
                code: "LOUD".into(),
                name: "LOUD".into(),
            }],
            "",
            Status::Error,
            None,
        ),
    )
    .await
    .expect("error response was drained instead of bounded")
    .expect_err("a 422 was accepted");
    drop(release);
    server.join().unwrap();
    let message = error.to_string();
    assert!(
        message.contains("422"),
        "error {message:?} does not name the status"
    );
    assert!(
        message.contains("invalid body"),
        "error {message:?} does not carry the body"
    );
    assert!(
        message.len() < 300,
        "error body exceeded the 200-byte limit"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// The context request and its answer, in live and read-only mode.
#[tokio::test]
async fn test_fetch_context_over_unix_socket() {
    let dir = temp_dir("context");
    let socket = dir.join("s.sock");
    let (tx, rx) = std::sync::mpsc::channel::<Captured>();
    unix_server(
        &socket,
        Arc::new(move |request| {
            let _ = tx.send(request);
            (
                200,
                "OK",
                br#"{"match_id":"123","map_number":2,"map_order":["Ascent","Lotus"],"teams":["LOUD","EDG"]}"#
                    .to_vec(),
            )
        }),
    );
    let mut pusher = Pusher::new(Config {
        socket: socket.to_string_lossy().into_owned(),
        token_file: dir.join("video-token").to_string_lossy().into_owned(),
        ..Default::default()
    })
    .unwrap();
    let got = pusher
        .fetch_context(&["LOUD".to_string(), "EDG".to_string()])
        .await
        .expect("fetch context");
    assert_eq!(
        got,
        Context {
            match_id: "123".into(),
            map_order: vec!["Ascent".into(), "Lotus".into()],
            teams: vec!["LOUD".into(), "EDG".into()],
        }
    );
    let request = rx.recv_timeout(Duration::from_secs(5)).expect("no request");
    assert_eq!(request.method, "GET");
    assert_eq!(request.path, vbt::push::CONTEXT_PATH);
    assert_eq!(request.raw_query, "codes=LOUD%2CEDG");
    assert_eq!(
        request.headers.get("x-video-token").map(String::as_str),
        Some(pusher.token.as_str())
    );

    // Shadow uses the existing primary token for GETs but cannot own it or send PUTs.
    std::fs::write(dir.join("video-token"), b"primary-token").unwrap();
    pusher.config.read_only = true;
    assert!(!pusher.config.enabled(), "read-only pusher enabled writes");
    assert!(matches!(
        pusher.write_token(),
        Err(vbt::push::Error::Disabled)
    ));
    assert!(matches!(
        pusher
            .publish(2, &score(&[("LOUD", 8)]), &[], "LOUD", Status::Ok, None)
            .await,
        Err(vbt::push::Error::Disabled)
    ));
    pusher
        .fetch_context(&["LOUD".to_string(), "EDG".to_string()])
        .await
        .expect("shadow fetch");
    let request = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("shadow request");
    assert_eq!(request.method, "GET");
    assert_eq!(
        request.headers.get("x-video-token").map(String::as_str),
        Some("primary-token")
    );
    assert!(
        rx.try_recv().is_err(),
        "no extra request (a PUT) from the read-only pusher"
    );
    assert_eq!(
        std::fs::read_to_string(dir.join("video-token")).unwrap(),
        "primary-token",
        "primary token changed"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// A backend that cannot resolve the match fails the fetch.
#[tokio::test]
async fn test_fetch_context_rejects_404() {
    let dir = temp_dir("context404");
    let socket = dir.join("s.sock");
    unix_server(
        &socket,
        Arc::new(|_| {
            (
                404,
                "Not Found",
                [
                    br#"{"detail":"No video score is stored"}"#.as_slice(),
                    &vec![b'x'; 4096],
                    b"UNREAD_TAIL",
                ]
                .concat(),
            )
        }),
    );
    let pusher = Pusher::new(Config {
        socket: socket.to_string_lossy().into_owned(),
        token_file: dir.join("video-token").to_string_lossy().into_owned(),
        ..Default::default()
    })
    .unwrap();
    let error = pusher
        .fetch_context(&[])
        .await
        .expect_err("a 404 was accepted");
    let message = error.to_string();
    assert!(
        message.contains("404"),
        "error {message:?} does not name the status"
    );
    assert!(
        message.contains("No video score"),
        "error {message:?} does not carry the body"
    );
    assert!(
        !message.contains("UNREAD_TAIL"),
        "unbounded context error excerpt"
    );
    assert!(
        message.len() < 300,
        "context error body exceeded the 200-byte limit"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// A hung backend fails the publish after the 2 second timeout.
#[tokio::test]
async fn test_publish_times_out_on_hung_backend() {
    let dir = temp_dir("hung");
    let socket = dir.join("s.sock");
    let (release, held) = std::sync::mpsc::channel::<()>();
    let held = Arc::new(std::sync::Mutex::new(held));
    unix_server(
        &socket,
        Arc::new(move |_| {
            let _ = held.lock().unwrap().recv_timeout(Duration::from_secs(10));
            (204, "No Content", Vec::new())
        }),
    );
    let pusher = Pusher::new(Config {
        socket: socket.to_string_lossy().into_owned(),
        token_file: dir.join("video-token").to_string_lossy().into_owned(),
        ..Default::default()
    })
    .unwrap();
    let start = Instant::now();
    let error = pusher
        .publish(1, &score(&[("LOUD", 3)]), &[], "", Status::Ok, None)
        .await;
    let elapsed = start.elapsed();
    assert!(error.is_err(), "a hung backend was accepted");
    assert!(
        elapsed >= Duration::from_secs(1) && elapsed <= Duration::from_secs(5),
        "publish took {elapsed:?}, want about the 2s timeout"
    );
    drop(release);
    let _ = std::fs::remove_dir_all(&dir);
}

// Without a socket and token file, publishing and fetching are disabled.
#[tokio::test]
async fn test_disabled_when_unconfigured() {
    for config in [
        Config::default(),
        Config {
            socket: "/run/backend.sock".into(),
            ..Default::default()
        },
        Config {
            token_file: "video-token".into(),
            ..Default::default()
        },
    ] {
        let pusher = Pusher::new(config.clone()).unwrap();
        assert!(!pusher.config.enabled(), "enabled for {config:?}");
        assert!(
            matches!(
                pusher
                    .publish(1, &score(&[("LOUD", 3)]), &[], "", Status::Ok, None)
                    .await,
                Err(vbt::push::Error::Disabled)
            ),
            "Publish error for {config:?}"
        );
    }
}

// .env is used when a setting is unset, and the process environment wins over it.
#[test]
fn test_config_from_env() {
    let dir = temp_dir("dotenv");
    let path = dir.join(".env");
    std::fs::write(
        &path,
        "# the backend's socket\nexport BACKEND_SOCKET=\"/run/backend.sock\"\nVIDEO_TOKEN_FILE=video-token\nUNKNOWN=ignored\n",
    )
    .unwrap();
    let values = vbt::push::read_dotenv(&path);
    assert_eq!(
        values.get("BACKEND_SOCKET").map(String::as_str),
        Some("/run/backend.sock")
    );
    assert_eq!(
        values.get("VIDEO_TOKEN_FILE").map(String::as_str),
        Some("video-token")
    );
    assert!(
        !values.contains_key("UNKNOWN")
            || values.get("UNKNOWN").map(String::as_str) == Some("ignored")
    );
    let _ = std::fs::remove_dir_all(&dir);

    // The process environment wins over .env.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = ENV_LOCK.lock().unwrap();
    unsafe {
        std::env::set_var("BACKEND_SOCKET", "/run/override.sock");
    }
    let config = Config::from_env();
    unsafe {
        std::env::remove_var("BACKEND_SOCKET");
    }
    assert_eq!(
        config.socket, "/run/override.sock",
        "environment must win over .env"
    );
}
