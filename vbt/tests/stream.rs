//! Stream loop tests.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use vbt::context::teams_match;
use vbt::ledger::Ledger;
use vbt::stream::{self, SOURCE_BENCH, SourcePool};
use vbt::tracker::{Riot, ScoreMap, Sides, Team, Tracker};
use vbt_vision::templates::Templates;

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn test_source_pool_benches_and_recovers() {
    let mut pool = SourcePool::new(&["a".to_string(), "b".to_string()]);
    let now = Instant::now();

    assert_eq!(pool.choose(now), "a");
    pool.fail("a", now);
    pool.fail("a", now);
    assert!(pool.fail("a", now), "third failure should bench the source");
    assert_eq!(pool.choose(now), "b");
    pool.fail("b", now);
    pool.fail("b", now);
    pool.fail("b", now);
    assert_eq!(pool.choose(now), "", "all benched");
    let later = now + SOURCE_BENCH + Duration::from_secs(1);
    assert_eq!(pool.choose(later), "a", "after the bench expires");
}

// The context is bound by team codes, not match id.
#[test]
fn test_teams_match_binds_by_code() {
    let case = |wanted: &[&str], got: &[&str]| {
        (
            wanted.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            got.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        )
    };
    let (w, g) = case(&["LOUD", "EDG"], &["LOUD", "EDG"]);
    assert!(teams_match(&w, &g), "exact codes");
    let (w, g) = case(&["loud", " EDG "], &["LOUD", "EDG"]);
    assert!(teams_match(&w, &g), "case and space insensitive");
    let (w, g) = case(&["EDG"], &["LOUD", "EDG"]);
    assert!(teams_match(&w, &g), "a tag may stand in for a name");
    let (w, g) = case(&["LOUD", "EDG"], &["GAM", "DEL"]);
    assert!(!teams_match(&w, &g), "another match is refused");
    let (w, g) = case(&["LOUD"], &[]);
    assert!(!teams_match(&w, &g), "an empty response is refused");
}

/// A Riot boundary that cancels the run on its second liveness check.
struct CancellingLive {
    checks: Arc<AtomicUsize>,
    cancel: vbt::cancel::CancellationFlag,
}

impl Riot for CancellingLive {
    fn live_now(&self) -> bool {
        let checks = self.checks.fetch_add(1, Ordering::SeqCst) + 1;
        if checks > 1 {
            self.cancel.store(true, Ordering::SeqCst);
        }
        true
    }

    fn has_event(&self) -> bool {
        false
    }

    fn match_id(&self) -> String {
        String::new()
    }

    fn teams(&self) -> Vec<Team> {
        Vec::new()
    }

    fn sides(&self, _match_id: &str, _map_number: i64) -> Option<Sides> {
        None
    }

    fn listed_order(&self, _match_id: &str) -> Sides {
        Sides::default()
    }

    fn map_number(&self, maps_played: i64) -> i64 {
        maps_played + 1
    }

    fn game_state(&self, _map_number: i64) -> Option<String> {
        None
    }

    fn game_wins(&self) -> ScoreMap {
        ScoreMap::new()
    }

    fn best_of(&self) -> i64 {
        0
    }
}

// A connection attempt that fails to open must still leave a lifecycle line, so
// every source the runner tried is visible in a later debugging pass.
#[test]
fn test_failed_stream_open_records_stream_failed() {
    let _guard = ENV_LOCK.lock().unwrap();
    // An empty PATH makes streamlink unlaunchable, so opening the stream fails without
    // touching the network.
    let original = std::env::var_os("PATH");
    let empty = std::env::temp_dir().join(format!("vbt-empty-path-{}", std::process::id()));
    std::fs::create_dir_all(&empty).unwrap();
    unsafe {
        std::env::set_var("PATH", &empty);
    }

    let cancel: vbt::cancel::CancellationFlag = Arc::new(AtomicBool::new(false));
    let live = CancellingLive {
        checks: Arc::new(AtomicUsize::new(0)),
        cancel: cancel.clone(),
    };
    let mut tracker = Tracker::new(
        Templates::load_embedded().expect("templates"),
        None,
        Box::new(CancellingLive {
            checks: Arc::new(AtomicUsize::new(0)),
            cancel: cancel.clone(),
        }),
        None,
        None,
    );
    let ledger_path = std::env::temp_dir().join(format!(
        "vbt-stream-failed-{}-{}.jsonl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut ledger = Ledger::open(ledger_path.to_str().unwrap()).expect("open ledger");
    const SOURCE: &str = "https://example.invalid/stream";
    {
        let mut lifecycle =
            |kind: &str, url: &str, second: i64| ledger.lifecycle(kind, url, second);
        stream::live(
            &cancel,
            &mut tracker,
            &live,
            &[SOURCE.to_string()],
            &mut lifecycle,
        );
    }
    drop(ledger);

    let data = std::fs::read_to_string(&ledger_path).expect("read ledger");
    let _ = std::fs::remove_file(&ledger_path);
    let mut kinds = Vec::new();
    for raw in data.lines() {
        let line: serde_json::Value = serde_json::from_str(raw).expect("decode line");
        kinds.push(line["kind"].as_str().unwrap_or_default().to_string());
        if line["kind"] == "stream_failed" {
            assert_eq!(
                line["stream"], SOURCE,
                "stream_failed wants the attempted source"
            );
        }
    }
    assert!(
        kinds.len() >= 2 && kinds[0] == "stream_connect" && kinds[1] == "stream_failed",
        "ledger kinds = {kinds:?}, want a connect attempt followed by its failure"
    );

    unsafe {
        match original {
            Some(path) => std::env::set_var("PATH", path),
            None => std::env::remove_var("PATH"),
        }
    }
    let _ = std::fs::remove_dir_all(&empty);
}

/// `--verbose` must reach the DEBUG events: the fmt subscriber's own INFO default
/// gated them out before the `Targets` filter was ever consulted.
#[test]
fn test_verbose_logs_debug_lines() {
    use std::os::unix::fs::PermissionsExt;

    let dir = std::env::temp_dir().join(format!("vbt-verbose-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // Six black 1280x720 RGB24 frames; the scheduled observation logs `frame` at
    // DEBUG on every fifth second.
    let ffmpeg = dir.join("ffmpeg");
    std::fs::write(
        &ffmpeg,
        "#!/bin/sh\nexec /usr/bin/head -c 16588800 /dev/zero\n",
    )
    .unwrap();
    std::fs::set_permissions(&ffmpeg, std::fs::Permissions::from_mode(0o700)).unwrap();

    let run = |verbose: bool| {
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_vbt"));
        command
            .arg("--file")
            .arg("unused")
            .current_dir(&dir)
            .env("PATH", &dir)
            .env_remove("BACKEND_SOCKET")
            .env_remove("VIDEO_TOKEN_FILE")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped());
        if verbose {
            command.arg("--verbose");
        }
        let output = command.output().expect("spawn vbt");
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        assert!(
            output.status.success(),
            "vbt exited with {}:\n{stderr}",
            output.status
        );
        stderr
    };

    let verbose = run(true);
    assert!(
        verbose.contains("DEBUG"),
        "--verbose produced no DEBUG lines:\n{verbose}"
    );
    assert!(
        verbose.contains("frame"),
        "--verbose produced no frame line:\n{verbose}"
    );
    let quiet = run(false);
    assert!(
        !quiet.contains("DEBUG"),
        "DEBUG lines without --verbose:\n{quiet}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A decoder that never writes a complete frame must still be killed and reaped
/// promptly on cancellation (the SIGTERM handler sets this same flag). Linux only:
/// it checks `/proc` for the decoder.
#[cfg(target_os = "linux")]
#[test]
fn test_replay_cancels_a_blocked_decoder_and_reaps_it() {
    use std::os::unix::fs::PermissionsExt;
    let _guard = ENV_LOCK.lock().unwrap();
    let dir = std::env::temp_dir().join(format!("vbt-blocked-decoder-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let pid_file = dir.join("pid");
    let decoder = dir.join("ffmpeg");
    std::fs::write(
        &decoder,
        format!(
            "#!/bin/sh\necho $$ > {}\nexec /usr/bin/sleep 3\n",
            pid_file.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&decoder, std::fs::Permissions::from_mode(0o700)).unwrap();
    let original = std::env::var_os("PATH");
    unsafe {
        std::env::set_var("PATH", &dir);
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = cancel.clone();
    let worker = std::thread::spawn(move || {
        let mut tracker = Tracker::new(
            Templates::load_embedded().unwrap(),
            None,
            Box::new(vbt::riot::LiveRiot::default()),
            None,
            None,
        );
        stream::replay(&flag, &mut tracker, "unused").unwrap();
    });
    let deadline = Instant::now() + Duration::from_secs(2);
    while !pid_file.exists() {
        assert!(Instant::now() < deadline, "decoder never launched");
        std::thread::sleep(Duration::from_millis(10));
    }
    let pid = std::fs::read_to_string(&pid_file).unwrap();
    let start = Instant::now();
    cancel.store(true, Ordering::SeqCst);
    worker.join().unwrap();
    unsafe {
        match original {
            Some(path) => std::env::set_var("PATH", path),
            None => std::env::remove_var("PATH"),
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(
        start.elapsed() < Duration::from_secs(1),
        "blocked read delayed cancellation"
    );
    assert!(
        !std::path::Path::new(&format!("/proc/{}", pid.trim())).exists(),
        "decoder was not reaped"
    );
}

/// The replay map order trims entries and refuses out-of-range numbers.
#[test]
fn test_static_map_order_trims() {
    use vbt::context::StaticMapOrder;
    use vbt::tracker::MapOrder;
    let order = StaticMapOrder::new(vec![" Ascent ".to_string(), "Lotus".to_string()]);
    assert_eq!(order.map_name("", 1), "Ascent");
    assert_eq!(order.map_name("", 2), "Lotus");
    assert_eq!(order.map_name("", 0), "");
    assert_eq!(order.map_name("", 3), "");
}
