//! The stream pipeline: streamlink piped into ffmpeg for the live stream, a plain ffmpeg decode for
//! a replay, the source pool, and the frame consumer.

use std::collections::BTreeMap;
use std::io::{self, Read};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use vbt_vision::frame::Frame;

use crate::cancel::{CancellationFlag, cancelled, sleep_cancelled_blocking};
use crate::tracker::Riot as RiotSource;
use crate::tracker::{FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH, Tracker};

/// The ffmpeg decode arguments, exact.
pub const DECODE_VF: &str = "fps=1,scale=1280:720:flags=fast_bilinear,format=rgb24";
/// How long without a live match before the loop re-checks.
pub const IDLE_WAIT: Duration = Duration::from_secs(30);
/// How long without a frame counts as a stalled stream.
pub const STALL_TIMEOUT: Duration = Duration::from_secs(30);
/// Benches a source after this many consecutive failures.
pub const SOURCE_FAILURES: u32 = 3;
/// How long a benched source stays out of rotation.
pub const SOURCE_BENCH: Duration = Duration::from_secs(120);
/// How long to wait before the next connection attempt.
pub const RECONNECT: Duration = Duration::from_secs(10);

/// The sources tried in order; Twitch's global channel first.
pub const DEFAULT_SOURCES: [&str; 3] = [
    "https://www.twitch.tv/valorant",
    "https://www.twitch.tv/valorant_br",
    "https://www.youtube.com/@ValorantEsports/live",
];

/// Why a stream stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Ended,
    Failed,
}

/// Reads exactly one RGB24 frame; a short read means the stream ended.
pub fn read_frame(reader: &mut impl Read) -> io::Result<Frame> {
    let mut buf = vec![0u8; FRAME_BYTES];
    reader.read_exact(&mut buf)?;
    Frame::from_rgb(FRAME_WIDTH, FRAME_HEIGHT, buf)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
}

/// Starts streamlink piped into ffmpeg so ffmpeg needs no network of its own.
pub fn open_stream(url: &str) -> io::Result<(Vec<Child>, Box<dyn Read + Send>)> {
    let mut streamlink = Command::new("streamlink")
        .args([
            "--stdout",
            // Since streamlink 7.2.0 `--quiet` suppresses errors too, hiding the
            // reason a stream failed; warnings go to the inherited stderr.
            "--loglevel",
            "warning",
            "--stream-segment-timeout",
            "20",
            url,
            "720p30,720p,best",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let streamlink_stdout = streamlink
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("streamlink stdout unavailable"))?;
    let decoder = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            "pipe:0",
            "-an",
            "-vf",
            DECODE_VF,
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        .stdin(Stdio::from(streamlink_stdout))
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn();
    let mut decoder = match decoder {
        Ok(decoder) => decoder,
        Err(error) => {
            // streamlink is already running; without this it stays alive with its pipe open.
            stop_pipeline(&mut [streamlink]);
            return Err(error);
        }
    };
    let frames = decoder
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("ffmpeg stdout unavailable"))?;
    Ok((vec![streamlink, decoder], Box::new(frames)))
}

/// Kills and reaps a connection attempt's processes.
pub fn stop_pipeline(children: &mut [Child]) {
    for child in children.iter_mut() {
        let _ = child.kill();
    }
    for child in children.iter_mut() {
        let _ = child.wait();
    }
}

/// Owns subprocesses so all returns (including I/O failures) kill and reap them.
struct Pipeline(Vec<Child>);

impl Drop for Pipeline {
    fn drop(&mut self) {
        stop_pipeline(&mut self.0);
    }
}

/// The consumer waits interruptibly; killing the decoder releases a blocked pipe
/// read before its thread is joined. A zero-capacity channel bounds frame memory.
fn frame_reader(
    mut frames: Box<dyn Read + Send>,
) -> (
    std::sync::mpsc::Receiver<io::Result<Frame>>,
    std::thread::JoinHandle<()>,
) {
    let (tx, rx) = std::sync::mpsc::sync_channel(0);
    let reader = std::thread::spawn(move || {
        loop {
            let item = read_frame(&mut frames);
            let failed = item.is_err();
            if tx.send(item).is_err() || failed {
                return;
            }
        }
    });
    (rx, reader)
}

/// Picks a stream source, benching one that keeps failing.
#[derive(Debug, Clone)]
pub struct SourcePool {
    sources: Vec<String>,
    failures: BTreeMap<String, u32>,
    until: BTreeMap<String, Instant>,
    current: String,
}

impl SourcePool {
    pub fn new(sources: &[String]) -> SourcePool {
        SourcePool {
            sources: sources.to_vec(),
            failures: BTreeMap::new(),
            until: BTreeMap::new(),
            current: String::new(),
        }
    }

    /// Returns the current source, dropping benched ones back in when their bench
    /// expires.
    pub fn choose(&mut self, now: Instant) -> String {
        let mut available: Vec<String> = Vec::with_capacity(self.sources.len());
        for url in &self.sources {
            if let Some(deadline) = self.until.get(url) {
                if *deadline > now {
                    continue;
                }
                self.until.remove(url);
                self.failures.insert(url.clone(), 0);
            }
            available.push(url.clone());
        }
        if available.is_empty() {
            return String::new();
        }
        if available.contains(&self.current) {
            return self.current.clone();
        }
        self.current = available[0].clone();
        self.current.clone()
    }

    /// Records a failure and benches the source once it is exhausted. Returns
    /// whether the source was benched.
    pub fn fail(&mut self, url: &str, now: Instant) -> bool {
        let failures = self.failures.entry(url.to_string()).or_insert(0);
        *failures += 1;
        if *failures < SOURCE_FAILURES {
            return false;
        }
        self.until.insert(url.to_string(), now + SOURCE_BENCH);
        self.current.clear();
        true
    }

    pub fn success(&mut self, url: &str) {
        self.failures.insert(url.to_string(), 0);
    }
}

/// The reader thread's handoff and the consumer's outcome.
pub struct Consumption {
    pub outcome: Outcome,
    /// Frames this connection delivered, for the failure log.
    pub frames: u64,
    pub reader: std::thread::JoinHandle<()>,
}

/// Feeds frames to the tracker until the stream ends or fails. A stall, or a
/// premature EOF while a match is still live, is `Failed`, so the caller benches
/// that source instead of retrying it forever.
pub fn consume(
    cancel: &CancellationFlag,
    tracker: &mut Tracker,
    riot: &dyn RiotSource,
    frames: Box<dyn Read + Send>,
    second: &mut i64,
    started: Instant,
) -> Consumption {
    let (rx, reader) = frame_reader(frames);

    let mut counted = 0u64;
    let mut deadline = Instant::now() + STALL_TIMEOUT;
    let outcome = loop {
        if cancelled(cancel) {
            break Outcome::Ended;
        }
        if !riot.live_now() {
            break Outcome::Ended;
        }
        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(Ok(frame)) => {
                counted += 1;
                // After a stall the pipe holds old frames; skipping ahead keeps
                // seconds on the wall clock.
                let elapsed = started.elapsed().as_secs() as i64;
                if elapsed > *second {
                    *second = elapsed;
                }
                tracker.step(*second, &frame);
                *second += 1;
                deadline = Instant::now() + STALL_TIMEOUT;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if Instant::now() >= deadline {
                    break Outcome::Failed;
                }
            }
            Ok(Err(_)) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                if !cancelled(cancel) && riot.live_now() {
                    break Outcome::Failed;
                }
                break Outcome::Ended;
            }
        }
    };
    drop(rx);
    Consumption {
        outcome,
        frames: counted,
        reader,
    }
}

/// Feeds a local file through the same decode pipeline as the live stream.
pub fn replay(cancel: &CancellationFlag, tracker: &mut Tracker, path: &str) -> io::Result<()> {
    let ffmpeg = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            path,
            "-an",
            "-vf",
            DECODE_VF,
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let mut pipeline = Pipeline(vec![ffmpeg]);
    let stdout = pipeline.0[0]
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("ffmpeg stdout unavailable"))?;
    let (rx, reader) = frame_reader(Box::new(stdout));
    tracing::info!(file = path, "replay start");
    let mut second = 0i64;
    loop {
        if cancelled(cancel) {
            drop(rx);
            drop(pipeline);
            let _ = reader.join();
            return Ok(());
        }
        match rx.recv_timeout(Duration::from_millis(250)) {
            Ok(Ok(frame)) => {
                tracker.step(second, &frame);
                second += 1;
            }
            Ok(Err(error)) => {
                tracing::info!(frames = second, error = %error, "replay read ended");
                break;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    drop(rx);
    let _ = reader.join();
    let status = loop {
        if cancelled(cancel) {
            return Ok(());
        }
        if let Some(status) = pipeline.0[0].try_wait()? {
            break status;
        }
        std::thread::sleep(Duration::from_millis(250));
    };
    if !status.success() {
        return Err(io::Error::other(format!("ffmpeg exited with {status}")));
    }
    Ok(())
}

/// Opens the live stream while Riot lists a live match, rotating sources and
/// reconnecting when the stream ends or stalls. `lifecycle` receives the stream
/// lifecycle lines; they go only to the events-file ledger.
pub fn live(
    cancel: &CancellationFlag,
    tracker: &mut Tracker,
    riot: &dyn RiotSource,
    sources: &[String],
    lifecycle: &mut dyn FnMut(&str, &str, i64),
) {
    let started = Instant::now();
    let mut second = 0i64;
    let mut pool = SourcePool::new(sources);
    loop {
        if cancelled(cancel) {
            return;
        }
        if !riot.live_now() {
            if sleep_cancelled_blocking(cancel, IDLE_WAIT) {
                return;
            }
            continue;
        }
        let url = pool.choose(Instant::now());
        if url.is_empty() {
            // Every source is benched; wait for the shortest bench to expire.
            tracing::warn!("all stream sources benched");
            if sleep_cancelled_blocking(cancel, RECONNECT) {
                return;
            }
            continue;
        }
        tracing::info!(stream = %url, second, "connecting");
        lifecycle("stream_connect", &url, second);
        let (children, frames) = match open_stream(&url) {
            Ok(opened) => opened,
            Err(error) => {
                tracing::warn!(stream = %url, error = %error, "open stream failed");
                pool.fail(&url, Instant::now());
                // The attempt still has an outcome: without this line the ledger
                // shows connects that never end, including the ones that benched
                // their source.
                lifecycle("stream_failed", &url, second);
                continue;
            }
        };
        let mut pipeline = Pipeline(children);
        let consumption = consume(cancel, tracker, riot, frames, &mut second, started);
        // `stop_pipeline` kills both children, and a killed child's status says
        // nothing about why it died, so read their exits first. `None` means this
        // child was still running when the failure was noticed.
        let exits: [Option<std::process::ExitStatus>; 2] = if consumption.outcome == Outcome::Failed
        {
            let mut children = pipeline.0.iter_mut();
            [
                children
                    .next()
                    .and_then(|child| child.try_wait().ok().flatten()),
                children
                    .next()
                    .and_then(|child| child.try_wait().ok().flatten()),
            ]
        } else {
            [None, None]
        };
        stop_pipeline(&mut pipeline.0);
        let _ = consumption.reader.join();
        if cancelled(cancel) {
            return;
        }
        if consumption.outcome == Outcome::Failed {
            let benched = pool.fail(&url, Instant::now());
            lifecycle("stream_failed", &url, second);
            tracing::warn!(
                stream = %url,
                second,
                benched,
                frames = consumption.frames,
                streamlink_exit = ?exits[0],
                ffmpeg_exit = ?exits[1],
                "stream failed"
            );
        } else {
            pool.success(&url);
            tracing::warn!(stream = %url, second, "stream ended; reconnecting");
        }
        if sleep_cancelled_blocking(cancel, RECONNECT) {
            return;
        }
    }
}
