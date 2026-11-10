//! The cancellation flag shared by the signal watcher, the Riot poller, the context
//! watcher and the stream loop.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Set by the signal watcher (SIGINT/SIGTERM) and read by every loop.
pub type CancellationFlag = Arc<AtomicBool>;

/// The longest single wait in a sleep, so a shutdown never waits out a poll.
const SLICE: Duration = Duration::from_millis(250);

/// Whether cancellation was requested.
pub fn cancelled(flag: &AtomicBool) -> bool {
    flag.load(Ordering::SeqCst)
}

/// The next slice to wait before `deadline`, or `None` once it has passed.
fn next_slice(deadline: Instant) -> Option<Duration> {
    let now = Instant::now();
    (now < deadline).then(|| (deadline - now).min(SLICE))
}

/// Sleeps up to `duration`, returning `true` when cancellation was noticed first.
pub async fn sleep_cancelled(flag: &AtomicBool, duration: Duration) -> bool {
    let deadline = Instant::now() + duration;
    while !cancelled(flag) {
        let Some(slice) = next_slice(deadline) else {
            return false;
        };
        tokio::time::sleep(slice).await;
    }
    true
}

/// The blocking twin of `sleep_cancelled`, for the frame thread.
pub fn sleep_cancelled_blocking(flag: &AtomicBool, duration: Duration) -> bool {
    let deadline = Instant::now() + duration;
    while !cancelled(flag) {
        let Some(slice) = next_slice(deadline) else {
            return false;
        };
        std::thread::sleep(slice);
    }
    true
}
