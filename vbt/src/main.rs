//! The `vbt` binary: CLI, wiring and signals. One process runs the stream decode,
//! the tracker state machine, Riot polling, the backend context and publishing.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use clap::Parser;

use vbt::cancel::CancellationFlag;
use vbt::context::{ContextSource, StaticMapOrder, watch_context};
use vbt::ledger::{self, Ledger};
use vbt::push::{Config, Pusher, SocketPublisher};
use vbt::riot::{LiveRiot, Poller, RiotState};
use vbt::stream;
use vbt::tracker::{Event, MapOrder, Publisher, Tracker};
use vbt_vision::learned::Crops;
use vbt_vision::recognizer::Recognizer;
use vbt_vision::templates::Templates;

/// Opens the live stream (or replays a file) at 1 fps, reads the score bar, and
/// publishes new scores to the backend.
#[derive(Parser, Debug)]
#[command(
    name = "vbt",
    version,
    about = "Reads Valorant broadcast round scores and publishes them to val-esports-backend"
)]
struct Args {
    /// Replay a video file instead of the live stream.
    #[arg(long)]
    file: Option<String>,
    /// Directory of `<digit>/*.png` templates (overrides the embedded templates).
    #[arg(long)]
    glyphs: Option<String>,
    /// Print one machine-readable line per tracked event.
    #[arg(long)]
    canonical: bool,
    /// Append every event, including rejects, as JSON lines.
    #[arg(long = "events-file")]
    events_file: Option<String>,
    /// Replay only: series map names, comma-separated, standing in for the veto.
    #[arg(long = "map-order")]
    map_order: Option<String>,
    /// Log at debug level.
    #[arg(long)]
    verbose: bool,
    /// Never write the token or PUT scores; emit visual audit JSONL.
    #[arg(long)]
    shadow: bool,
    /// Replay only: two competing team codes, comma-separated.
    #[arg(long)]
    teams: Option<String>,
    /// Directory of the team tag crops learned at runtime, created on first use.
    #[arg(long = "tag-crops", default_value = "data/tag-crops")]
    tag_crops: String,
    /// Stream URL to try; repeatable (defaults to Twitch then YouTube).
    #[arg(long = "source")]
    source: Vec<String>,
    /// Enables the in-process scoreboard with a context file.
    #[arg(long = "scoreboard-context")]
    scoreboard_context: Option<String>,
    /// Intra-op threads for the in-process model readers (the scoreboard table and
    /// the map strip's name and the team tags).
    #[arg(long = "scoreboard-threads", default_value_t = 1)]
    scoreboard_threads: usize,
    /// Test-only Riot base URL override.
    #[arg(long = "riot-base", hide = true)]
    riot_base: Option<String>,
}

/// A failure and the exit code it ends the process with: 2 for bad flags, 1 otherwise.
struct Exit {
    code: u8,
    message: String,
}

impl Exit {
    fn invalid(message: impl Into<String>) -> Exit {
        Exit {
            code: 2,
            message: message.into(),
        }
    }

    fn failed(message: impl Into<String>) -> Exit {
        Exit {
            code: 1,
            message: message.into(),
        }
    }
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    let level = if args.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };
    // Log this program at the requested level, and third-party crates at warn only.
    use tracing_subscriber::layer::SubscriberExt as _;
    use tracing_subscriber::util::SubscriberInitExt as _;
    let filter = tracing_subscriber::filter::Targets::new()
        .with_default(tracing::Level::WARN)
        .with_target("vbt", level);
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        // The fmt builder's default (INFO) gates events before `Targets` is
        // consulted, so the requested level must be set here too.
        .with_max_level(level)
        .finish()
        .with(filter)
        .init();
    match run(args) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(exit) => {
            tracing::error!("{}", exit.message);
            std::process::ExitCode::from(exit.code)
        }
    }
}

fn run(args: Args) -> Result<(), Exit> {
    let replay = args.file.is_some();

    if args.shadow && args.canonical {
        return Err(Exit::invalid("--shadow cannot mix with --canonical"));
    }
    if let Some(teams) = &args.teams
        && (!replay || teams.split(',').count() != 2)
    {
        return Err(Exit::invalid(
            "--teams requires a replay and exactly two codes",
        ));
    }
    if !replay && args.map_order.is_some() {
        return Err(Exit::invalid(
            "--map-order is replay-only; a live run takes map names from the video context",
        ));
    }

    // Embedded by default, so a deployed binary needs no glyph directory.
    let templates = match &args.glyphs {
        Some(dir) => Templates::load_dir(Path::new(dir))
            .map_err(|e| Exit::failed(format!("load glyph templates from {dir}: {e}")))?,
        None => Templates::load_embedded()
            .map_err(|e| Exit::failed(format!("load embedded glyph templates: {e}")))?,
    };
    let recognizer = vbt::tracker_recognizer(
        Recognizer::new().map_err(|e| Exit::failed(format!("load recognizer: {e}")))?,
        args.scoreboard_threads,
    );

    let ledger = match &args.events_file {
        Some(path) => {
            Some(Ledger::open(path).map_err(|e| Exit::failed(format!("open events file: {e}")))?)
        }
        None => None,
    };

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|e| Exit::failed(format!("tokio runtime: {e}")))?;
    let cancel: CancellationFlag = Arc::new(AtomicBool::new(false));

    let mut config = Config::from_env();
    config.read_only = args.shadow || replay;
    let mut pusher = Pusher::new(config.clone()).map_err(|e| Exit::failed(e.to_string()))?;
    // Only a live run owns the token file: a replay writing a new one would
    // invalidate the token a running tracker is already using.
    if !replay && config.enabled() {
        pusher
            .write_token()
            .map_err(|e| Exit::failed(format!("write video token: {e}")))?;
    }
    let pusher = Arc::new(pusher);
    let publisher: Option<Box<dyn Publisher + Send>> = if config.enabled() {
        Some(Box::new(SocketPublisher::new(
            pusher.clone(),
            runtime.handle().clone(),
        )))
    } else {
        None
    };

    let riot_state = Arc::new(Mutex::new(RiotState::default()));
    if !replay {
        let base = args.riot_base.as_deref().unwrap_or(vbt::riot::API);
        let poller = Poller::with_base_url(base, riot_state.clone())
            .map_err(|e| Exit::failed(e.to_string()))?;
        runtime.spawn(poller.run(cancel.clone()));
    }
    let live_riot = LiveRiot::from_state(riot_state);

    // A replay has no backend to ask for the veto, so an explicit map order stands
    // in for it and exercises the veto naming path offline. Live runs instead use
    // the context source.
    let mut context_source: Option<Arc<ContextSource>> = None;
    let map_order: Option<Box<dyn MapOrder + Send>> = if replay {
        args.map_order.as_ref().map(|order| {
            let names = order.split(',').map(|name| name.to_string()).collect();
            Box::new(StaticMapOrder::new(names)) as Box<dyn MapOrder + Send>
        })
    } else if config.context_enabled() {
        let source = Arc::new(ContextSource::new(pusher.clone()));
        context_source = Some(source.clone());
        Some(Box::new(source) as Box<dyn MapOrder + Send>)
    } else {
        None
    };
    if let Some(source) = &context_source {
        runtime.spawn(watch_context(
            cancel.clone(),
            source.clone(),
            live_riot.clone(),
        ));
    }

    let mut tracker = Tracker::new(
        templates,
        Some(recognizer),
        Box::new(live_riot.clone()),
        publisher,
        map_order,
    );
    tracker.shadow = args.shadow;
    // Only the OCR teaches the crops, so a build without it reads no team tags and never
    // joins a map in progress.
    if cfg!(feature = "scoreboard") {
        tracker.plates =
            Some(Crops::open(Some(Path::new(&args.tag_crops))).map_err(|e| {
                Exit::failed(format!("load tag crops from {}: {e}", args.tag_crops))
            })?);
        tracker.tag_reader = vbt::tag_reader(args.scoreboard_threads);
    } else {
        tracing::warn!("built without the scoreboard feature: no team tags, no mid-map joins");
    }
    if let Some(teams) = &args.teams {
        tracker.expected_teams = teams.split(',').map(|code| code.to_string()).collect();
    }

    #[cfg(feature = "scoreboard")]
    if let Some(context) = &args.scoreboard_context {
        let adapter = vbt::scoreboard::InProcess::new(Path::new(context), args.scoreboard_threads)
            .map_err(Exit::failed)?;
        tracker.scoreboard = Some(Box::new(adapter));
        tracing::info!(
            context = %context,
            threads = args.scoreboard_threads,
            "scoreboard enabled"
        );
    }
    #[cfg(not(feature = "scoreboard"))]
    if args.scoreboard_context.is_some() {
        return Err(Exit::invalid(
            "--scoreboard-context was given, but this build has no scoreboard feature",
        ));
    }

    let event_sink: Option<Arc<Mutex<Ledger>>> = ledger.map(|ledger| Arc::new(Mutex::new(ledger)));
    let sink_for_events = event_sink.clone();
    let canonical = args.canonical;
    let shadow = args.shadow;
    tracker.on_event = Some(Box::new(move |event: &Event| {
        if canonical {
            let stdout = std::io::stdout();
            let mut out = stdout.lock();
            ledger::print_canonical(&mut out, event);
        }
        match &sink_for_events {
            Some(sink) => {
                if let Ok(mut sink) = sink.lock() {
                    sink.record(event);
                }
            }
            None if shadow => {
                let stdout = std::io::stdout();
                let mut out = stdout.lock();
                let _ = ledger::write_line(&mut out, &ledger::event_record(event));
            }
            None => {}
        }
    }));

    // SIGINT/SIGTERM set the shared flag; every loop notices it within a slice.
    let signal_cancel = cancel.clone();
    runtime.spawn(async move {
        use tokio::signal::unix::{SignalKind, signal};
        let Ok(mut interrupt) = signal(SignalKind::interrupt()) else {
            return;
        };
        let Ok(mut terminate) = signal(SignalKind::terminate()) else {
            return;
        };
        tokio::select! {
            _ = interrupt.recv() => {}
            _ = terminate.recv() => {}
        }
        signal_cancel.store(true, Ordering::SeqCst);
    });

    let sources: Vec<String> = if args.source.is_empty() {
        stream::DEFAULT_SOURCES
            .iter()
            .map(|url| url.to_string())
            .collect()
    } else {
        args.source.clone()
    };
    let file = args.file.clone();
    let lifecycle_sink = event_sink.clone();
    let frame_cancel = cancel.clone();
    let handle = std::thread::Builder::new()
        .name("frames".to_string())
        .spawn(move || -> Result<(), String> {
            if let Some(path) = file {
                stream::replay(&frame_cancel, &mut tracker, &path)
                    .map_err(|e| format!("replay failed: {e}"))
            } else {
                let mut lifecycle = move |kind: &str, url: &str, second: i64| {
                    if let Some(sink) = &lifecycle_sink
                        && let Ok(mut sink) = sink.lock()
                    {
                        sink.lifecycle(kind, url, second);
                    }
                };
                stream::live(
                    &frame_cancel,
                    &mut tracker,
                    &live_riot,
                    &sources,
                    &mut lifecycle,
                );
                Ok(())
            }
        })
        .map_err(|e| Exit::failed(format!("spawn frame thread: {e}")))?;

    match handle.join() {
        Ok(Ok(())) => {}
        Ok(Err(message)) => return Err(Exit::failed(message)),
        Err(_) => return Err(Exit::failed("frame thread panicked")),
    }
    cancel.store(true, Ordering::SeqCst);
    Ok(())
}
