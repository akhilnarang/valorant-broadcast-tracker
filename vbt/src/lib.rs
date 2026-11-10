//! Reads Valorant broadcast score bars and publishes map scores.

pub mod cancel;
pub mod context;
pub mod ledger;
pub mod push;
pub mod riot;
pub mod stream;
pub mod tracker;

#[cfg(feature = "scoreboard")]
pub mod scoreboard;

/// The tracker's recognizer. With the `scoreboard` feature the model reads the map
/// strip and the templates are its fallback; without it the templates read alone.
pub fn tracker_recognizer(
    vision: vbt_vision::recognizer::Recognizer,
    threads: usize,
) -> Box<dyn tracker::Recognizer + Send> {
    #[cfg(feature = "scoreboard")]
    return scoreboard::strip_recognizer(vision, threads);
    #[cfg(not(feature = "scoreboard"))]
    {
        let _ = threads;
        Box::new(tracker::VisionRecognizer(vision))
    }
}

/// The OCR that reads and teaches the team tags, or `None` when its model does not load.
/// Without the `scoreboard` feature there is no OCR, so nothing teaches the learned crops.
pub fn tag_reader(threads: usize) -> Option<Box<dyn tracker::TagReader + Send>> {
    #[cfg(feature = "scoreboard")]
    return scoreboard::tag_reader(threads);
    #[cfg(not(feature = "scoreboard"))]
    {
        let _ = threads;
        None
    }
}
