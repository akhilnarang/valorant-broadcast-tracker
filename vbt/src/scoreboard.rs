//! The in-process scoreboard adapter: owns a `vbt_scoreboard::Reader` and the
//! context file, and maps the reader's scoreboard into the tracker's ledger shape.
//!
//! With the `scoreboard` feature on, this module also wires the model-backed map-strip
//! name reader ([`StripRecognizer`]) in front of the template recognizer, and the team
//! tag OCR.

use std::path::Path;

use vbt_scoreboard::reader::Reader;
use vbt_scoreboard::schema::Context;
use vbt_vision::frame::Frame;
use vbt_vision::recognizer::{PauseRead, Recognizer};

use crate::tracker::{ScoreboardObservation, ScoreboardPlayer, ScoreboardReader, ScoreboardTeam};

/// The recognizer the tracker runs with the model available: the template recognizer,
/// with the strip name read answered by the CTC lexicon reader instead of by the
/// per-map median templates.
pub struct StripRecognizer {
    vision: Recognizer,
    strip: vbt_scoreboard::strip::Reader,
}

/// Chooses the tracker's strip name reader: the CTC lexicon reader when its model
/// session loads, else the template recognizer alone. Kept separate from the session
/// call so the fallback is testable without failing ONNX Runtime in-process.
fn select_strip(
    vision: Recognizer,
    strip: Result<vbt_scoreboard::strip::Reader, String>,
) -> Box<dyn crate::tracker::Recognizer + Send> {
    match strip {
        Ok(strip) => Box::new(StripRecognizer { vision, strip }),
        Err(error) => {
            tracing::warn!(
                error = %error,
                "strip model unavailable; using the template strip reader alone"
            );
            Box::new(crate::tracker::VisionRecognizer(vision))
        }
    }
}

/// The strip reader the tracker runs: the CTC lexicon reader with the per-map templates
/// as its fallback. A model or ORT failure logs a warning and falls back to the template
/// reader alone (with its `0.75` anchor floor, as the no-`scoreboard` build uses) rather
/// than exiting: a library problem on the broadcast host must degrade the strip read,
/// not take the tracker down.
pub fn strip_recognizer(
    vision: Recognizer,
    threads: usize,
) -> Box<dyn crate::tracker::Recognizer + Send> {
    let strip = vbt_scoreboard::strip::Reader::new(threads).map_err(|e| e.to_string());
    select_strip(vision, strip)
}

impl crate::tracker::Recognizer for StripRecognizer {
    fn score(&self, crop: &Frame) -> Option<i32> {
        self.vision.score(crop)
    }

    fn map(&self, strip: &Frame) -> String {
        self.strip.map(&self.vision, strip)
    }

    fn pause(&self, header: &Frame, card: &Frame) -> (PauseRead, bool) {
        self.vision.pause(header, card)
    }
}

/// The team tag OCR. A model or ORT failure logs a warning and leaves the tags to the
/// crops already learned, as the strip falls back to its templates.
pub fn tag_reader(threads: usize) -> Option<Box<dyn crate::tracker::TagReader + Send>> {
    match vbt_scoreboard::tags::Reader::new(threads) {
        Ok(reader) => Some(Box::new(reader)),
        Err(error) => {
            tracing::warn!(
                error = %error,
                "tag model unavailable; reading team tags from learned crops alone"
            );
            None
        }
    }
}

impl crate::tracker::TagReader for vbt_scoreboard::tags::Reader {
    fn read(&self, frame: &Frame, codes: [&str; 2]) -> [String; 2] {
        self.read(frame, codes).map(|read| read.code)
    }
}

/// The adapter over the native scoreboard reader, behind the `scoreboard` feature.
pub struct InProcess {
    reader: Reader,
    context: Context,
}

impl InProcess {
    /// Loads the context file and the embedded recognition model.
    pub fn new(context_path: &Path, threads: usize) -> Result<InProcess, String> {
        let data = std::fs::read(context_path)
            .map_err(|e| format!("scoreboard context {}: {e}", context_path.display()))?;
        let context: Context = serde_json::from_slice(&data)
            .map_err(|e| format!("scoreboard context {}: {e}", context_path.display()))?;
        let reader = Reader::new(threads).map_err(|e| e.to_string())?;
        Ok(InProcess { reader, context })
    }
}

impl ScoreboardReader for InProcess {
    fn analyze(&mut self, frame: &Frame) -> Result<Option<ScoreboardObservation>, String> {
        match self.reader.analyze(frame, &self.context) {
            Ok(None) => Ok(None),
            Ok(Some(table)) => Ok(Some(ScoreboardObservation {
                round_displayed: table.round,
                // No detector yet: the native reader only ever reports the broadcast table.
                layout: "broadcast_table".to_string(),
                teams: table
                    .teams
                    .into_iter()
                    .map(|team| ScoreboardTeam {
                        name: team.name,
                        score: team.score,
                    })
                    .collect(),
                players: table
                    .players
                    .into_iter()
                    .map(|player| ScoreboardPlayer {
                        team: player.team,
                        name: player.name,
                        kda: player.kda,
                    })
                    .collect(),
            })),
            Err(error) => Err(error.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vbt_vision::regions::MAP_STRIP;

    include!("../../testdata/fixture.rs");

    /// A model or ORT session failure must select the template reader - the reader the
    /// no-`scoreboard` build uses - instead of exiting. Under the simulated error the
    /// model is never available, and the templates still name `haven-0-0.jpg` Haven
    /// (the Haven template scores 0.99), so a named read proves the fallback is live.
    #[test]
    fn session_failure_falls_back_to_the_template_reader() {
        let vision = Recognizer::new().expect("recognizer");
        let recognizer = select_strip(vision, Err("simulated session failure".into()));
        let bytes = std::fs::read(fixture("testdata/frames/haven-0-0.jpg")).expect("fixture");
        let image = image::load_from_memory(&bytes).expect("decode");
        let strip: Frame = Frame::from_rgb_image_owned(image.to_rgb8()).crop_raw(MAP_STRIP);
        assert_eq!(recognizer.map(&strip), "Haven");
    }
}
