//! The `CURRENT: <MAP>` strip name reader: lexicon-constrained CTC over the name window.
//!
//! The template matcher in `vbt_vision::recognizer` is a pixel-correlation fallback. Its
//! margin over the runner-up can collapse when the broadcast's rendering changes (see the
//! Summit/Sunset case in `docs/scoreboard-native.md`). This reader scores all thirteen map
//! names as CTC label sequences over the model's per-timestep softmax and accepts the best
//! only past a likelihood margin, so it needs no per-map asset.
//!
//! A lexicon score alone cannot tell a map name from arbitrary text: with only known names
//! as competitors, unknown text still makes its closest name win by the same wide margin
//! (a window holding `SPLINT` reads Split at 44.6 nats). The window is therefore also
//! decoded freely, and a read is only kept when that decode supports the winning name:
//! the same word, or a likelihood within [`FREE_TOLERANCE`] nats of the name's own, so
//! noise on the name's glyphs passes while a different word is refused.
//!
//! Geometry: the vision recognizer locates `CURRENT:`, and the name window starts
//! [`vbt_vision::recognizer::NAME_GAP`] px after it, 12 rows tall, [`NAME_WINDOW`] wide.

use std::collections::HashMap;
use std::sync::Mutex;

use ort::session::Session;

use vbt_vision::frame::{Frame, Rect};
use vbt_vision::recognizer::{Anchor, Recognizer};

use crate::ctc;
use crate::tensor::{Crop, timesteps};
use crate::{Error, dlog};

/// The thirteen map names: the strip's upper-case rendering and the display name the
/// tracker uses.
pub const MAP_NAMES: [(&str, &str); 13] = [
    ("ABYSS", "Abyss"),
    ("ASCENT", "Ascent"),
    ("BIND", "Bind"),
    ("BREEZE", "Breeze"),
    ("CORRODE", "Corrode"),
    ("FRACTURE", "Fracture"),
    ("HAVEN", "Haven"),
    ("ICEBOX", "Icebox"),
    ("LOTUS", "Lotus"),
    ("PEARL", "Pearl"),
    ("SPLIT", "Split"),
    ("SUMMIT", "Summit"),
    ("SUNSET", "Sunset"),
];

/// The name window's width in 720p pixels: `FRACTURE` is the longest name at about 52 px.
pub const NAME_WINDOW: i64 = 64;
/// The rows the name occupies, from the anchor's first row.
pub const NAME_ROWS: i64 = 12;

/// The CTC margin (nats) the best name needs over the runner-up.
///
/// Measured over 4152 real frames: every accepted read scored at least 21.0 nats and every
/// refused candidate at most 6.4, so this floor sits in an empty band. The margin's job is
/// to refuse the short names that win a blurry or text-free window (`BIND` is the usual
/// runner-up).
pub const MAP_MARGIN: f64 = 16.0;
/// The CTC score per character (nats) below which a read is refused: a longer name must
/// not lose to a short one just because it has more terms to explain.
pub const MAP_PER_CHAR: f64 = -3.0;
/// The `CURRENT:` correlation below which the name read cannot carry the strip alone.
///
/// The template reader refuses below [`vbt_vision::recognizer::ANCHOR_THRESHOLD`] (0.75),
/// which costs the dimmer 720p renders a read even when the name is plainly legible. Such
/// strips anchor at about 0.70 and their names score 20-60 nats. The anchor false positives
/// in 0.60-0.75 are all text-free, and their best candidate tops out at 6.4 nats. So a
/// decisive name is enough at this floor. Below it, the anchor is not trusted at all.
pub const ANCHOR_LOW: f64 = 0.60;
/// The blank run (timesteps) that ends the name in the free decode.
///
/// Every strip renders a thumbnail icon right after the name; the free decode is cut at a
/// gap this long so the icon (or the strip's next panel, which the window can reach for a
/// short name) does not join the word. Letter pairs inside a name never gap by five
/// timesteps on the corpus, while the icon either follows a space or sits several
/// timesteps away.
pub const FREE_GAP: usize = 5;
/// The nats within which the free decode may differ from the winning name's likelihood.
///
/// A real rendering's free decode can misread one of the name's own glyphs (`ABYS5` for
/// Abyss, `LOTU5` for Lotus) or attach the first icon glyph; on the 4,152-frame corpus
/// such noise stays under 5.0 nats except one `ABYS5` window at 6.12, and an attached
/// `NRG` team code on an Icebox strip reaches 14.9. Unknown text is a different word:
/// `CORRODED` beats Corrode by 6.57 and `SPLINT` beats Split by 6.62, up to 17.9. The
/// floor sits in that band: it refuses the one corpus read and the team-code windows
/// along with all 180 synthetic unknown renders.
pub const FREE_TOLERANCE: f64 = 5.0;

/// One strip name read: the winning candidate and the evidence for and against it.
#[derive(Debug, Clone, PartialEq)]
pub struct MapRead {
    /// The display name, e.g. `Summit`.
    pub name: String,
    /// The `CURRENT:` anchor correlation that placed the window.
    pub anchor: f64,
    /// CTC log-likelihood of the winning name.
    pub score: f64,
    /// Its margin over the runner-up, in nats.
    pub margin: f64,
    /// `score / characters`.
    pub per_char: f64,
    /// The window's free decode: its first word, as the model emitted it, empty when the
    /// window holds no text. Cut at a space or [`FREE_GAP`] blanks.
    pub free: String,
    /// `score(free) - score`: how much better the free word explains the window.
    pub free_delta: f64,
    /// Whether the free word supports the winning name; see [`word_matches`].
    pub known: bool,
    /// Whether the name clears every acceptance rule.
    pub accepted: bool,
}

/// The letters of `text`: upper-cased, with everything that is not a letter or digit
/// dropped, so case and the renderer's spacing do not matter.
fn letters(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_uppercase)
        .collect()
}

/// Whether the free word `word`, scored `word_score`, supports the winning `name`, scored
/// `name_score`: the same letters (so `SPLINT` cannot pass as `SPLIT`), or a likelihood
/// within [`FREE_TOLERANCE`] of the name's, so a misread name glyph or an attached icon
/// letter is tolerated while a word the model reads more than the tolerance better is
/// treated as unknown text.
fn word_matches(word: &str, word_score: f64, name: &str, name_score: f64) -> bool {
    letters(word) == letters(name) || (word_score - name_score).abs() <= FREE_TOLERANCE
}

/// The strip name reader: one recognition session and the map-name lexicon.
pub struct Reader {
    session: Mutex<Session>,
    classes: Vec<char>,
    index: HashMap<char, usize>,
}

impl Reader {
    /// Loads the embedded model with `threads` intra-op threads.
    pub fn new(threads: usize) -> Result<Reader, Error> {
        let mut session = crate::model::session(threads)?;
        let (classes, index) = crate::model::classes(&mut session)?;
        Ok(Reader {
            session: Mutex::new(session),
            classes,
            index,
        })
    }

    /// The strip's map name: the model read when it is decisive and the window's free
    /// decode supports that name, else the template matcher's name when it also agrees,
    /// else `""`.
    ///
    /// The agreement rule keeps the templates useful where the model is weakest (the
    /// older, blurrier Summit renderings score 5.9 nats, under the floor, while the
    /// templates still match them at 0.99). A template can never name a map the model
    /// did not rank first. Both paths require the free decode to support the model's
    /// name, so text that is not a map is refused instead of being matched to its
    /// closest name.
    pub fn map(&self, vision: &Recognizer, strip: &Frame) -> String {
        let Some(read) = self.read(vision, strip) else {
            return String::new();
        };
        if read.accepted {
            return read.name;
        }
        if !read.known {
            return String::new();
        }
        let template = vision.map(strip);
        if !template.is_empty() && template.eq_ignore_ascii_case(&read.name) {
            return template;
        }
        String::new()
    }

    /// The name window's CTC read, whether or not it clears the acceptance rules.
    /// `None` when `CURRENT:` is not located well enough to place the window.
    pub fn read(&self, vision: &Recognizer, strip: &Frame) -> Option<MapRead> {
        let anchor = vision.anchor(strip);
        if anchor.score < ANCHOR_LOW {
            return None;
        }
        let probs = self.window_probs(vision, strip, &anchor)?;
        let candidates: Vec<String> = MAP_NAMES
            .iter()
            .map(|(upper, _)| upper.to_string())
            .collect();
        let best = ctc::best(
            &probs.0,
            self.classes.len(),
            probs.1,
            &self.index,
            &candidates,
            true,
        )?;
        let free = self.free_word(&probs.0, probs.1);
        let free_score = ctc::extension(&free, &self.index)
            .map(|ext| ctc::score(&probs.0, self.classes.len(), probs.1, &ext))
            .unwrap_or(f64::NEG_INFINITY);
        let (upper, display) = MAP_NAMES[best.index];
        let per_char = best.score / upper.chars().count() as f64;
        let known = word_matches(&free, free_score, upper, best.score);
        Some(MapRead {
            name: display.to_string(),
            anchor: anchor.score,
            score: best.score,
            margin: best.margin,
            per_char,
            free,
            free_delta: free_score - best.score,
            known,
            accepted: known && best.margin >= MAP_MARGIN && per_char >= MAP_PER_CHAR,
        })
    }

    /// The window's free CTC decode: the first word of the argmax path, repeats collapsed
    /// and blanks dropped. This is what the model thinks the text is, with no lexicon to
    /// lean on, so it can say `SPLINT` where the lexicon scoring can only offer `SPLIT`.
    /// The word ends at a space or after [`FREE_GAP`] blank timesteps.
    fn free_word(&self, probs: &[f32], t_used: usize) -> String {
        let mut word = String::new();
        let mut previous = usize::MAX;
        let mut blanks = 0usize;
        for t in 0..t_used {
            let row = &probs[t * self.classes.len()..][..self.classes.len()];
            let class = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index)
                .unwrap_or(0);
            if class == 0 {
                blanks += 1;
                continue;
            }
            if self.classes[class] == ' ' {
                return word;
            }
            if class == previous && blanks == 0 {
                continue;
            }
            if !word.is_empty() && blanks >= FREE_GAP {
                return word;
            }
            word.push(self.classes[class]);
            previous = class;
            blanks = 0;
        }
        word
    }

    /// Runs the model on the name window: the softmax timesteps and how many to use.
    fn window_probs(
        &self,
        vision: &Recognizer,
        strip: &Frame,
        anchor: &Anchor,
    ) -> Option<(Vec<f32>, usize)> {
        let x0 = vision.name_x(anchor.x);
        let window = strip.crop_raw(Rect::new(
            x0,
            anchor.y,
            x0 + NAME_WINDOW,
            anchor.y + NAME_ROWS,
        ));
        let crop = match Crop::prepare(&window) {
            Ok(crop) => crop,
            Err(error) => {
                dlog!("strip: name window is unusable: {error}");
                return None;
            }
        };
        let mut session = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match crate::model::run(&mut session, 1, crop.width, &crop.data) {
            Ok((probs, t, _)) => Some((probs, t.min(timesteps(crop.width)))),
            Err(error) => {
                dlog!("strip: name window inference failed: {error}");
                None
            }
        }
    }
}
