//! Pixel code shared by the tracker and the scoreboard reader: frames, region crops,
//! the resampler, the score-bar template readers, the map-strip and pause recognizer,
//! and the learned team tag crops.

pub mod frame;
pub mod learned;
pub mod plates;
pub mod recognizer;
pub mod regions;
pub mod resample;
pub mod scene;
pub mod templates;

/// The score-plate logistic regression.
pub const DIGITS_LR_BYTES: &[u8] = include_bytes!("../../assets/recognizer/digits_lr.bin");

/// The map-strip word templates (`current` plus one per map).
pub const STRIP_TEMPLATES_BYTES: &[u8] =
    include_bytes!("../../assets/recognizer/strip_templates.json");

/// The pause overlay phrase templates.
pub const PAUSE_TEMPLATES_BYTES: &[u8] =
    include_bytes!("../../assets/recognizer/pause_templates.json");

include!(concat!(env!("OUT_DIR"), "/assets.rs"));

/// Library error type. The binary boundary converts these to `Box<dyn Error>`.
#[derive(Debug)]
pub enum Error {
    /// An embedded asset or model binary is unusable.
    Model(String),
    /// A frame buffer is unusable.
    Frame(String),
    /// An asset directory or file is unusable.
    Assets(String),
    /// An image decode failure.
    Image(image::ImageError),
    /// A JSON asset failure.
    Json(serde_json::Error),
    /// An I/O failure.
    Io(std::io::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Model(m) => write!(f, "vbt-vision: model: {m}"),
            Error::Frame(m) => write!(f, "vbt-vision: frame: {m}"),
            Error::Assets(m) => write!(f, "vbt-vision: assets: {m}"),
            Error::Image(e) => write!(f, "vbt-vision: image: {e}"),
            Error::Json(e) => write!(f, "vbt-vision: json: {e}"),
            Error::Io(e) => write!(f, "vbt-vision: io: {e}"),
        }
    }
}

impl std::error::Error for Error {}

macro_rules! from_error {
    ($from:ty, $variant:ident) => {
        impl From<$from> for Error {
            fn from(value: $from) -> Self {
                Error::$variant(value)
            }
        }
    };
}

from_error!(image::ImageError, Image);
from_error!(serde_json::Error, Json);
from_error!(std::io::Error, Io);
