//! Native scoreboard reader for the broadcast buy-phase table.
//!
//! One tiny PP-OCRv6 recognition model plus lexicon-constrained CTC scoring. There is no
//! detector and no free-text OCR: every cell holds a string from a known set (roster names,
//! `0`..`99`, `ROUND 1`..`ROUND 50`, the two match tags), so each candidate's CTC
//! log-likelihood is scored and accepted only past a margin over the runner-up. Team scores
//! come from the logistic-regression plate reader in `vbt-vision`, not the model.
//!
//! See `docs/scoreboard-native.md` in the repository root for the measurements and rules.

pub mod ctc;
pub mod geometry;
pub mod model;
pub mod reader;
pub mod schema;
pub mod strip;
pub mod tags;
mod tensor;

/// The recognition model, embedded so the binary carries exactly one copy.
///
/// `PP-OCRv6_rec_pruned.onnx` is RapidOCR's `PP-OCRv6_rec_tiny.onnx` with its
/// classification head sliced to the 128 classes a scoreboard cell can hold
/// (blank, printable ASCII, the Latin-1 letters, space); see
/// `tools/prune_head.py`. Its `character` metadata lists exactly that alphabet,
/// so [`reader`]'s class list and the model stay in step. `build.rs` fetches it.
pub const MODEL_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/PP-OCRv6_rec_pruned.onnx"));

/// Library error type. The CLI boundary converts these to `Box<dyn Error>`.
#[derive(Debug)]
pub enum Error {
    /// An ONNX Runtime failure (session, tensor, or inference).
    Ort(String),
    /// An image decode failure.
    Image(image::ImageError),
    /// An I/O failure on the frame or context streams.
    Io(std::io::Error),
    /// A context or JSON serialization failure.
    Json(serde_json::Error),
    /// The model or its metadata is unusable.
    Model(String),
    /// A crop or batch cannot be built.
    Tensor(String),
    /// A frame is unusable.
    Frame(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Ort(e) => write!(f, "vbt-scoreboard: onnxruntime: {e}"),
            Error::Image(e) => write!(f, "vbt-scoreboard: image: {e}"),
            Error::Io(e) => write!(f, "vbt-scoreboard: io: {e}"),
            Error::Json(e) => write!(f, "vbt-scoreboard: json: {e}"),
            Error::Model(m) => write!(f, "vbt-scoreboard: model: {m}"),
            Error::Tensor(m) => write!(f, "vbt-scoreboard: tensor: {m}"),
            Error::Frame(m) => write!(f, "vbt-scoreboard: frame: {m}"),
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
from_error!(std::io::Error, Io);
from_error!(serde_json::Error, Json);

impl From<vbt_vision::Error> for Error {
    fn from(value: vbt_vision::Error) -> Self {
        match value {
            vbt_vision::Error::Image(e) => Error::Image(e),
            vbt_vision::Error::Json(e) => Error::Json(e),
            vbt_vision::Error::Io(e) => Error::Io(e),
            vbt_vision::Error::Frame(m) => Error::Frame(m),
            other => Error::Frame(other.to_string()),
        }
    }
}

impl<R> From<ort::Error<R>> for Error {
    fn from(value: ort::Error<R>) -> Self {
        Error::Ort(value.to_string())
    }
}

/// Debug logging, enabled with `VBT_SCOREBOARD_DEBUG=1`; always to stderr.
pub(crate) fn debug_enabled() -> bool {
    std::env::var_os("VBT_SCOREBOARD_DEBUG").is_some_and(|v| v != "0")
}

macro_rules! dlog {
    ($($arg:tt)*) => {
        if $crate::debug_enabled() {
            eprintln!("vbt-scoreboard: {}", format_args!($($arg)*));
        }
    };
}
pub(crate) use dlog;
