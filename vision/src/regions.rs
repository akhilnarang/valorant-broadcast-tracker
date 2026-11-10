//! The hand-measured broadcast boxes, at 1920x1080.

pub use crate::frame::{REFERENCE_WIDTH, Rect};

/// Canvas size the digit classifier was trained on.
pub const GLYPH_WIDTH: usize = 24;
pub const GLYPH_HEIGHT: usize = 32;

/// The production crop starts at x=0 and ends at this 720p coordinate.
pub const MAP_STRIP_WIDTH: i64 = 435;

/// The match timer, at 1920x1080.
pub const TIMER: Rect = Rect::new(895, 24, 1002, 64);
/// The broadcast's left score plate.
pub const SCORE_LEFT: Rect = Rect::new(784, 4, 858, 56);
/// The broadcast's right score plate.
pub const SCORE_RIGHT: Rect = Rect::new(1042, 2, 1118, 56);
/// The replay banner that marks a replay scene.
pub const REPLAY_BANNER: Rect = Rect::new(1612, 996, 1852, 1044);

/// The pause overlay's lower-third header phrase band, measured on the 720p broadcast
/// and stored at the 1080p scale the others use (the 720p box times 1.5).
pub const PAUSE_HEADER: Rect = Rect::new(218, 818, 623, 881);

/// The full-screen break card's title band, at the 1080p scale. Broadcasts place the
/// card's title at different heights (152-167 and 165-180 at 720p have both been seen),
/// so the box is a band the 25-row template slides through.
pub const PAUSE_CARD: Rect = Rect::new(75, 216, 300, 288);

/// The left team tag plate, at 1920x1080.
pub const TEAM_LEFT: Rect = Rect::new(730, 8, 780, 36);
/// The right team tag plate, at 1920x1080.
pub const TEAM_RIGHT: Rect = Rect::new(1130, 8, 1200, 36);

/// "CURRENT: <MAP>" at 720p, in absolute pixels.
pub const MAP_STRIP: Rect = Rect::new(0, 0, MAP_STRIP_WIDTH, 18);
