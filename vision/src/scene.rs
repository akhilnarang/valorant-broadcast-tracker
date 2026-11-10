//! Scene classification and the score-bar visibility gate.

use std::fmt;

use crate::frame::Frame;
use crate::regions;

/// What the broadcast shows, judged by its overlays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scene {
    Gameplay,
    Replay,
    NoHud,
}

impl Scene {
    /// The lowercase name the ledger carries.
    pub fn as_str(self) -> &'static str {
        match self {
            Scene::Gameplay => "gameplay",
            Scene::Replay => "replay",
            Scene::NoHud => "no_hud",
        }
    }
}

impl fmt::Display for Scene {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The share of pixels brighter than `threshold` in the frame's luma.
pub fn bright_fraction(f: &Frame, threshold: i64) -> f64 {
    let gray = f.gray();
    let count = gray.iter().filter(|v| **v as i64 > threshold).count();
    count as f64 / gray.len() as f64
}

/// The share of strongly red pixels, like the planted-spike icon.
pub fn red_fraction(f: &Frame) -> f64 {
    let mut count = 0usize;
    for i in 0..f.w * f.h {
        let r = f.pix[3 * i] as f64;
        let g = f.pix[3 * i + 1] as f64;
        let b = f.pix[3 * i + 2] as f64;
        if r > 130.0 && r > 1.8 * g && r > 1.8 * b {
            count += 1;
        }
    }
    count as f64 / (f.w * f.h) as f64
}

/// Whether either score plate shows bright digits.
pub fn score_visible(f: &Frame) -> bool {
    let left = bright_fraction(&f.crop(regions::SCORE_LEFT), 195);
    let right = bright_fraction(&f.crop(regions::SCORE_RIGHT), 195);
    left.max(right) > 0.05
}

/// Classifies a frame by the broadcast's overlays. A replay banner wins over a visible HUD.
pub fn scene_of(f: &Frame) -> Scene {
    if bright_fraction(&f.crop(regions::REPLAY_BANNER), 200) > 0.25 {
        return Scene::Replay;
    }
    let timer = f.crop(regions::TIMER);
    if score_visible(f) || bright_fraction(&timer, 195) > 0.08 || red_fraction(&timer) > 0.15 {
        return Scene::Gameplay;
    }
    Scene::NoHud
}
