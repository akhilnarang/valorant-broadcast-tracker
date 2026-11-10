//! The §2 pixel boxes at 1920x1080 and the pre-inference presence gate.

use vbt_vision::frame::{Frame, Rect, round_half_even};

/// Vertical pitch of the player rows: ink rows sit at `cy-7 .. cy+7`.
pub const ROW_PITCH: i64 = 56;

/// Centre row of player `i` (0..4).
pub const fn row_centre(i: usize) -> i64 {
    810 + ROW_PITCH * i as i64
}

/// Left (false) and right (true) K/D/A cell centres.
pub const fn cell_centre(right: bool, j: usize) -> f64 {
    match (right, j) {
        (false, 0) => 849.5,
        (false, 1) => 886.5,
        (false, 2) => 923.5,
        (true, 0) => 997.0,
        (true, 1) => 1034.0,
        (true, 2) => 1071.0,
        _ => unreachable!(),
    }
}

/// Player name box; ink is left-aligned on the left and right-aligned on the right.
pub fn name_box(right: bool, i: usize) -> Rect {
    let cy = row_centre(i);
    if right {
        Rect::new(1100, cy - 14, 1272, cy + 14)
    } else {
        Rect::new(648, cy - 14, 800, cy + 14)
    }
}

/// K/D/A cell box, `cx ± 13 x cy ± 12`; two-digit values stay centred.
pub fn cell_box(right: bool, i: usize, j: usize) -> Rect {
    let cy = row_centre(i);
    let cx = cell_centre(right, j);
    Rect::new(
        round_half_even(cx - 13.0),
        cy - 12,
        round_half_even(cx + 13.0),
        cy + 12,
    )
}

/// Round header box; `ROUND 6` ink is y 13..23.
pub const ROUND: Rect = Rect::new(900, 8, 1020, 30);

/// The team tag and score plate boxes the tracker reads.
pub use vbt_vision::regions::{
    SCORE_LEFT, SCORE_RIGHT, TEAM_LEFT as TAG_LEFT, TEAM_RIGHT as TAG_RIGHT,
};

/// The luma level above which a K/D/A cell counts as inked.
const CELL_INK_LUMA: u8 = 170;

/// The dark panel must stay below this median luma.
const PANEL_MEDIAN_LUMA: f64 = 60.0;

/// A row-side passes when its three cells are inked and its surrounding panel is dark.
fn row_side_present(frame: &Frame, right: bool, i: usize) -> bool {
    let cy = row_centre(i);
    let mut inked = true;
    for j in 0..3 {
        let cx = cell_centre(right, j);
        let cell = frame.crop(Rect::new(
            round_half_even(cx - 13.0),
            cy - 8,
            round_half_even(cx + 13.0),
            cy + 8,
        ));
        let max = cell.gray().into_iter().max().unwrap_or(0);
        if max < CELL_INK_LUMA {
            inked = false;
            break;
        }
    }
    if !inked {
        return false;
    }
    let x0 = cell_centre(right, 0);
    let x2 = cell_centre(right, 2);
    let panel = frame.crop(Rect::new(
        round_half_even(x0 - 15.0),
        cy - 20,
        round_half_even(x2 + 15.0),
        cy + 20,
    ));
    median(&panel.gray()) <= PANEL_MEDIAN_LUMA
}

/// The table counts as present when at least 8 of the 10 row-sides pass. Pure pixel work.
pub fn table_present(frame: &Frame) -> bool {
    let mut passes = 0;
    for i in 0..5 {
        for right in [false, true] {
            if row_side_present(frame, right, i) {
                passes += 1;
                if passes >= 8 {
                    return true;
                }
            }
        }
    }
    false
}

fn median(values: &[u8]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid] as f64
    } else {
        (sorted[mid - 1] as f64 + sorted[mid] as f64) / 2.0
    }
}
