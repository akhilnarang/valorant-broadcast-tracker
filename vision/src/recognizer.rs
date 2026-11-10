//! The score-plate classifier plus the `CURRENT:` map-strip and pause phrase readers.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::Error;
use crate::frame::Frame;
use crate::plates::{PlateReader, autocontrast};
use crate::templates::{flatten, normalise};

/// The minimum correlation for finding the word CURRENT: on the strip.
pub const ANCHOR_THRESHOLD: f64 = 0.75;
/// The gap between the `CURRENT:` template and the map name that follows it.
pub const NAME_GAP: i64 = 2;
/// The minimum correlation for accepting a map-name match.
pub const MAP_THRESHOLD: f64 = 0.55;
/// How far the best map name must beat the runner-up.
pub const MAP_MARGIN: f64 = 0.05;
/// The minimum correlation for reading a pause overlay's header phrase.
pub const PAUSE_THRESHOLD: f64 = 0.60;
/// How far the best phrase must beat the runner-up.
pub const PAUSE_MARGIN: f64 = 0.05;
/// The luma above which a pause header's pixels are taken to be text.
pub const PAUSE_MASK_LEVEL: i64 = 150;

#[derive(Deserialize)]
struct StripAsset {
    current: Vec<Vec<f64>>,
    maps: BTreeMap<String, Vec<Vec<f64>>>,
}

#[derive(Deserialize)]
struct PauseAsset {
    #[serde(default)]
    phrases: Vec<PausePhrase>,
}

/// One pause overlay phrase. `text` is the whole phrase the template draws; a template
/// for "TECH PAUSE - GEAR" carries its reason, so a bare "TECH PAUSE" leaves `reason`
/// empty. `region` is `""` for the lower-third header or `"card"` for the full-screen
/// break card's title line.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PausePhrase {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub region: String,
    #[serde(default)]
    pub rows: Vec<Vec<f64>>,
    #[serde(skip)]
    norm: Vec<f64>,
}

/// One frame's read of a pause overlay phrase.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PauseRead {
    pub kind: String,
    pub reason: String,
    pub score: f64,
}

/// Where `CURRENT:` was found on a strip, and how well it correlated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anchor {
    pub score: f64,
    pub x: i64,
    pub y: i64,
}

struct MapTemplate {
    name: String,
    rows: Vec<Vec<f64>>,
    norm: Vec<f64>,
    w: i64,
    h: i64,
}

/// Holds the plate classifier and the strip templates.
pub struct Recognizer {
    plate: PlateReader,
    current: Vec<Vec<f64>>,
    current_norm: Vec<f64>,
    current_w: i64,
    current_h: i64,
    maps: Vec<MapTemplate>,
    pauses: Vec<PausePhrase>,
}

impl Recognizer {
    /// Loads the embedded model, strip templates and pause phrases.
    pub fn new() -> Result<Recognizer, Error> {
        let plate = PlateReader::new()?;
        let strip: StripAsset = serde_json::from_slice(crate::STRIP_TEMPLATES_BYTES)?;
        let current = strip.current;
        let (current_w, current_h) = template_shape(&current);
        let current_norm = normalised(&current);
        let mut maps: Vec<MapTemplate> = strip
            .maps
            .into_iter()
            .map(|(name, tmpl)| {
                let rows = trim_separator(tmpl);
                let (w, h) = template_shape(&rows);
                MapTemplate {
                    name,
                    norm: normalised(&rows),
                    rows,
                    w,
                    h,
                }
            })
            .collect();
        maps.sort_by(|a, b| a.name.cmp(&b.name));
        let pause_asset: PauseAsset = serde_json::from_slice(crate::PAUSE_TEMPLATES_BYTES)?;
        let pauses = pause_asset
            .phrases
            .into_iter()
            .map(|mut phrase| {
                phrase.norm = normalised(&phrase.rows);
                phrase
            })
            .collect();
        Ok(Recognizer {
            plate,
            current,
            current_norm,
            current_w,
            current_h,
            maps,
            pauses,
        })
    }

    /// Reads a plate crop: one or two digits at softmax >= 0.90, or `None`.
    pub fn score(&self, crop: &Frame) -> Option<i32> {
        self.plate.score(crop).value.map(|v| v as i32)
    }

    /// The `CURRENT:` template rows, for tests that build synthetic strips.
    #[doc(hidden)]
    pub fn current_rows(&self) -> &[Vec<f64>] {
        &self.current
    }

    /// One map template's rows, for tests that build synthetic strips.
    #[doc(hidden)]
    pub fn map_rows(&self, name: &str) -> Option<&[Vec<f64>]> {
        self.maps
            .iter()
            .find(|m| m.name == name)
            .map(|m| m.rows.as_slice())
    }

    /// Reads the map name following `CURRENT:` on the strip, or `""` when it finds none.
    pub fn map(&self, strip: &Frame) -> String {
        let Some((gray, x, y)) = self.gray_anchor(strip) else {
            return String::new();
        };
        let (w, h) = (strip.w as i64, strip.h as i64);
        let x0 = self.name_x(x);
        let mut best_name = String::new();
        let mut best_score = -2.0f64;
        let mut second_score = -2.0f64;
        for t in &self.maps {
            let (s, _, _) = best_shift(
                &gray,
                Shift {
                    w,
                    h,
                    th: t.h,
                    tw: t.w,
                    x0: x0 - 3,
                    x1: x0 + 6,
                    y0: y,
                    y1: y,
                },
                &t.norm,
            );
            if s > best_score {
                second_score = best_score;
                best_score = s;
                best_name = t.name.clone();
            } else if s > second_score {
                second_score = s;
            }
        }
        if best_score < MAP_THRESHOLD || best_score - second_score < MAP_MARGIN {
            return String::new();
        }
        best_name
    }

    /// Reads a pause overlay's phrase: the best match above the threshold and past the
    /// runner-up margin, as the map reader requires. `score` is the best correlation
    /// seen even when nothing matched.
    pub fn pause(&self, header: &Frame, card: &Frame) -> (PauseRead, bool) {
        let mut best = -2.0f64;
        let mut second = -2.0f64;
        let mut best_idx: Option<usize> = None;
        let mut scored = false;
        for (i, phrase) in self.pauses.iter().enumerate() {
            if phrase.rows.is_empty() || phrase.rows[0].is_empty() {
                continue;
            }
            let crop = if phrase.region == "card" {
                card
            } else {
                header
            };
            if crop.w == 0 || crop.h == 0 {
                continue;
            }
            let mask = crop.bright_mask(PAUSE_MASK_LEVEL);
            let tw = phrase.rows[0].len() as i64;
            let th = phrase.rows.len() as i64;
            let (score, _, _) = best_shift(
                &mask,
                Shift {
                    w: crop.w as i64,
                    h: crop.h as i64,
                    th,
                    tw,
                    x0: 0,
                    x1: crop.w as i64 - tw,
                    y0: 0,
                    y1: crop.h as i64 - th,
                },
                &phrase.norm,
            );
            scored = true;
            if score > best {
                second = best;
                best = score;
                best_idx = Some(i);
            } else if score > second {
                second = score;
            }
        }
        if !scored {
            return (PauseRead::default(), false);
        }
        if best < PAUSE_THRESHOLD || best - second < PAUSE_MARGIN {
            return (
                PauseRead {
                    kind: String::new(),
                    reason: String::new(),
                    score: best,
                },
                false,
            );
        }
        let phrase = &self.pauses[best_idx.unwrap()];
        (
            PauseRead {
                kind: phrase.kind.clone(),
                reason: phrase.reason.clone(),
                score: best,
            },
            true,
        )
    }

    /// The strip's best `CURRENT:` correlation and where it was found, whether or not it
    /// clears [`ANCHOR_THRESHOLD`]. The plate and template readers only run past the
    /// threshold; the model-backed strip name reader uses a lower-confidence anchor when
    /// the name itself still separates decisively.
    pub fn anchor(&self, strip: &Frame) -> Anchor {
        self.locate(strip).1
    }

    /// The x where the map name starts: [`NAME_GAP`] px after the `CURRENT:` anchor.
    pub fn name_x(&self, anchor_x: i64) -> i64 {
        anchor_x + self.current_w + NAME_GAP
    }

    /// The strip's autocontrasted luma and the best `CURRENT:` shift, without a threshold.
    fn locate(&self, strip: &Frame) -> (Vec<u8>, Anchor) {
        let gray = autocontrast(&strip.gray());
        if self.current_w == 0 || self.current_h == 0 {
            return (
                gray,
                Anchor {
                    score: -2.0,
                    x: -1,
                    y: -1,
                },
            );
        }
        let (w, h) = (strip.w as i64, strip.h as i64);
        let (score, x, y) = best_shift(
            &gray,
            Shift {
                w,
                h,
                th: self.current_h,
                tw: self.current_w,
                x0: 0,
                x1: w - self.current_w,
                y0: 0,
                y1: 3,
            },
            &self.current_norm,
        );
        (gray, Anchor { score, x, y })
    }

    /// The strip's autocontrasted luma and the anchor position of `CURRENT:`.
    fn gray_anchor(&self, strip: &Frame) -> Option<(Vec<u8>, i64, i64)> {
        let (gray, anchor) = self.locate(strip);
        if anchor.score >= ANCHOR_THRESHOLD {
            Some((gray, anchor.x, anchor.y))
        } else {
            None
        }
    }
}

fn template_shape(rows: &[Vec<f64>]) -> (i64, i64) {
    if rows.is_empty() || rows[0].is_empty() {
        (0, 0)
    } else {
        (rows[0].len() as i64, rows.len() as i64)
    }
}

/// A template's normalised pixels, or nothing for an empty template.
fn normalised(rows: &[Vec<f64>]) -> Vec<f64> {
    if template_shape(rows).0 == 0 {
        return Vec::new();
    }
    normalise(&flatten(rows))
}

/// Cuts a map template at the separator its fixed-width capture picked up after the
/// name, and at the low-contrast background beyond it.
pub(crate) fn trim_separator(tmpl: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    if tmpl.is_empty() || tmpl[0].is_empty() {
        return tmpl;
    }
    let width = tmpl[0].len();
    let mean = tmpl.iter().flat_map(|row| row.iter()).sum::<f64>() / (tmpl.len() * width) as f64;
    let mut contrast = vec![0f64; width];
    for row in &tmpl {
        for (x, v) in row.iter().enumerate() {
            let c = (v - mean).abs();
            if c > contrast[x] {
                contrast[x] = c;
            }
        }
    }
    let peak = contrast.iter().copied().fold(0f64, f64::max);
    const LOW_CUT: f64 = 0.55;
    const MIN_RUN: usize = 3;
    let mut cut = width;
    let mut x = width / 2;
    while x < width {
        if contrast[x] >= LOW_CUT * peak {
            x += 1;
            continue;
        }
        let start = x;
        while x < width && contrast[x] < LOW_CUT * peak {
            x += 1;
        }
        if x - start >= MIN_RUN {
            cut = start;
        }
    }
    // A map name is dozens of pixels wide; anything shorter means the rule caught a gap
    // inside a name, and the template is better left whole.
    if cut == width || cut < 20 {
        return tmpl;
    }
    tmpl.into_iter().map(|row| row[..cut].to_vec()).collect()
}

/// The window and template shape `best_shift` searches.
pub(crate) struct Shift {
    pub w: i64,
    pub h: i64,
    pub th: i64,
    pub tw: i64,
    pub x0: i64,
    pub x1: i64,
    pub y0: i64,
    pub y1: i64,
}

/// Slides a normalised template over the gray image and returns the best correlation
/// and where it was found. `x1` and `y1` are inclusive.
pub(crate) fn best_shift(gray: &[u8], shift: Shift, template: &[f64]) -> (f64, i64, i64) {
    let Shift {
        w,
        h,
        th,
        tw,
        x0,
        x1,
        y0,
        y1,
    } = shift;
    if th == 0 || tw == 0 || th > h || tw > w {
        return (-2.0, -1, -1);
    }
    let x1 = x1.min(w - tw);
    let y1 = y1.min(h - th);
    let mut best = -2.0f64;
    let mut bx = -1i64;
    let mut by = -1i64;
    let mut y = y0;
    while y <= y1 {
        if y >= 0 {
            let mut x = x0;
            while x <= x1 {
                if x >= 0 {
                    let s = correlate(gray, w, x, y, th, tw, template);
                    if s > best {
                        best = s;
                        bx = x;
                        by = y;
                    }
                }
                x += 1;
            }
        }
        y += 1;
    }
    (best, bx, by)
}

/// The dot product of the normalised template with the normalised patch. It computes
/// `(v / norm) * t` in that order so reads stay bit-identical to the recorded fixtures.
fn correlate(gray: &[u8], w: i64, x: i64, y: i64, th: i64, tw: i64, template: &[f64]) -> f64 {
    let mut mean = 0.0;
    for dy in 0..th {
        let row = (y + dy) * w + x;
        for dx in 0..tw {
            mean += gray[(row + dx) as usize] as f64 / 255.0;
        }
    }
    let n = (th * tw) as f64;
    mean /= n;
    let mut norm = 0.0;
    let mut patch = Vec::with_capacity((th * tw) as usize);
    for dy in 0..th {
        let row = (y + dy) * w + x;
        for dx in 0..tw {
            let v = gray[(row + dx) as usize] as f64 / 255.0 - mean;
            patch.push(v);
            norm += v * v;
        }
    }
    norm = norm.sqrt() + 1e-6;
    let mut sum = 0.0;
    for (i, v) in patch.iter().enumerate() {
        sum += (v / norm) * template[i];
    }
    sum
}
