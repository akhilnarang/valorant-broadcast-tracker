//! The team tag reader: the two expected codes scored as CTC lexicon entries over each tag
//! line, accepted only when the line's free decode spells the winner.
//!
//! It needs no per-team asset, so it reads teams the learned crops in
//! `vbt_vision::learned` have not seen yet, and teaches them.
//!
//! Measured over labelled gameplay frames of 23 teams at 720p: no side was accepted as the
//! wrong team. Per-side read rates with the raw crop alone, then with the stretched
//! variants of [`STRETCHES`]: TL 7/240 to 181/240, T1 1/247 to 240/248, KC 87/98 to 94/98
//! and TYL 21/125 to 125/125.

use std::collections::HashMap;
use std::sync::Mutex;

use image::imageops::FilterType;
use ort::session::Session;

use vbt_vision::frame::{Frame, Rect};

use crate::ctc;
use crate::tensor::{Batch, Crop, timesteps};
use crate::{Error, dlog};

/// The left tag line at 1920x1080: the tag only, without the seed line below it.
pub const TAG_LEFT: Rect = Rect::new(722, 10, 784, 36);
/// The right tag line at 1920x1080.
pub const TAG_RIGHT: Rect = Rect::new(1124, 10, 1204, 36);

/// The CTC margin (nats) the better code needs over the other.
///
/// TL against TYL legitimately sits near 5. False accepts were the same at every margin
/// from 0 to 10, because the free decode's exact spelling does the refusing.
pub const TAG_MARGIN: f64 = 3.0;
/// The CTC score per character (nats) the better code needs. The worst accepted positive
/// on the 2026 HUD scored -1.04; negatives that passed the spelling check scored -2.90 or
/// less.
pub const TAG_PER_CHAR: f64 = -1.5;
/// Horizontal stretches tried in order until a side is accepted; 1.0 is the raw crop.
///
/// The HUD's condensed font makes the model drop a narrow `T`, so TL decodes as `L` and
/// T1 as `1`. Widening the gray crop separates the glyphs. Each stretched variant costs
/// another model run: a two-tag read took 13.4 ms raw, 19.4 ms at 1.3 and 26.7 ms at 1.8.
pub const STRETCHES: [f64; 3] = [1.0, 1.3, 1.8];
/// Edge pixels added on each side of a stretched crop, so the stretched glyphs at the
/// window's edges are not cut off at the tensor border.
const PAD: u32 = 8;

/// One tag line's read: the better expected code and the evidence for it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TagRead {
    /// The accepted code, upper-cased, or `""`.
    pub code: String,
    /// The better of the two expected codes.
    pub best: String,
    /// The line's free decode.
    pub free: String,
    /// The better code's margin over the other, in nats.
    pub margin: f64,
}

/// The tag reader: one recognition session.
pub struct Reader {
    session: Mutex<Session>,
    classes: Vec<char>,
    index: HashMap<char, usize>,
}

/// The letters of `text` as the free decode is compared: alphanumerics only, upper-cased,
/// with the model's confusions on this font folded (`6` for `G`, `0` for `O`, `Ü` for
/// `U`). `L` and `1` are not folded, or TL would read as T1.
fn letters(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_uppercase)
        .map(|c| match c {
            '6' => 'G',
            '0' => 'O',
            'Ü' => 'U',
            c => c,
        })
        .collect()
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

    /// Reads both tag lines (left, right) as one of `codes`, each on its own.
    pub fn read(&self, frame: &Frame, codes: [&str; 2]) -> [TagRead; 2] {
        let mut reads = [TagRead::default(), TagRead::default()];
        let [a, b] = codes.map(|code| code.trim().to_uppercase());
        if a.is_empty() || a == b {
            return reads;
        }
        let mut pending = [true, true];
        for stretch in STRETCHES {
            let todo: Vec<usize> = (0..2).filter(|&side| pending[side]).collect();
            if todo.is_empty() {
                break;
            }
            let rects = todo.iter().map(|&side| [TAG_LEFT, TAG_RIGHT][side]);
            let windows = match self.windows(frame, rects, stretch) {
                Ok(windows) => windows,
                Err(error) => {
                    dlog!("tags: inference failed: {error}");
                    break;
                }
            };
            for (side, (probs, t)) in todo.into_iter().zip(windows) {
                reads[side] = self.side(&probs, t, &a, &b);
                pending[side] = reads[side].code.is_empty();
            }
        }
        reads
    }

    /// Scores both codes and the free decode on one line's softmax.
    fn side(&self, probs: &[f32], t: usize, a: &str, b: &str) -> TagRead {
        let c = self.classes.len();
        let score = |word: &str| {
            [word.to_string(), word.to_lowercase()]
                .iter()
                .filter_map(|variant| ctc::extension(variant, &self.index))
                .map(|ext| ctc::score(probs, c, t, &ext))
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let (sa, sb) = (score(a), score(b));
        let (best, top, margin) = if sb > sa {
            (b, sb, sb - sa)
        } else {
            (a, sa, sa - sb)
        };
        let mut free = String::new();
        let mut previous = 0;
        for row in probs[..t * c].chunks(c) {
            let class = row
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.total_cmp(y.1))
                .map_or(0, |(index, _)| index);
            if class != 0 && class != previous {
                free.push(self.classes[class]);
            }
            previous = class;
        }
        let accepted = letters(&free) == letters(best)
            && margin >= TAG_MARGIN
            && top / best.chars().count() as f64 >= TAG_PER_CHAR;
        TagRead {
            code: if accepted {
                best.to_string()
            } else {
                String::new()
            },
            best: best.to_string(),
            free: free.trim().to_string(),
            margin,
        }
    }

    /// Runs the model on the given tag lines at `stretch`, in one batch: each line's
    /// softmax and the timesteps to use.
    fn windows(
        &self,
        frame: &Frame,
        rects: impl Iterator<Item = Rect>,
        stretch: f64,
    ) -> Result<Vec<(Vec<f32>, usize)>, Error> {
        let crops = rects
            .map(|rect| Crop::prepare(&stretched(&frame.crop(rect), stretch)))
            .collect::<Result<Vec<_>, _>>()?;
        let batch = Batch::new(&crops, Batch::width_for(&crops, 0));
        let mut session = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (probs, t, c) = crate::model::run(&mut session, batch.n, batch.w, &batch.data)?;
        Ok(crops
            .iter()
            .enumerate()
            .map(|(i, crop)| {
                (
                    probs[i * t * c..(i + 1) * t * c].to_vec(),
                    t.min(timesteps(crop.width)),
                )
            })
            .collect())
    }
}

/// The crop as is at `stretch` 1.0; otherwise gray, widened by `stretch` and edge-padded
/// by [`PAD`] on each side.
fn stretched(crop: &Frame, stretch: f64) -> Frame {
    if stretch == 1.0 {
        return crop.clone();
    }
    let gray = crop.gray();
    let image = image::RgbImage::from_fn(crop.w as u32, crop.h as u32, |x, y| {
        let v = gray[y as usize * crop.w + x as usize];
        image::Rgb([v, v, v])
    });
    let width = (crop.w as f64 * stretch).round() as u32;
    let wide = image::imageops::resize(&image, width, crop.h as u32, FilterType::CatmullRom);
    let padded = image::RgbImage::from_fn(width + 2 * PAD, crop.h as u32, |x, y| {
        *wide.get_pixel(x.saturating_sub(PAD).min(width - 1), y)
    });
    Frame::from_rgb_image_owned(padded)
}
