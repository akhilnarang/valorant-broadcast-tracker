//! Team tag plates matched against crops learned at runtime. Nothing team-specific ships:
//! the tracker teaches the store a match's two tags from frames where the OCR reader
//! spells both, and the white tag foreground is then matched by shape on every frame.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::Error;
use crate::frame::Frame;
use crate::regions::{TEAM_LEFT, TEAM_RIGHT};
use crate::resample::{Filter, resize_gray};

/// Crops kept per code and frame height. A crop is only learned for a plate the code's
/// existing crops did not read, so each one covers a rendering the others missed, and once
/// a code has three the oldest is replaced: crops of a HUD that changed age out. Three
/// keep the per-frame match cheap (0.27 ms for both plates with 23 codes learned, since
/// every learned code is matched for the margin).
pub const KEEP: usize = 3;
/// The similarity a code's best crop needs to name a plate.
///
/// With the shown team's own crops removed, the best wrong-code crop scored at most 0.878
/// over about 8,000 labelled gameplay plates, and TL and T1 cross at 0.814 at most. Own
/// crops from other windows read TL 240/240, T1 246/247, KC 98/98 and GE 103/109 at 720p.
pub const MATCH: f64 = 0.93;
/// The similarity a code's best crop needs over every other learned code's best. With
/// [`MATCH`] it gave no false match from 0.90/0.05 upwards on the same corpus.
pub const MARGIN: f64 = 0.08;

struct Crop {
    code: String,
    /// The height of the frame it was cut from. Crops do not transfer across
    /// resolutions: a 1080p crop scores 0.84 at the 5th percentile on the same team's
    /// 720p plate.
    height: usize,
    /// The `<n>` of its file name.
    slot: usize,
    mask: Vec<u8>,
    w: usize,
    h: usize,
    ratio: f64,
}

/// The learned tag crops, saved as `CODE_<height>_<n>.png` plate crops in `dir`.
pub struct Crops {
    dir: Option<PathBuf>,
    crops: Vec<Crop>,
}

/// Carries independent evidence for both plates. `ok` is false for missing, unknown,
/// ambiguous or duplicate tags; one readable plate never determines the other by
/// elimination.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Read {
    pub left: String,
    pub right: String,
    pub left_confidence: f64,
    pub right_confidence: f64,
    pub ok: bool,
}

impl Crops {
    /// Loads the crops saved in `dir`, oldest first, which need not exist yet. `None`
    /// keeps learned crops in memory only. A file that does not decode to tag ink is
    /// skipped: it is relearned from the broadcast.
    pub fn open(dir: Option<&Path>) -> Result<Crops, Error> {
        let mut crops = Vec::new();
        if let Some(dir) = dir.filter(|dir| dir.exists()) {
            let entries = std::fs::read_dir(dir)
                .map_err(|e| Error::Assets(format!("tag crops: {}: {e}", dir.display())))?;
            let mut files: Vec<_> = entries
                .flatten()
                .filter_map(|entry| {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    let (code, height, slot) = parse_name(&name)?;
                    let modified = entry.metadata().and_then(|m| m.modified()).ok();
                    Some((modified, name, code, height, slot))
                })
                .collect();
            files.sort();
            for (_, name, code, height, slot) in files {
                let path = dir.join(&name);
                let frame =
                    image::open(&path).map(|image| Frame::from_rgb_image_owned(image.to_rgb8()));
                let Some((mask, w, h, ratio)) = frame.as_ref().ok().and_then(foreground) else {
                    tracing::warn!(path = %path.display(), "skipping an unreadable tag crop");
                    continue;
                };
                crops.push(Crop {
                    code,
                    height,
                    slot,
                    mask,
                    w,
                    h,
                    ratio,
                });
            }
        }
        Ok(Crops {
            dir: dir.map(Path::to_path_buf),
            crops,
        })
    }

    /// The crops learned for `code` at frame height `height`.
    pub fn count(&self, code: &str, height: usize) -> usize {
        self.crops
            .iter()
            .filter(|crop| crop.height == height && crop.code.eq_ignore_ascii_case(code))
            .count()
    }

    /// Learns `codes` (left, right) from this frame's plates; pass `""` for a plate not to
    /// learn. Each code given must have been read on this frame by the OCR, on a plate its
    /// own crops did not read. A code already holding [`KEEP`] crops at this height loses
    /// its oldest. Returns how many crops it learned. A failed save keeps the crop in
    /// memory, stops saving and returns the error once.
    pub fn learn(&mut self, frame: &Frame, codes: [&str; 2]) -> Result<usize, Error> {
        let mut learned = 0;
        let mut failed = None;
        for (code, region) in codes.into_iter().zip([TEAM_LEFT, TEAM_RIGHT]) {
            let code = code.trim().to_uppercase();
            // The code names a file, so only a plain tag is learned.
            if code.is_empty() || !code.chars().all(|c| c.is_ascii_alphanumeric()) {
                continue;
            }
            let crop = frame.crop(region);
            let Some((mask, w, h, ratio)) = foreground(&crop) else {
                continue;
            };
            let same = |c: &Crop| c.height == frame.h && c.code == code;
            let slot = if self.count(&code, frame.h) >= KEEP {
                let oldest = self.crops.iter().position(same).unwrap_or(0);
                self.crops.remove(oldest).slot
            } else {
                (0..KEEP)
                    .find(|n| !self.crops.iter().any(|c| same(c) && c.slot == *n))
                    .unwrap_or(0)
            };
            let name = format!("{code}_{}_{slot}.png", frame.h);
            self.crops.push(Crop {
                code,
                height: frame.h,
                slot,
                mask,
                w,
                h,
                ratio,
            });
            learned += 1;
            if let Some(dir) = &self.dir
                && let Err(error) = save(dir, &name, crop)
            {
                self.dir = None;
                failed = Some(error);
            }
        }
        failed.map_or(Ok(learned), Err)
    }

    /// Reads each plate as one of the two distinct expected codes, from crops learned at
    /// this frame's height.
    pub fn read(&self, frame: &Frame, codes: &[String]) -> Read {
        if codes.len() != 2 {
            return Read::default();
        }
        let a = codes[0].trim().to_uppercase();
        let b = codes[1].trim().to_uppercase();
        if a.is_empty() || a == b {
            return Read::default();
        }
        let (left, left_confidence) = self.match_one(&frame.crop(TEAM_LEFT), frame.h, &a, &b);
        let (right, right_confidence) = self.match_one(&frame.crop(TEAM_RIGHT), frame.h, &a, &b);
        Read {
            ok: !left.is_empty() && !right.is_empty() && left != right,
            left,
            right,
            left_confidence,
            right_confidence,
        }
    }

    /// The expected code whose best crop clears [`MATCH`] and beats every other learned
    /// code by [`MARGIN`], with the better expected code's similarity.
    fn match_one(&self, crop: &Frame, height: usize, a: &str, b: &str) -> (String, f64) {
        let Some((mask, mask_w, mask_h, ratio)) = foreground(crop) else {
            return (String::new(), 0.0);
        };
        let mut best: BTreeMap<&str, f64> = BTreeMap::new();
        for t in self.crops.iter().filter(|crop| crop.height == height) {
            let resized = if mask_w == t.w && mask_h == t.h {
                mask.clone()
            } else {
                resize_gray(&mask, mask_w, mask_h, t.w, t.h, Filter::Linear)
            };
            let mut intersection = 0.0;
            let mut total = 0.0;
            for (v, w) in resized.iter().zip(&t.mask) {
                let v = *v as f64 / 255.0;
                let w = *w as f64 / 255.0;
                intersection += v * w;
                total += v * v + w * w;
            }
            let score = 2.0 * intersection / total * (ratio / t.ratio).min(t.ratio / ratio);
            let entry = best.entry(&t.code).or_insert(f64::NEG_INFINITY);
            *entry = entry.max(score);
        }
        let score = |code: &str| best.get(code).copied().unwrap_or(0.0);
        let (code, top) = if score(b) > score(a) {
            (b, score(b))
        } else {
            (a, score(a))
        };
        let rival = best
            .iter()
            .filter(|(other, _)| **other != code)
            .map(|(_, score)| *score)
            .fold(0.0, f64::max);
        if top < MATCH || top - rival < MARGIN {
            return (String::new(), top);
        }
        (code.to_string(), top)
    }
}

/// `CODE_<height>_<n>.png` as the code, the frame height and `n`.
fn parse_name(name: &str) -> Option<(String, usize, usize)> {
    let mut parts = name.strip_suffix(".png")?.split('_');
    let code = parts.next()?.to_uppercase();
    let height = parts.next()?.parse().ok()?;
    let slot = parts.next()?.parse().ok()?;
    Some((code, height, slot))
}

/// Saves `crop` as `dir/name` through a temporary file, so a crash never leaves a torn
/// crop under a name `open` loads.
fn save(dir: &Path, name: &str, crop: Frame) -> Result<(), Error> {
    std::fs::create_dir_all(dir)?;
    let temp = dir.join(format!(".{name}.tmp"));
    image::RgbImage::from_raw(crop.w as u32, crop.h as u32, crop.pix)
        .ok_or_else(|| Error::Frame("tag crop buffer is short".into()))?
        .save_with_format(&temp, image::ImageFormat::Png)?;
    std::fs::rename(&temp, dir.join(name))?;
    Ok(())
}

/// Ignores the dim regional seed line below the tag, and keeps the white text. Tight
/// bounds remove alignment differences; aspect ratio remains part of the match score.
fn foreground(f: &Frame) -> Option<(Vec<u8>, usize, usize, f64)> {
    let gray = f.gray();
    let mut min_x = f.w as i64;
    let mut min_y = f.h as i64;
    let mut max_x = 0i64;
    let mut max_y = 0i64;
    let mut ink = 0usize;
    for y in 0..f.h {
        for x in 0..f.w {
            if gray[y * f.w + x] >= 180 {
                min_x = min_x.min(x as i64);
                min_y = min_y.min(y as i64);
                max_x = max_x.max(x as i64 + 1);
                max_y = max_y.max(y as i64 + 1);
                ink += 1;
            }
        }
    }
    let dx = max_x - min_x;
    let dy = max_y - min_y;
    if dx < 3 || dy < 6 || ink < 12 || ink as f64 / (dx * dy) as f64 > 0.85 {
        return None;
    }
    let (w, h) = (dx as usize, dy as usize);
    let mut mask = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let sy = min_y as usize + y;
            let sx = min_x as usize + x;
            if gray[sy * f.w + sx] >= 180 {
                mask[y * w + x] = 255;
            }
        }
    }
    Some((mask, w, h, w as f64 / h as f64))
}
