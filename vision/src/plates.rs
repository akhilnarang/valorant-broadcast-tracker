//! The score-plate digit reader: autocontrast, Otsu, glyph boxes, a square-centre
//! canvas, imaging's Lanczos to 24x32, then a logistic-regression softmax.

use crate::Error;
use crate::frame::{Frame, Rect};
use crate::regions::{GLYPH_HEIGHT, GLYPH_WIDTH};
use crate::resample::{Filter, resize_gray};

/// The per-plate softmax floor below which a read is refused.
pub const CONFIDENCE: f64 = 0.90;

/// The digit logistic regression: one weight row per class over the 24x32 canvas.
pub struct Lr {
    pub classes: Vec<u8>,
    coef: Vec<Vec<f64>>,
    intercept: Vec<f64>,
}

impl Lr {
    /// Parses `digits_lr.bin`: i32 classes, i32 features, i32 ids, f32 intercepts,
    /// f32 coefficients row-major, all little-endian.
    pub fn parse(raw: &[u8]) -> Result<Lr, Error> {
        let mut pos = 0usize;
        let take = |pos: &mut usize, n: usize| -> Result<&[u8], Error> {
            if *pos + n > raw.len() {
                return Err(Error::Model("digits_lr.bin is truncated".into()));
            }
            let slice = &raw[*pos..*pos + n];
            *pos += n;
            Ok(slice)
        };
        let classes = i32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap());
        let features = i32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap());
        if classes <= 0 || features <= 0 {
            return Err(Error::Model("digits_lr.bin has bad dimensions".into()));
        }
        let (classes, features) = (classes as usize, features as usize);
        let ids = take(&mut pos, classes * 4)?;
        let intercepts = take(&mut pos, classes * 4)?;
        let coef = take(&mut pos, classes * features * 4)?;
        let mut class_ids = Vec::with_capacity(classes);
        for i in 0..classes {
            class_ids.push(i32::from_le_bytes(ids[i * 4..i * 4 + 4].try_into().unwrap()) as u8);
        }
        let intercept = (0..classes)
            .map(|i| f32::from_le_bytes(intercepts[i * 4..i * 4 + 4].try_into().unwrap()) as f64)
            .collect();
        let mut weights = vec![vec![0f64; features]; classes];
        for (c, row) in weights.iter_mut().enumerate() {
            for (f, weight) in row.iter_mut().enumerate() {
                let i = (c * features + f) * 4;
                *weight = f32::from_le_bytes(coef[i..i + 4].try_into().unwrap()) as f64;
            }
        }
        Ok(Lr {
            classes: class_ids,
            coef: weights,
            intercept,
        })
    }

    /// Softmax of the class logits for one feature vector, shifted by the largest logit
    /// before `exp` so it cannot overflow.
    pub fn probabilities(&self, x: &[f64]) -> Vec<f64> {
        let mut logits = vec![0f64; self.classes.len()];
        let mut best = f64::NEG_INFINITY;
        for (c, logit) in logits.iter_mut().enumerate() {
            let mut sum = self.intercept[c];
            for (i, v) in x.iter().enumerate() {
                sum += self.coef[c][i] * v;
            }
            *logit = sum;
            if sum > best {
                best = sum;
            }
        }
        let mut total = 0.0;
        for v in logits.iter_mut() {
            *v = (*v - best).exp();
            total += *v;
        }
        for v in logits.iter_mut() {
            *v /= total;
        }
        logits
    }
}

/// The classifier's verdict on one plate.
pub struct PlateScore {
    /// `None` when the plate was refused: no glyph boxes, or softmax below the floor.
    pub value: Option<i64>,
    /// The minimum best-class softmax across the plate's digits (0.0 when unsegmented).
    pub confidence: f64,
}

/// Reads the embedded classifier once, then scores plate crops.
pub struct PlateReader {
    lr: Lr,
}

impl PlateReader {
    pub fn new() -> Result<PlateReader, Error> {
        Ok(PlateReader {
            lr: Lr::parse(crate::DIGITS_LR_BYTES)?,
        })
    }

    /// Reads a plate crop: one or two digits at softmax >= 0.90, or `None`.
    pub fn score(&self, f: &Frame) -> PlateScore {
        let (gray, boxes) = segment(f);
        if boxes.is_empty() {
            return PlateScore {
                value: None,
                confidence: 0.0,
            };
        }
        let mut value = 0i64;
        let mut floor = 1.0f64;
        for b in boxes {
            let Some(x) = canvas(&gray, f.w, b) else {
                return PlateScore {
                    value: None,
                    confidence: 0.0,
                };
            };
            let probs = self.lr.probabilities(&x);
            let mut best = 0usize;
            for c in 1..probs.len() {
                if probs[c] > probs[best] {
                    best = c;
                }
            }
            if probs[best] < floor {
                floor = probs[best];
            }
            value = value * 10 + self.lr.classes[best] as i64;
        }
        if floor < CONFIDENCE {
            return PlateScore {
                value: None,
                confidence: floor,
            };
        }
        PlateScore {
            value: Some(value),
            confidence: floor,
        }
    }
}

/// Binarizes a plate crop and returns the stretched luma plus the glyph boxes after
/// touching-digit splitting and speck removal, or no boxes when it does not hold 1-2 digits.
pub fn segment(f: &Frame) -> (Vec<u8>, Vec<Rect>) {
    let gray = autocontrast(&f.gray());
    let mut mask_gray = gray.clone();
    let mean: f64 = gray.iter().map(|v| *v as f64).sum::<f64>() / gray.len() as f64;
    if mean > 128.0 {
        for v in mask_gray.iter_mut() {
            *v = 255 - *v;
        }
    }
    let threshold = otsu(&mask_gray);
    let mask: Vec<u8> = mask_gray
        .iter()
        .map(|v| u8::from(*v as usize > threshold))
        .collect();
    let mut boxes = ink_boxes(&mask, f.w, f.h);
    if boxes.is_empty() {
        return (gray, Vec::new());
    }
    if boxes.len() == 1 {
        boxes = split_touching(&mask, f.w, boxes[0]);
    }
    let inks: Vec<usize> = boxes
        .iter()
        .map(|b| {
            let mut sum = 0usize;
            for y in b.top..b.bottom {
                for x in b.left..b.right {
                    sum += mask[(y * f.w as i64 + x) as usize] as usize;
                }
            }
            sum
        })
        .collect();
    let median = median_int(&inks);
    let kept: Vec<Rect> = boxes
        .into_iter()
        .zip(&inks)
        .filter(|(_, ink)| **ink as f64 >= 0.3 * median)
        .map(|(b, _)| b)
        .collect();
    if kept.is_empty() || kept.len() > 2 {
        return (gray, Vec::new());
    }
    (gray, kept)
}

/// Pillow-style autocontrast with no cutoff: the lowest and highest used levels map to 0/255.
pub fn autocontrast(gray: &[u8]) -> Vec<u8> {
    let mut hist = [0usize; 256];
    for v in gray {
        hist[*v as usize] += 1;
    }
    let mut lo = 0usize;
    while lo < 256 && hist[lo] == 0 {
        lo += 1;
    }
    let mut hi = 255i64;
    while hi >= 0 && hist[hi as usize] == 0 {
        hi -= 1;
    }
    if hi <= lo as i64 {
        return gray.to_vec();
    }
    let scale = 255.0 / (hi - lo as i64) as f64;
    let offset = -(lo as f64) * scale;
    let mut lut = [0u8; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        let mapped = (i as f64 * scale + offset) as i64;
        *v = mapped.clamp(0, 255) as u8;
    }
    gray.iter().map(|v| lut[*v as usize]).collect()
}

/// Otsu's threshold, scanning levels in order and keeping the first maximum.
pub fn otsu(gray: &[u8]) -> usize {
    let mut hist = [0usize; 256];
    let mut total = 0f64;
    for v in gray {
        hist[*v as usize] += 1;
        total += *v as f64;
    }
    let mut dark_count = 0usize;
    let mut dark_sum = 0f64;
    let mut best_threshold = 128usize;
    let mut best_variance = -1.0f64;
    for (level, &count) in hist.iter().enumerate() {
        dark_count += count;
        if dark_count == 0 {
            continue;
        }
        let light_count = gray.len() - dark_count;
        if light_count == 0 {
            break;
        }
        dark_sum += (level * count) as f64;
        let mean_gap = dark_sum / dark_count as f64 - (total - dark_sum) / light_count as f64;
        let variance = dark_count as f64 * light_count as f64 * mean_gap * mean_gap;
        if variance > best_variance {
            best_variance = variance;
            best_threshold = level;
        }
    }
    best_threshold
}

/// Glyphs: runs of columns with ink that are at least 30% of the mask tall.
fn ink_boxes(mask: &[u8], w: usize, h: usize) -> Vec<Rect> {
    let mut column_ink = vec![false; w];
    for x in 0..w {
        for y in 0..h {
            if mask[y * w + x] != 0 {
                column_ink[x] = true;
                break;
            }
        }
    }
    let mut boxes = Vec::new();
    let mut x = 0usize;
    while x < w {
        if !column_ink[x] {
            x += 1;
            continue;
        }
        let start = x;
        while x < w && column_ink[x] {
            x += 1;
        }
        let mut top = -1i64;
        let mut bottom = -1i64;
        for y in 0..h {
            for cx in start..x {
                if mask[y * w + cx] != 0 {
                    if top < 0 {
                        top = y as i64;
                    }
                    bottom = y as i64 + 1;
                    break;
                }
            }
        }
        if top < 0 {
            continue;
        }
        if (bottom - top) as f64 >= 0.3 * h as f64 {
            boxes.push(Rect::new(start as i64, top, x as i64, bottom));
        }
    }
    boxes
}

/// Splits one wide glyph box at a clear ink valley when two digits render touching.
fn split_touching(mask: &[u8], w: usize, b: Rect) -> Vec<Rect> {
    let width = (b.right - b.left) as usize;
    let mut ink = vec![0usize; width];
    let mut max_ink = 0usize;
    for i in 0..width {
        let mut sum = 0usize;
        for y in b.top..b.bottom {
            sum += mask[(y * w as i64 + b.left + i as i64) as usize] as usize;
        }
        ink[i] = sum;
        if sum > max_ink {
            max_ink = sum;
        }
    }
    let height = (b.bottom - b.top) as f64;
    if width < 20 || ((b.right - b.left) as f64) < 1.6 * 0.72 * height {
        return vec![b];
    }
    let low = (width as f64 * 0.35) as usize;
    let high = (width as f64 * 0.7) as usize;
    if high <= low {
        return vec![b];
    }
    let mut window_min = usize::MAX;
    let mut arg_min = low;
    let mut i = low;
    while i < high && i < ink.len() {
        if ink[i] < window_min {
            window_min = ink[i];
            arg_min = i;
        }
        i += 1;
    }
    if window_min > max_ink / 2 {
        return vec![b];
    }
    if arg_min < 8 || arg_min > width - 8 {
        return vec![b];
    }
    let middle = b.left + arg_min as i64;
    vec![
        Rect::new(b.left, b.top, middle, b.bottom),
        Rect::new(middle, b.top, b.right, b.bottom),
    ]
}

/// Centres a glyph on a square canvas at template size, scaled to 0-1 row-major.
pub fn canvas(gray: &[u8], w: usize, b: Rect) -> Option<Vec<f64>> {
    let gw = (b.right - b.left).max(0) as usize;
    let gh = (b.bottom - b.top).max(0) as usize;
    if gw == 0 || gh == 0 {
        return None;
    }
    let side = gw.max(gh);
    let mut canvas = vec![0u8; side * side];
    let off_x = (side - gw) / 2;
    let off_y = (side - gh) / 2;
    for y in 0..gh {
        for x in 0..gw {
            let v = gray[(b.top + y as i64) as usize * w + (b.left + x as i64) as usize];
            canvas[(off_y + y) * side + off_x + x] = v;
        }
    }
    let resized = resize_gray(
        &canvas,
        side,
        side,
        GLYPH_WIDTH,
        GLYPH_HEIGHT,
        Filter::Lanczos,
    );
    Some(resized.iter().map(|v| *v as f64 / 255.0).collect())
}

fn median_int(values: &[usize]) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lr_parses_and_softmaxes() {
        let lr = Lr::parse(crate::DIGITS_LR_BYTES).unwrap();
        assert_eq!(lr.classes.len(), 10);
        let probs = lr.probabilities(&vec![0.0; 24 * 32]);
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn autocontrast_stretches_min_max() {
        let out = autocontrast(&[10, 10, 20, 30]);
        assert_eq!(out, vec![0, 0, 127, 255]);
    }
}
