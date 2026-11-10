//! Digit templates: the class-list glyphs with their one-pixel wraps and 8% scalings,
//! and the template score reader.

use std::collections::BTreeMap;
use std::path::Path;

use crate::Error;
use crate::frame::{Frame, rgba_luma};
use crate::plates::{canvas, segment};
use crate::regions::{GLYPH_HEIGHT, GLYPH_WIDTH};

/// A score read by template matching.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Read {
    /// `-1` when no score was read.
    pub value: i32,
    pub confidence: f64,
}

impl Default for Read {
    fn default() -> Self {
        Read {
            value: -1,
            confidence: 0.0,
        }
    }
}

/// Holds every digit template, one normalised vector per row.
#[derive(Debug, Clone, Default)]
pub struct Templates {
    pub vectors: Vec<Vec<f64>>,
    pub digits: Vec<u8>,
}

struct GlyphEntry {
    name: String,
    digit: u8,
    data: Vec<u8>,
}

impl Templates {
    /// Loads the glyphs compiled into the binary, so nothing depends on the working directory.
    pub fn load_embedded() -> Result<Templates, Error> {
        let mut entries = Vec::with_capacity(crate::GLYPHS.len());
        for (name, digit, data) in crate::GLYPHS {
            entries.push(GlyphEntry {
                name: (*name).to_string(),
                digit: *digit,
                data: data.to_vec(),
            });
        }
        build_templates(entries)
    }

    /// Loads `glyphs/<digit>/*.png` from a directory, with one-pixel shifts and 8% scalings.
    pub fn load_dir(dir: &Path) -> Result<Templates, Error> {
        let mut found: Vec<GlyphEntry> = Vec::new();
        let entries =
            std::fs::read_dir(dir).map_err(|e| Error::Assets(format!("{}: {e}", dir.display())))?;
        let mut digit_dirs: Vec<(String, u8)> = Vec::new();
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.len() != 1 || !name.as_bytes()[0].is_ascii_digit() || !entry.path().is_dir() {
                continue;
            }
            digit_dirs.push((name.clone(), name.as_bytes()[0] - b'0'));
        }
        digit_dirs.sort();
        for (name, digit) in digit_dirs {
            let sub = dir.join(&name);
            let files = std::fs::read_dir(&sub)
                .map_err(|e| Error::Assets(format!("{}: {e}", sub.display())))?;
            let mut paths: Vec<std::path::PathBuf> = files
                .flatten()
                .map(|f| f.path())
                .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("png"))
                .collect();
            paths.sort();
            for path in paths {
                let data = std::fs::read(&path)
                    .map_err(|e| Error::Assets(format!("{}: {e}", path.display())))?;
                found.push(GlyphEntry {
                    name: path.to_string_lossy().into_owned(),
                    digit,
                    data,
                });
            }
        }
        build_templates(found)
    }

    /// Reads a one- or two-digit score by template matching.
    pub fn read_score(&self, f: &Frame) -> Read {
        let (gray, kept) = segment(f);
        if kept.is_empty() {
            return Read::default();
        }
        let mut digits = Vec::with_capacity(kept.len());
        let mut matches = Vec::with_capacity(kept.len());
        let mut leads = Vec::with_capacity(kept.len());
        for &b in &kept {
            let Some(vec) = canvas(&gray, f.w, b) else {
                return Read::default();
            };
            let vec = normalise(&vec);
            let mut best: BTreeMap<u8, f64> = BTreeMap::new();
            for (i, tv) in self.vectors.iter().enumerate() {
                let score = dot(tv, &vec);
                let digit = self.digits[i];
                let entry = best.entry(digit).or_insert(f64::NEG_INFINITY);
                if score > *entry {
                    *entry = score;
                }
            }
            let mut ranked: Vec<(f64, u8)> = best.into_iter().map(|(d, s)| (s, d)).collect();
            ranked.sort_by(|a, b| {
                if a.0 != b.0 {
                    b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    b.1.cmp(&a.1)
                }
            });
            if ranked.len() < 2 {
                return Read::default();
            }
            digits.push(ranked[0].1);
            matches.push(ranked[0].0);
            leads.push(ranked[0].0 - ranked[1].0);
        }
        let mut value = 0i32;
        for d in digits {
            value = value * 10 + d as i32;
        }
        let confidence = min_slice(&matches) * 0.6 + min_slice(&leads) * 2.0;
        Read {
            value,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

fn build_templates(mut entries: Vec<GlyphEntry>) -> Result<Templates, Error> {
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    let mut templates = Templates::default();
    for entry in entries {
        let base =
            decode_glyph(&entry.data).map_err(|e| Error::Assets(format!("{}: {e}", entry.name)))?;
        for variant in variants(&base) {
            templates.vectors.push(normalise(&variant));
            templates.digits.push(entry.digit);
        }
    }
    Ok(templates)
}

/// Reads a glyph PNG as 0-1 floats through [`rgba_luma`]. A glyph must have the
/// canvas size, since every read is scored against a canvas of that size.
fn decode_glyph(data: &[u8]) -> Result<Vec<Vec<f64>>, Error> {
    let img = image::load_from_memory_with_format(data, image::ImageFormat::Png)?;
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width() as usize, rgba.height() as usize);
    if (w, h) != (GLYPH_WIDTH, GLYPH_HEIGHT) {
        return Err(Error::Assets(format!(
            "glyph is {w}x{h}, need {GLYPH_WIDTH}x{GLYPH_HEIGHT}"
        )));
    }
    let mut rows = Vec::with_capacity(h);
    for y in 0..h {
        let mut row = Vec::with_capacity(w);
        for x in 0..w {
            let p = rgba.get_pixel(x as u32, y as u32);
            row.push(rgba_luma(p[0], p[1], p[2], p[3]) as f64 / 255.0);
        }
        rows.push(row);
    }
    Ok(rows)
}

/// The base glyph with six one-pixel wrap-around shifts and two 8% scalings, each
/// flattened row-major.
fn variants(base: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let h = base.len();
    let w = base[0].len();
    let mut out = vec![flatten(base)];
    let roll = |dy: i64, dx: i64| -> Vec<f64> {
        let mut shifted = vec![vec![0f64; w]; h];
        for (y, row) in shifted.iter_mut().enumerate() {
            for (x, cell) in row.iter_mut().enumerate() {
                let sy = (y as i64 + dy).rem_euclid(h as i64) as usize;
                let sx = (x as i64 + dx).rem_euclid(w as i64) as usize;
                *cell = base[sy][sx];
            }
        }
        flatten(&shifted)
    };
    for (dy, dx) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)] {
        out.push(roll(dy, dx));
    }
    for scale in [0.92f64, 1.08] {
        let mut rows = vec![0usize; h];
        let mut cols = vec![0usize; w];
        for (i, row) in rows.iter_mut().enumerate() {
            *row = (((i as f64 - h as f64 / 2.0) / scale + h as f64 / 2.0) as i64)
                .clamp(0, h as i64 - 1) as usize;
        }
        for (i, col) in cols.iter_mut().enumerate() {
            *col = (((i as f64 - w as f64 / 2.0) / scale + w as f64 / 2.0) as i64)
                .clamp(0, w as i64 - 1) as usize;
        }
        let mut scaled = vec![vec![0f64; w]; h];
        for (y, row) in scaled.iter_mut().enumerate() {
            for (x, cell) in row.iter_mut().enumerate() {
                *cell = base[rows[y]][cols[x]];
            }
        }
        out.push(flatten(&scaled));
    }
    out
}

pub(crate) fn flatten(matrix: &[Vec<f64>]) -> Vec<f64> {
    let mut out = Vec::with_capacity(matrix.len() * matrix[0].len());
    for row in matrix {
        out.extend_from_slice(row);
    }
    out
}

/// Flattens pixels to a zero-mean unit vector, so a dot product is their correlation.
pub fn normalise(vec: &[f64]) -> Vec<f64> {
    let mean = vec.iter().sum::<f64>() / vec.len() as f64;
    let norm = vec
        .iter()
        .map(|v| (v - mean) * (v - mean))
        .sum::<f64>()
        .sqrt()
        + 1e-6;
    vec.iter().map(|v| (v - mean) / norm).collect()
}

fn min_slice(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}
