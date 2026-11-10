//! A port of `Resize` from imaging v1.6.2 (MIT, see NOTICE), bit for bit, for opaque
//! single- and three-channel images.
//!
//! It keeps imaging's pass order (horizontal then vertical, rounding to `u8`
//! between the two), its weight normalisation and its float operation order: the
//! alpha factor of 255 stays in the sums, and the result is `c * (1/a)`, not `c/a`.

use std::f64::consts::PI;

/// The imaging filters the tracker uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    /// 3-lobe Lanczos, imaging's `Lanczos`.
    Lanczos,
    /// Bilinear, imaging's `Linear`.
    Linear,
}

impl Filter {
    fn support(self) -> f64 {
        match self {
            Filter::Lanczos => 3.0,
            Filter::Linear => 1.0,
        }
    }
}

fn kernel(f: Filter, x: f64) -> f64 {
    let x = x.abs();
    match f {
        Filter::Lanczos => {
            if x < 3.0 {
                sinc(x) * sinc(x / 3.0)
            } else {
                0.0
            }
        }
        Filter::Linear => {
            if x < 1.0 {
                1.0 - x
            } else {
                0.0
            }
        }
    }
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

/// The source indices and normalized filter weights for each destination sample.
fn weights(dst: usize, src: usize, filter: Filter) -> Vec<Vec<(usize, f64)>> {
    let du = src as f64 / dst as f64;
    let scale = du.max(1.0);
    let ru = (scale * filter.support()).ceil();

    let mut out = Vec::with_capacity(dst);
    for v in 0..dst {
        let fu = (v as f64 + 0.5) * du - 0.5;
        let begin = ((fu - ru).ceil() as i64).max(0);
        let end = ((fu + ru).floor() as i64).min(src as i64 - 1);

        let mut row = Vec::new();
        let mut sum = 0.0;
        let mut u = begin;
        while u <= end {
            let w = kernel(filter, (u as f64 - fu) / scale);
            if w != 0.0 {
                sum += w;
                row.push((u as usize, w));
            }
            u += 1;
        }
        if sum != 0.0 {
            for (_, w) in row.iter_mut() {
                *w /= sum;
            }
        }
        out.push(row);
    }
    out
}

/// imaging's `clamp`: `x + 0.5` truncated, then clamped to 0..255.
fn clamp(x: f64) -> u8 {
    let v = (x + 0.5) as i64;
    v.clamp(0, 255) as u8
}

fn horizontal_gray(src: &[u8], sw: usize, sh: usize, dw: usize, filter: Filter) -> Vec<u8> {
    let weights = weights(dw, sw, filter);
    let mut dst = vec![0u8; dw * sh];
    for y in 0..sh {
        let row = &src[y * sw..(y + 1) * sw];
        for (x, weight) in weights.iter().enumerate() {
            let mut c = 0.0;
            let mut a = 0.0;
            for &(u, w) in weight {
                let aw = 255.0 * w; // the scan line is opaque gray: alpha 255
                c += row[u] as f64 * aw;
                a += aw;
            }
            if a != 0.0 {
                dst[y * dw + x] = clamp(c * (1.0 / a));
            }
        }
    }
    dst
}

fn vertical_gray(src: &[u8], sw: usize, sh: usize, dh: usize, filter: Filter) -> Vec<u8> {
    let weights = weights(dh, sh, filter);
    let mut dst = vec![0u8; sw * dh];
    for (y, weight) in weights.iter().enumerate() {
        for x in 0..sw {
            let mut c = 0.0;
            let mut a = 0.0;
            for &(u, w) in weight {
                let aw = 255.0 * w;
                c += src[u * sw + x] as f64 * aw;
                a += aw;
            }
            if a != 0.0 {
                dst[y * sw + x] = clamp(c * (1.0 / a));
            }
        }
    }
    dst
}

fn horizontal_rgb(src: &[u8], sw: usize, sh: usize, dw: usize, filter: Filter) -> Vec<u8> {
    let weights = weights(dw, sw, filter);
    let mut dst = vec![0u8; dw * sh * 3];
    for y in 0..sh {
        let row = &src[y * sw * 3..(y + 1) * sw * 3];
        for (x, weight) in weights.iter().enumerate() {
            let (mut r, mut g, mut b, mut a) = (0.0, 0.0, 0.0, 0.0);
            for &(u, w) in weight {
                let aw = 255.0 * w;
                r += row[u * 3] as f64 * aw;
                g += row[u * 3 + 1] as f64 * aw;
                b += row[u * 3 + 2] as f64 * aw;
                a += aw;
            }
            if a != 0.0 {
                let a_inv = 1.0 / a;
                let j = (y * dw + x) * 3;
                dst[j] = clamp(r * a_inv);
                dst[j + 1] = clamp(g * a_inv);
                dst[j + 2] = clamp(b * a_inv);
            }
        }
    }
    dst
}

fn vertical_rgb(src: &[u8], sw: usize, sh: usize, dh: usize, filter: Filter) -> Vec<u8> {
    let weights = weights(dh, sh, filter);
    let mut dst = vec![0u8; sw * dh * 3];
    for (y, weight) in weights.iter().enumerate() {
        for x in 0..sw {
            let (mut r, mut g, mut b, mut a) = (0.0, 0.0, 0.0, 0.0);
            for &(u, w) in weight {
                let aw = 255.0 * w;
                r += src[(u * sw + x) * 3] as f64 * aw;
                g += src[(u * sw + x) * 3 + 1] as f64 * aw;
                b += src[(u * sw + x) * 3 + 2] as f64 * aw;
                a += aw;
            }
            if a != 0.0 {
                let a_inv = 1.0 / a;
                let j = (y * sw + x) * 3;
                dst[j] = clamp(r * a_inv);
                dst[j + 1] = clamp(g * a_inv);
                dst[j + 2] = clamp(b * a_inv);
            }
        }
    }
    dst
}

/// Resizes a single-channel image the way `imaging.Resize` does on an `*image.Gray`.
///
/// The input is opaque, so every sample's alpha is 255.
pub fn resize_gray(
    src: &[u8],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
    filter: Filter,
) -> Vec<u8> {
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return Vec::new();
    }
    if sw != dw && sh != dh {
        let mid = horizontal_gray(src, sw, sh, dw, filter);
        vertical_gray(&mid, dw, sh, dh, filter)
    } else if sw != dw {
        horizontal_gray(src, sw, sh, dw, filter)
    } else if sh != dh {
        vertical_gray(src, sw, sh, dh, filter)
    } else {
        src.to_vec()
    }
}

/// Resizes an opaque RGB24 image the way `imaging.Resize` does on an `*image.RGBA`
/// (every sample's alpha 255, so imaging's alpha weighting reduces to channel weighting).
pub fn resize_rgb(
    src: &[u8],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
    filter: Filter,
) -> Vec<u8> {
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return Vec::new();
    }
    if sw != dw && sh != dh {
        let mid = horizontal_rgb(src, sw, sh, dw, filter);
        vertical_rgb(&mid, dw, sh, dh, filter)
    } else if sw != dw {
        horizontal_rgb(src, sw, sh, dw, filter)
    } else if sh != dh {
        vertical_rgb(src, sw, sh, dh, filter)
    } else {
        src[..sw * sh * 3].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_upscale_matches_hand_computed_weights() {
        // src [0, 255] -> 3 samples: precomputeWeights(3, 2, Linear) gives the
        // weights 1; 0.5/0.5; 1, so the middle sample lands at 127.5 -> 128.
        assert_eq!(
            resize_gray(&[0, 255], 2, 1, 3, 1, Filter::Linear),
            vec![0, 128, 255]
        );
    }

    #[test]
    fn constant_images_stay_constant() {
        let src = vec![80u8; 6 * 4];
        assert_eq!(
            resize_gray(&src, 6, 4, 3, 7, Filter::Lanczos),
            vec![80u8; 3 * 7]
        );
        assert_eq!(resize_gray(&src, 6, 4, 6, 4, Filter::Lanczos), src);
    }

    #[test]
    fn rgb_resize_keeps_channels_independent() {
        let src = vec![255, 0, 0, 0, 255, 0];
        let out = resize_rgb(&src, 2, 1, 1, 1, Filter::Linear);
        assert_eq!(out, vec![128, 128, 0]);
    }
}
