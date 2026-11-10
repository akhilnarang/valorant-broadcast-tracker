//! RGB frames and 1080p-relative crops.

use crate::Error;

/// Reference width for the hand-measured regions. All boxes in [`crate::regions`] are
/// expressed at 1920x1080 and scaled to the frame's width.
pub const REFERENCE_WIDTH: i64 = 1920;

/// A pixel region by its edges; right and bottom are exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub left: i64,
    pub top: i64,
    pub right: i64,
    pub bottom: i64,
}

impl Rect {
    pub const fn new(left: i64, top: i64, right: i64, bottom: i64) -> Self {
        Rect {
            left,
            top,
            right,
            bottom,
        }
    }
}

/// An RGB24 image with the same layout ffmpeg writes.
#[derive(Clone, PartialEq, Eq)]
pub struct Frame {
    pub w: usize,
    pub h: usize,
    /// `len == w * h * 3`.
    pub pix: Vec<u8>,
}

impl Frame {
    /// Wraps raw RGB24 bytes after checking their length.
    pub fn from_rgb(w: usize, h: usize, pix: Vec<u8>) -> Result<Frame, Error> {
        if pix.len() < w * h * 3 {
            return Err(Error::Frame(format!(
                "rgb buffer is {} bytes, need {} for {w}x{h}",
                pix.len(),
                w * h * 3
            )));
        }
        Ok(Frame { w, h, pix })
    }

    /// Converts an image crate buffer to a frame.
    pub fn from_rgb_image(img: &image::RgbImage) -> Frame {
        Frame {
            w: img.width() as usize,
            h: img.height() as usize,
            pix: img.as_raw().clone(),
        }
    }

    /// Takes ownership of a decoded RGB buffer without copying it.
    pub fn from_rgb_image_owned(img: image::RgbImage) -> Frame {
        Frame {
            w: img.width() as usize,
            h: img.height() as usize,
            pix: img.into_raw(),
        }
    }

    /// Scales a 1080p region to this frame, rounding each edge half-to-even.
    pub fn scale(&self, r: Rect) -> Rect {
        let k = self.w as f64 / REFERENCE_WIDTH as f64;
        Rect {
            left: round_half_even(r.left as f64 * k),
            top: round_half_even(r.top as f64 * k),
            right: round_half_even(r.right as f64 * k),
            bottom: round_half_even(r.bottom as f64 * k),
        }
    }

    /// Crops a 1080p region, scaled to the frame. Pixels outside the frame come back black.
    pub fn crop(&self, r: Rect) -> Frame {
        self.crop_raw(self.scale(r))
    }

    /// Crops an absolute pixel box without 1080p scaling.
    pub fn crop_raw(&self, r: Rect) -> Frame {
        let w = (r.right - r.left).max(0) as usize;
        let h = (r.bottom - r.top).max(0) as usize;
        let mut pix = vec![0u8; w * h * 3];
        for y in 0..h {
            let sy = r.top + y as i64;
            if sy < 0 || sy >= self.h as i64 {
                continue;
            }
            for x in 0..w {
                let sx = r.left + x as i64;
                if sx < 0 || sx >= self.w as i64 {
                    continue;
                }
                let src = (sy as usize * self.w + sx as usize) * 3;
                let dst = (y * w + x) * 3;
                pix[dst..dst + 3].copy_from_slice(&self.pix[src..src + 3]);
            }
        }
        Frame { w, h, pix }
    }

    /// Converts to 8-bit luma with [`luma`].
    pub fn gray(&self) -> Vec<u8> {
        self.pix[..self.w * self.h * 3]
            .as_chunks::<3>()
            .0
            .iter()
            .map(|&[r, g, b]| luma(r, g, b))
            .collect()
    }

    /// The frame's luma as a 0/255 mask of the pixels brighter than `threshold`.
    pub fn bright_mask(&self, threshold: i64) -> Vec<u8> {
        let gray = self.gray();
        gray.iter()
            .map(|v| if *v as i64 > threshold { 255 } else { 0 })
            .collect()
    }
}

/// Rounds half-to-even (banker's rounding). Region edges are scaled with this rule.
pub fn round_half_even(v: f64) -> i64 {
    let floor = v.floor();
    let diff = v - floor;
    if diff > 0.5 || (diff == 0.5 && (floor as i64) % 2 != 0) {
        floor as i64 + 1
    } else {
        floor as i64
    }
}

/// Luma of one pixel: integer BT.601 weights, truncated.
pub fn luma(r: u8, g: u8, b: u8) -> u8 {
    ((299 * r as u32 + 587 * g as u32 + 114 * b as u32) / 1000) as u8
}

/// One pixel's luma after premultiplying its alpha in 16 bits and truncating each
/// channel back to 8 bits. The glyph templates were built with this rounding.
pub fn rgba_luma(r: u8, g: u8, b: u8, a: u8) -> u8 {
    let premultiply = |v: u8| ((v as u32 * 257 * a as u32 / 255) >> 8) as u8;
    luma(premultiply(r), premultiply(g), premultiply(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_even_rounds_ties_to_even() {
        assert_eq!(round_half_even(0.5), 0);
        assert_eq!(round_half_even(1.5), 2);
        assert_eq!(round_half_even(2.5), 2);
        assert_eq!(round_half_even(-0.5), 0);
        assert_eq!(round_half_even(-1.5), -2);
        assert_eq!(round_half_even(-2.5), -2);
        assert_eq!(round_half_even(836.5), 836);
        assert_eq!(round_half_even(862.5), 862);
    }

    #[test]
    fn scale_720p_edges() {
        let frame = Frame::from_rgb(1280, 720, vec![0; 1280 * 720 * 3]).unwrap();
        // 1100 * 2/3 = 733.33 -> 733, 1272 * 2/3 = 848.0 -> 848.
        assert_eq!(
            frame.scale(Rect::new(1100, 0, 1272, 10)),
            Rect::new(733, 0, 848, 7)
        );
        // 796 * 2/3 = 530.67 -> 531, 824 * 2/3 = 549.33 -> 549.
        assert_eq!(
            frame.scale(Rect::new(0, 796, 10, 824)),
            Rect::new(0, 531, 7, 549)
        );
    }

    #[test]
    fn crop_outside_frame_is_black() {
        let frame = Frame::from_rgb(2, 1, vec![255, 0, 0, 0, 255, 0]).unwrap();
        let c = frame.crop_raw(Rect::new(1, 0, 4, 1));
        assert_eq!(c.w, 3);
        assert_eq!(&c.pix[0..3], &[0, 255, 0]);
        assert_eq!(&c.pix[3..6], &[0, 0, 0]);
        assert_eq!(&c.pix[6..9], &[0, 0, 0]);
    }

    #[test]
    fn bright_mask_keeps_pixels_above_the_level() {
        let mut frame = Frame::from_rgb(3, 1, vec![0; 9]).unwrap();
        // 299*120/1000 = 35.88 -> 35; the 200 gray crosses 150.
        frame.pix[0] = 120;
        frame.pix[1] = 120;
        frame.pix[2] = 120;
        frame.pix[3] = 200;
        frame.pix[4] = 200;
        frame.pix[5] = 200;
        frame.pix[6] = 151;
        frame.pix[7] = 151;
        frame.pix[8] = 151;
        assert_eq!(frame.bright_mask(150), vec![0, 255, 255]);
    }
}
