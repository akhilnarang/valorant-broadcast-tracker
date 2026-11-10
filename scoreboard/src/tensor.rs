//! Crop to CHW float32: Catmull-Rom resize to 48 px tall, BGR channels, `v/255*2-1`.

use image::imageops::FilterType;
use vbt_vision::frame::{Frame, round_half_even};

use crate::Error;

/// The recognition model's fixed input height.
pub const TENSOR_H: usize = 48;

/// One prepared crop: planar BGR data at `3 * TENSOR_H * width`.
pub struct Crop {
    /// Resized width `max(16, round(w*48/h))`.
    pub width: usize,
    pub data: Vec<f32>,
}

impl Crop {
    /// Resize and lay out one RGB crop, as §3.1 specifies.
    pub fn prepare(crop: &Frame) -> Result<Crop, Error> {
        if crop.w == 0 || crop.h == 0 {
            return Err(Error::Tensor("cannot prepare an empty crop".into()));
        }
        let width =
            (round_half_even(crop.w as f64 * TENSOR_H as f64 / crop.h as f64) as usize).max(16);
        let src = image::RgbImage::from_raw(crop.w as u32, crop.h as u32, crop.pix.clone())
            .ok_or_else(|| Error::Tensor("crop buffer does not match its dimensions".into()))?;
        let resized =
            image::imageops::resize(&src, width as u32, TENSOR_H as u32, FilterType::CatmullRom);
        let mut data = vec![0f32; 3 * TENSOR_H * width];
        for y in 0..TENSOR_H {
            for x in 0..width {
                let p = resized.get_pixel(x as u32, y as u32);
                let i = y * width + x;
                // B, G, R order, the RapidOCR convention.
                data[i] = p[2] as f32 / 127.5 - 1.0;
                data[TENSOR_H * width + i] = p[1] as f32 / 127.5 - 1.0;
                data[2 * TENSOR_H * width + i] = p[0] as f32 / 127.5 - 1.0;
            }
        }
        Ok(Crop { width, data })
    }
}

/// A padded NCHW batch ready for one `Run` call.
pub struct Batch {
    pub n: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Batch {
    /// Right-pads every crop with `0.0` to `width`, which [`Batch::width_for`] picks so
    /// that every crop fits.
    ///
    /// Panics when `crops` is empty or a crop is wider than `width`.
    pub fn new(crops: &[Crop], width: usize) -> Batch {
        assert!(
            !crops.is_empty() && crops.iter().all(|c| c.width <= width),
            "batch of {} crops does not fit width {width}",
            crops.len()
        );
        let mut data = vec![0f32; crops.len() * 3 * TENSOR_H * width];
        for (i, crop) in crops.iter().enumerate() {
            for ch in 0..3 {
                let src = &crop.data[ch * TENSOR_H * crop.width..][..TENSOR_H * crop.width];
                let dst_base = (i * 3 + ch) * TENSOR_H * width;
                for y in 0..TENSOR_H {
                    let src_row = &src[y * crop.width..][..crop.width];
                    let dst = &mut data[dst_base + y * width..][..crop.width];
                    dst.copy_from_slice(src_row);
                }
            }
        }
        Batch {
            n: crops.len(),
            w: width,
            data,
        }
    }

    /// The batch width the §3.1 constants ask for, grown to a multiple of 8 when a crop
    /// resizes wider. The 720p right-name boxes resize to 307 px, for example.
    pub fn width_for(crops: &[Crop], nominal: usize) -> usize {
        let widest = crops.iter().map(|c| c.width).max().unwrap_or(0);
        let rounded = (widest + 7) & !7;
        nominal.max(rounded)
    }
}

/// Timesteps the model emits for a given input width, derived from its output shape formula.
pub fn timesteps(width: usize) -> usize {
    width.saturating_sub(4).div_ceil(8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vbt_vision::frame::Frame;

    #[test]
    fn prepare_is_planar_bgr() {
        // A 1x4 blue column resizes to the 16 px floor; height stays 48.
        let frame = Frame::from_rgb(1, 4, [0u8, 0, 255].repeat(4)).unwrap();
        let crop = Crop::prepare(&frame).unwrap();
        assert_eq!(crop.width, 16);
        let n = TENSOR_H * 16;
        assert!(crop.data[0] > 0.9, "blue channel first");
        assert!(crop.data[n] < -0.9, "green channel second");
        assert!(crop.data[2 * n] < -0.9, "red channel last");
    }

    #[test]
    fn batch_pads_to_width() {
        let frame = Frame::from_rgb(4, 8, vec![128; 4 * 8 * 3]).unwrap();
        let crop = Crop::prepare(&frame).unwrap();
        assert!(crop.width < 32);
        let batch = Batch::new(std::slice::from_ref(&crop), 32);
        assert_eq!(batch.n, 1);
        assert_eq!(batch.data.len(), 3 * TENSOR_H * 32);
        assert!(batch.data[3 * TENSOR_H * 32 - 1].abs() < f32::EPSILON);
    }

    #[test]
    fn width_formula_matches_model() {
        // Measured with onnxruntime on this model: T(300) = 37, T(64) = 8, T(312) = 39.
        assert_eq!(timesteps(300), 37);
        assert_eq!(timesteps(64), 8);
        assert_eq!(timesteps(312), 39);
        assert_eq!(timesteps(16), 2);
    }
}
