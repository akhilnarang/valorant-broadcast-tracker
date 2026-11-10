//! Learned team tag crops at the source and production frame sizes.

mod common;

use common::fixture;
use vbt_vision::frame::{Frame, round_half_even};
use vbt_vision::learned::Crops;
use vbt_vision::regions::{TEAM_LEFT, TEAM_RIGHT};
use vbt_vision::resample::{Filter, resize_rgb};

/// `testdata/teamplates/<name>` at `width`, scaled as ffmpeg's 720p decode would.
fn load(name: &str, width: usize) -> Frame {
    let path = fixture(&format!("testdata/teamplates/{name}"));
    let image = image::open(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .to_rgb8();
    if width == image.width() as usize {
        return Frame::from_rgb_image_owned(image);
    }
    let height = 53;
    let resized = resize_rgb(
        image.as_raw(),
        image.width() as usize,
        image.height() as usize,
        width,
        height,
        Filter::Linear,
    );
    Frame::from_rgb(width, height, resized).unwrap()
}

/// `frame` with its two tag plates swapped, each centred in the other's box.
fn swapped(mut frame: Frame) -> Frame {
    let left = frame.crop(TEAM_LEFT);
    let right = frame.crop(TEAM_RIGHT);
    let scale = frame.w as f64 / 1920.0;
    for (source, dest) in [(right, TEAM_LEFT), (left, TEAM_RIGHT)] {
        let x0 = round_half_even(dest.left as f64 * scale);
        let y0 = round_half_even(dest.top as f64 * scale);
        let w = frame.crop(dest).w as i64;
        for y in 0..source.h {
            for x in 0..w {
                let i = (((y0 + y as i64) * frame.w as i64 + x0 + x) * 3) as usize;
                let sx = x - (w - source.w as i64) / 2;
                let pixel = if sx >= 0 && sx < source.w as i64 {
                    let si = (y * source.w + sx as usize) * 3;
                    [source.pix[si], source.pix[si + 1], source.pix[si + 2]]
                } else {
                    [0; 3]
                };
                frame.pix[i..i + 3].copy_from_slice(&pixel);
            }
        }
    }
    frame
}

/// Crops learned from the map-start frame and saved read a different round's plates at
/// both decode sizes, follow the tags when the plates swap sides (so they match text, not
/// a side), and refuse a replay HUD. Guards the Abyss reversal, where Riot listed the
/// sides the wrong way round.
#[test]
fn test_learned_crops_read_the_tags_they_were_taught() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("learned-tag-crops");
    let _ = std::fs::remove_dir_all(&dir);
    let codes = ["G2".to_string(), "TL".to_string()];
    for width in [1920usize, 1280] {
        let mut crops = Crops::open(Some(&dir)).expect("open an empty store");
        assert!(!crops.read(&load("abyss-round5.png", width), &codes).ok);
        crops
            .learn(&load("abyss-start.png", width), ["TL", "G2"])
            .expect("learn");
        // The saved crops, not the in-memory ones, must carry the read.
        let crops = Crops::open(Some(&dir)).expect("reload");
        let round5 = load("abyss-round5.png", width);
        let got = crops.read(&round5, &codes);
        assert!(
            got.ok && got.left == "TL" && got.right == "G2",
            "width {width}: {got:?}"
        );
        let got = crops.read(&swapped(round5), &codes);
        assert!(
            got.ok && got.left == "G2" && got.right == "TL",
            "width {width} swapped plates: {got:?}"
        );
        let got = crops.read(&load("abyss-replay.png", width), &codes);
        assert!(!got.ok, "width {width} replay accepted: {got:?}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
