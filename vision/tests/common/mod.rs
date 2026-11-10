//! Shared helpers for the vision integration tests.
#![allow(dead_code)] // each test binary uses a different subset

use std::path::{Path, PathBuf};

use vbt_vision::frame::Frame;

include!("../../../testdata/fixture.rs");

/// `testdata/frames/` at the repository root.
pub fn frames_dir() -> PathBuf {
    fixture("testdata/frames")
}

/// Decodes an image file into an RGB frame.
pub fn load_frame(path: &Path) -> Frame {
    let image = image::open(path).unwrap_or_else(|e| panic!("decode {}: {e}", path.display()));
    Frame::from_rgb_image_owned(image.to_rgb8())
}

/// Decodes `dir/name` into an RGB frame.
pub fn load_named(dir: &Path, name: &str) -> Frame {
    load_frame(&dir.join(name))
}
