//! Puts the pruned recognition model in `OUT_DIR` for `include_bytes!`.
//!
//! Set `VBT_SCOREBOARD_MODEL` to the absolute path of a local copy to skip the download. Otherwise the model
//! is downloaded once and kept in `OUT_DIR`. Either way its sha256 must match.

use std::{env, fs, path::PathBuf};

const URL: &str = "https://files.akhilnarang.dev/cdn/valorant/ocr/v1/PP-OCRv6_rec_pruned.onnx";
const SHA256: &str = "8c857f945502eab54827376bee605c81e01faccbc269538ab3d096a8764500fc";

fn main() {
    println!("cargo::rerun-if-env-changed=VBT_SCOREBOARD_MODEL");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("PP-OCRv6_rec_pruned.onnx");
    let (bytes, source) = if let Some(path) = env::var_os("VBT_SCOREBOARD_MODEL") {
        let path = PathBuf::from(path);
        println!("cargo::rerun-if-changed={}", path.display());
        let bytes = fs::read(&path)
            .unwrap_or_else(|e| panic!("VBT_SCOREBOARD_MODEL={}: {e}", path.display()));
        (bytes, path.display().to_string())
    } else if fs::read(&out).is_ok_and(|bytes| sha256(&bytes) == SHA256) {
        return;
    } else {
        (download(), URL.to_string())
    };
    let got = sha256(&bytes);
    if got != SHA256 {
        panic!("{source}: sha256 is {got}, expected {SHA256}");
    }
    fs::write(&out, bytes).unwrap_or_else(|e| panic!("{}: {e}", out.display()));
}

fn download() -> Vec<u8> {
    ureq::get(URL)
        .call()
        .and_then(|response| response.into_body().read_to_vec())
        .unwrap_or_else(|e| {
            panic!(
                "download {URL}: {e}; set VBT_SCOREBOARD_MODEL to the absolute path of a local copy"
            )
        })
}

fn sha256(bytes: &[u8]) -> String {
    hmac_sha256::Hash::hash(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
