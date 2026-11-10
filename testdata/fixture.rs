// Finds the test images, which are not in git. Each test crate pulls this in with
// `include!`, so it has no `use` items of its own.

/// `relative` under the repository root. Panics with the fetch command when it is missing.
pub fn fixture(relative: &str) -> std::path::PathBuf {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join(relative);
    assert!(
        path.exists(),
        "{} is missing; fetch the test images with scripts/fetch-testdata.sh",
        path.display()
    );
    path
}
