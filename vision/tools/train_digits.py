# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1", "scikit-learn==1.9.1", "pillow"]
# ///
"""Train the score-plate digit classifier, `assets/recognizer/digits_lr.bin`.

The classifier is a multinomial logistic regression over the 24x32 grayscale glyph
canvas (768 pixels, row-major, scaled to 0-1), one weight row per digit. It is trained
on two glyph sets:

- the 6,267 harvested glyphs in `digits-train-v1.tar.zst` on the asset host, cut from
  four maps of a live broadcast and labelled with the confirmed score at each second;
- the 192 glyphs in `assets/glyphs/`.

The output layout, all little-endian: i32 classes, i32 features, i32 class ids,
f32 intercepts, f32 coefficients row-major. With the pinned versions above the output
is byte-identical to the committed file on x86-64. On aarch64 some weights differ in
their last float32 bit.

Usage:
    uv run vision/tools/train_digits.py [--glyphs DIR] [--output PATH]

Without `--glyphs` the archive is downloaded once into `~/.cache/vbt/` and its sha256
checked. `--output` defaults to `assets/recognizer/digits_lr.bin`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

ARCHIVE_URL = "https://files.akhilnarang.dev/cdn/valorant/ocr/v1/digits-train-v1.tar.zst"
ARCHIVE_SHA256 = "88d14c7d9796ffc4a9faaea8b6bbf4823b2cb81f3a6af615b0a3b5760bb643e7"
ROOT = Path(__file__).resolve().parents[2]


def fetch_archive() -> Path:
    """Download the harvested glyphs into the user cache and unpack them, once."""
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "vbt"
    glyphs = cache / "digits-train-v1"
    if glyphs.exists():
        return glyphs / "glyphs"
    # The asset host refuses urllib's default User-Agent.
    request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "vbt-tools"})
    raw = urllib.request.urlopen(request).read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise SystemExit(f"{ARCHIVE_URL}: sha256 is {digest}, expected {ARCHIVE_SHA256}")
    partial = cache / "digits-train-v1.part"
    partial.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:zst") as tar:
        tar.extractall(partial, filter="data")
    partial.rename(glyphs)
    return glyphs / "glyphs"


def load(folders: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Each `<folder>/<digit>/*.png`, in sorted order, as feature rows and labels."""
    pixels, labels = [], []
    for folder in folders:
        for path in sorted(folder.glob("[0-9]/*.png")):
            pixels.append(np.asarray(Image.open(path).convert("L"), dtype=np.float32).ravel() / 255.0)
            labels.append(int(path.parent.name))
    return np.stack(pixels).astype(np.float64), np.array(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--glyphs", type=Path, help="an unpacked harvested glyph folder")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/recognizer/digits_lr.bin")
    args = parser.parse_args()

    features, labels = load([args.glyphs or fetch_archive(), ROOT / "assets/glyphs"])
    model = LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced").fit(features, labels)
    classes, width = model.coef_.shape
    with args.output.open("wb") as out:
        out.write(np.array([classes, width], dtype="<i4").tobytes())
        out.write(model.classes_.astype("<i4").tobytes())
        out.write(model.intercept_.astype("<f4").tobytes())
        out.write(model.coef_.astype("<f4").tobytes())
    print(f"{args.output}: {len(labels)} glyphs, train accuracy {model.score(features, labels):.4f}")


if __name__ == "__main__":
    main()
