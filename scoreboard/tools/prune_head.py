# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx", "numpy"]
# ///
"""Prune the PP-OCRv6 recognition head to the scoreboard's alphabet.

The scoreboard reader never decodes free text. Every cell is scored against a
lexicon (roster names, `ROUND 1`..`ROUND 50`, `0`..`99`, the two match tags), so
the only characters that need a class are the ones that can appear in those
strings: the printable ASCII block plus the Latin-1 letters. RapidOCR's model
ships a 6904-entry dict (CJK, fullwidth, symbols, ...) whose final MatMul/Add
head is 80x6906, about half the model's bytes and a large share of its
per-timestep compute.

This script slices that head to the kept classes. It writes the kept list back
into `metadata["character"]`, so `scoreboard/src/model.rs` keeps building its
class list the same way (`[blank] + metadata lines + [space]`) and no index
table lives in code.

Classes in the pruned model:
    0            blank
    1..94        the printable ASCII block, `!` (0x21) through `~` (0x7E)
    95..126      the Latin-1 letters, `ß` through `ÿ` (minus `÷`)
    127          space

The original model's class indices for `!`..`~` are 1..94 and for the Latin-1
letters they are 155..187; space is 6905. Everything else is dropped.

Softmax renormalises over the kept classes, so the score of any lexicon entry
gains `-sum_t ln Z_t` (`Z_t` = kept mass at timestep `t`) and is otherwise
identical: the winning candidate and its margin over the runner-up are exactly
the same as with the full head, on every crop.

Usage:
    uv run scoreboard/tools/prune_head.py [--input PATH] [--output PATH]

The input is the full `PP-OCRv6_rec_tiny.onnx` from RapidOCR v3.9.2 (Apache-2.0).
Without `--input` it is downloaded once from RapidOCR into `~/.cache/vbt/`. Its
sha256 is checked before pruning. `--output` defaults to
`PP-OCRv6_rec_pruned.onnx` in the current directory.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path

import onnx
from onnx import numpy_helper

SOURCE_URL = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/rec/PP-OCRv6_rec_tiny.onnx"
SOURCE_SHA256 = "e16e242de5937ad92609223f19bc2aff3727ee40b095f996907c24749bad251b"
DEFAULT_OUTPUT = "PP-OCRv6_rec_pruned.onnx"

BLANK, SPACE = "\0", " "

# The characters a scoreboard cell can hold beyond blank and space: every
# printable ASCII character (`-`, `_`, `.`, digits, both letter cases) and the
# Latin-1 letters for accented roster names.
ASCII = [chr(c) for c in range(0x21, 0x7F)]
LATIN1 = "ßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"
KEPT_CHARS = ASCII + list(LATIN1)


def class_names(model: onnx.ModelProto) -> list[str]:
    """The class list the Rust reader builds: blank, dict lines, then space."""
    meta = {p.key: p.value for p in model.metadata_props}
    return [BLANK] + meta["character"].split("\n") + [SPACE]


def prune(model: onnx.ModelProto, kept: list[str]) -> None:
    """Slice the final MatMul/Add head to `kept` classes and rewrite the metadata, in place."""
    graph = model.graph
    names = class_names(model)
    if names[0] != BLANK or names[-1] != SPACE:
        raise SystemExit("blank is not class 0 or space is not the last class")
    missing = [c for c in kept if c not in names]
    if missing:
        raise SystemExit(f"not in the model dict: {missing!r}")
    indices = [names.index(c) for c in kept]
    if len(set(indices)) != len(indices):
        raise SystemExit("duplicate class in the kept list")

    producer = {out: node for node in graph.node for out in node.output}

    # The head is: MatMul(x, W) -> Add(bias) -> Identity -> Softmax -> output.
    softmax = producer[graph.output[0].name]
    if softmax.op_type != "Softmax":
        raise SystemExit(f"graph output comes from {softmax.op_type}")
    identity = producer[softmax.input[0]]
    add = producer[identity.input[0]]
    matmul = producer[add.input[0]]
    if (matmul.op_type, add.op_type) != ("MatMul", "Add"):
        raise SystemExit(f"unexpected head: {matmul.op_type} then {add.op_type}")

    weight_name, bias_name = matmul.input[1], add.input[1]
    initializers = {t.name: t for t in graph.initializer}
    weight = numpy_helper.to_array(initializers[weight_name])
    bias = numpy_helper.to_array(initializers[bias_name])
    if weight.shape[1] != len(names) or bias.shape[0] != len(names):
        raise SystemExit(
            f"head is {weight.shape}/{bias.shape}, class list is {len(names)}"
        )

    print(
        f"head: {weight.shape[0]}x{weight.shape[1]} -> "
        f"{weight.shape[0]}x{len(indices)}; classes: {len(names)} -> {len(indices)}"
    )
    for name, array in (
        (weight_name, weight[:, indices].copy()),
        (bias_name, bias[indices].copy()),
    ):
        initializers[name].CopyFrom(numpy_helper.from_array(array, name))

    # Outputs and saved intermediates carry the class count as a static last dim.
    for value in [*graph.value_info, *graph.output]:
        dims = value.type.tensor_type.shape.dim
        if dims and dims[-1].dim_value == len(names):
            dims[-1].dim_value = len(indices)

    # The reader derives its classes from this list, in order: kept[1:-1].
    meta = {p.key: p for p in model.metadata_props}
    meta["character"].value = "\n".join(kept[1:-1])


def fetch_source() -> Path:
    """Download the full model into the user cache unless it is already there."""
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "vbt"
    path = cache / "PP-OCRv6_rec_tiny.onnx"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(SOURCE_URL, partial)
        partial.rename(path)
    return path


def check_source(path: Path) -> None:
    """Refuse an input that is not the expected full model."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise SystemExit(f"{path}: sha256 is {digest}, expected {SOURCE_SHA256}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    args = parser.parse_args()
    args.input = args.input or fetch_source()
    if args.output.resolve() == args.input.resolve():
        raise SystemExit("--output must differ from --input")

    check_source(args.input)
    model = onnx.load(args.input)
    before = len(class_names(model))
    prune(model, [BLANK] + KEPT_CHARS + [SPACE])
    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    print(f"{args.input}: {before} classes -> {args.output}: {len(KEPT_CHARS) + 2}")


if __name__ == "__main__":
    main()
