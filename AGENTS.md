# Repository guidance

## Code

- Make the smallest change that works. Add retries, locks or extra states only for a
  failure that has actually happened.
- Keep each crate to its job: `vision/` holds stateless pixel readers, `scoreboard/` holds
  the ONNX model readers behind `vbt`'s `scoreboard` feature, and `vbt/` holds I/O
  (`stream`, `riot`, `context`, `push`, `ledger`) and the state machine (`tracker/`).
- Give repeated string literals a const, as `tracker::kind` does for ledger kinds.
- Inline single-use bindings.
- Fix clippy findings instead of adding `#[allow]`.
- Log with `tracing`. Only the CLI's own output (`--canonical`, shadow JSONL) goes to
  stdout.
- Return errors from paths that depend on frame or file contents; do not `unwrap` there.
- Comment only what the code does not show. No history or "ported from" notes.

## Assets

- Image assets are embedded at build time. A new glyph, strip or pause template needs a
  provenance row (broadcast, VOD, timestamp) in `assets/recognizer/README.md`.
- Nothing team-specific ships: team tags are read by OCR and by crops learned at runtime.
- Regenerate the scoreboard model only with `scoreboard/tools/prune_head.py`, then update
  the sha256 in `scoreboard/build.rs` and `docs/scoreboard-native.md` and upload the new
  file to the asset host under a new version folder.
- Retrain the digit classifier only with `vision/tools/train_digits.py`. New training
  glyphs go into a new archive version on the asset host, with its URL and sha256 in the
  tool.
- Test images are not in git. A new fixture goes into the test data archive, and
  `scripts/fetch-testdata.sh` gets the new archive URL and sha256.

## Contracts

- The `PUT /api/v1/video/score` body and the ledger JSONL keys are read by other
  programs. A change to either updates the README's push contract and the ledger tables
  in `docs/architecture.md` in the same PR.
- When a rule or threshold changes, update `docs/architecture.md`; the code wins where
  they disagree.

## Testing

- Avoid new tests. Extend an existing focused test when it can cover the change; add a
  case only for a contract nothing else covers.
- Assert what other code can see: published payloads, ledger events, HTTP behaviour. Do
  not test private state.
- Cover the hot path and at most one failure case per contract. Each case should name the
  regression it guards against.

There is no CI, so run these before pushing:

```sh
scripts/fetch-testdata.sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo clippy -p vbt --no-default-features --all-targets -- -D warnings
```

## Docs

- Plain sentences for another engineer. Say what the thing does.
- No hostnames, home directories or private repositories in docs, comments or examples.

## Pull requests

- One commit per PR, amended as it changes, with a conventional-commit subject.
- No AI attribution lines in commits or PR bodies.
