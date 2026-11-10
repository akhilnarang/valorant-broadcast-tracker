# Native scoreboard reader

`vbt-scoreboard` reads the buy-phase table of the broadcast: the round, both team tags and
scores, and 10 player rows of name plus K/D/A. It runs in-process. The same model also
reads the map strip's name (§6), which is what the tracker's map naming uses first.

## 1. Decision

One approach: **lexicon-constrained CTC scoring on a single tiny recognition model**.
There is no detector and no free-text OCR.

- Model: `PP-OCRv6_rec_pruned.onnx` (2.2 MB), RapidOCR's `PP-OCRv6_rec_tiny.onnx`
  (4.5 MB, Apache-2.0, RapidOCR v3.9.2 release, sha256
  `e16e242de5937ad92609223f19bc2aff3727ee40b095f996907c24749bad251b`) with its
  classification head sliced to the 128 classes a cell can hold (blank, printable ASCII,
  the Latin-1 letters, space) by `scoreboard/tools/prune_head.py`, which downloads the full model
  from RapidOCR (or takes `--input`) and checks its sha256. Only the pruned model is published; pruned sha256
  `8c857f945502eab54827376bee605c81e01faccbc269538ab3d096a8764500fc`.
  The head is about half the model's bytes and a large share of its per-timestep compute.
- We already know every string a cell can contain:
  - each name cell holds one of that team's roster names;
  - each K/D/A cell holds `0`..`99`;
  - the header holds `ROUND 1`..`ROUND 50`;
  - each tag holds one of the two match tags.
- So the model never *decodes* text. For each candidate string we compute its CTC
  log-likelihood from the model's per-timestep probabilities, take the best candidate, and
  accept it only when it clears a margin over the runner-up.
- Team **scores** keep the production logistic-regression plate reader
  (`vision::plates`, weights `assets/recognizer/digits_lr.bin`). It is already trusted at
  97.5% coverage and about 0.1% error.

**Measured effect of the pruned head.** Medians over alternated `--bench 80` runs pinned to
P-cores on a 12th-gen i5 (relative numbers only; the machine drifts):

| Input | threads | full head | pruned, pattern off | pruned, pattern on |
|---|---|---|---|---|
| TL 1080p PNG | 1 | 82.9 ms | 80.2 ms | 72.9 ms |
| TL 1080p PNG | 2 | 58.3 ms | 59.4 ms | 53.3 ms |
| XLG 720p JPEG | 1 | 82.4 ms | 80.7 ms | 72.3 ms |
| XLG 720p JPEG | 2 | 60.0 ms | 61.0 ms | 54.0 ms |

Peak RSS drops from 49.6/49.3 MB (TL 1080p at 1/2 threads) to 42.8-43.6/43.6-43.8 MB, and
from 46.4/46.2 MB to 40.3/40.3 MB on XLG. All 270 field decisions across the three fixtures
keep their winner and verdict; margins are unchanged to f32 rounding (≤ 2e-7) and absolute
scores rise by at most 0.44 nats (mean 0.03), from the `-sum_t ln Z_t` renormalisation, which
crosses no acceptance threshold on the fixtures.

### Why not pure template matching

- Names: we do not have the broadcast name font, and rendering candidates in a substitute
  font gives poor correlation. Harvested per-player templates do not exist for a new
  roster. CTC scoring needs neither.
- K/D/A digits: one frame holds only digits 0–6, so a template set has to be harvested
  across many frames. The model already reads all 30 cells correctly on both test frames
  at 720p and 1080p, with large margins (table below).
- The cost is one onnxruntime dependency. It is contained in its own crate
  (`scoreboard/`), so a build that disables the scoreboard feature never links ONNX.

### Measured evidence

Measurements used ONNX Runtime CPU, PP-OCRv6 tiny, crops at the boxes in §2, and BGR
input. Margins are in nats (log-likelihood of best minus runner-up).

| Field | TL–G2 R6, 1080p PNG | same frame downscaled to 720p | XLG–KC R19, 720p JPEG |
|---|---|---|---|
| 10 names (roster lexicon) | all correct, margin ≥ 25, per-char ≥ −1.34 | all correct, margin ≥ 31 | all correct, margin ≥ 30 |
| 30 K/D/A cells (`0`..`99`) | all correct, margin ≥ 8.6, abs ≥ −0.19 | all correct, margin ≥ 7.7 | all correct, incl. 2-digit values, margin ≥ 8.1 |
| round, lexicon 1..40 | `ROUND 6`, margin 8.1 | margin **0.6** (6 vs 8) | `ROUND 19`, margin 7.1 |
| round, lexicon {s, s+1} | margin 11.0 | margin 6.6 | margin 8.8 |
| tags ({two tags}) | TL 14.7, G2 30.8 | TL 14.0, G2 31.7 | XLG 21.9, KC 27.0 |

Other measurements:

- Wrong roster names score at least −7.8 per character. An empty background cell scores
  −2.9 for its best number.
- For comparison, free greedy decoding of whole K/D/A strings confuses `/` with `1` or
  `3`, and the round at 720p reads as `ROUNDB`. That is why we use per-cell crops and
  lexicons.
- Inference cost for the whole frame (1 names batch of 10 plus 1 cells batch of 30): about
  66 ms at 1 intra-op thread and about 45 ms at 2 threads. A native binary adds roughly 1–2 ms
  of preprocessing.
- Memory: an ORT tiny session is about 17–20 MB. A batch of 10 at 48×240 adds about 20 MB
  of activations.

## 2. Geometry (1080p pixel boxes, scaled by frame width like `Frame::crop`)

All boxes are `Left, Top, Right, Bottom` at 1920×1080, with right and bottom exclusive.
Scale every edge by `W/1920` and round half-to-even, as `vision::frame::Frame::crop`
does. These boxes were measured on both test frames.

Two details are easy to get wrong:

1. The row pitch is **56 px (0.05185·H)**, not 0.0503·H. With 0.0503 the boxes drift
   about 1.7 px per row, and row 5's text (y 1027–1041) gets clipped.
2. Right-side names are **right-aligned and end at x≈1266**. A window of
   x 0.62–0.67 (1190–1286) cuts off BABYBAY (it starts at 1181) and JAWGEMO (it starts at
   1171).

| Region | Box at 1080p |
|---|---|
| Row centre `cy(i)`, i = 0..4 | `810 + 56*i` (ink rows 803–817, 859–873, …) |
| Left name (left-aligned, ink starts at 654) | `648, cy-14, 800, cy+14` |
| Right name (right-aligned, ink ends at 1266) | `1100, cy-14, 1272, cy+14` |
| K/D/A cell centres, left | `849.5, 886.5, 923.5` |
| K/D/A cell centres, right | `997, 1034, 1071` |
| K/D/A cell | `round(cx-13), cy-12, round(cx+13), cy+12`. The dim slashes sit at about 864–869 / 901–906 (left) and 1012–1017 / 1049–1054 (right) and stay outside the cell. |
| Round header | `900, 8, 1020, 30` (`ROUND 6` ink is y 13–23) |
| Left tag | `vision::regions::TEAM_LEFT` = `730, 8, 780, 36` |
| Right tag | `vision::regions::TEAM_RIGHT` = `1130, 8, 1200, 36` |
| Left / right score plate | `vision::regions::SCORE_LEFT` / `SCORE_RIGHT` |

Two-digit K/D/A values are centred on the same cell centre. This was verified on XLG–KC
R19 (`13/14/2`, `24/7/5`, `6/8/10`).

**Presence gate.** This is pure pixel work and runs before any inference.

- A row-side passes when two conditions hold:
  - all three of its K/D/A cells, cropped to `cx±13, cy±8`, have max luma ≥ 170;
  - the median luma of `x0-15 .. x2+15, cy±20` is ≤ 60 (the dark panel), where `x0` and
    `x2` are the first and last cell centres.
- The table counts as present when ≥ 8 of the 10 row-sides pass.
- Otherwise return "no table" (JSON `null`) without running the model.
- Use luma `(299R+587G+114B)/1000`, the same as `vision::frame`.

## 3. Algorithm

### 3.1 Model I/O

- Input `x`: float32 `[N, 3, 48, W]`. Output `fetch_name_0`: float32 `[N, T, C]`, already
  softmaxed, with `T ≈ W/8`.
- Classes: `["<blank>"] + metadata["character"].split("\n") + [" "]`, one class per line.
  - Blank is index 0 and space is the last index.
  - The pruned model's dict already lists only the kept alphabet, so the class indices are
    the dense pruned ones and no mapping table lives in code.
  - At init, assert `len(classes) == C` using a 1×3×48×64 dry run. Fail if they differ.
- A name or tag with a character outside the alphabet is not scorable and is skipped; on
  the fixtures every roster name is inside it (the Latin-1 letters cover accented names).
- Preprocessing for one crop:
  1. Crop RGB at frame resolution.
  2. Resize to height 48 and width `max(16, round(w*48/h))` with Catmull-Rom. Every crop
     here is upscaled, which matches the measured PIL bicubic.
  3. Order channels **B, G, R**, as in the RapidOCR convention.
  4. Compute `v/255*2-1` and lay it out NCHW.
  5. Right-pad with `0.0` to the batch width.
- Batches: names N=10, cells N=30, header N=3 (tags plus round), each chunked so
  activations stay bounded; the actual width grows to a multiple of 8 when a crop resizes
  wider, so the 720p right-name crops keep their full width instead of being truncated.
  The tests rerun every field per crop (`BatchMode::PerCrop`) to check that padding does
  not change any field's winning candidate on the fixtures.
- Session options:
  - intra-op threads 2 (configurable) and inter-op threads 1;
  - `session.intra_op.allow_spinning = "0"`;
  - memory pattern on (ORT's default): buffers are reused across runs of the same shape.
    Left off, the pruned model's activations are churned through mmap/munmap (about 11k
    minor page faults per read, measured), and that overhead cancels the head's saving;
  - CPU memory arena off.

### 3.2 CTC candidate score (log domain)

```
lp[t][c] = ln(max(p[t][c], 1e-12))
ext = [0, c1, 0, c2, 0, ..., cn, 0]                     # S = 2n+1, 0 = blank
alpha = [-inf]*S; alpha[0] = lp[0][0]; alpha[1] = lp[0][c1]
for t in 1..T-1:
  for s in S-1 down to 0 (or use a copy):
    a = alpha[s]
    if s >= 1: a = lse(a, alpha[s-1])
    if s >= 2 and ext[s] != 0 and ext[s] != ext[s-2]: a = lse(a, alpha[s-2])
    new[s] = a + lp[t][ext[s]]
score = lse(alpha[S-1], alpha[S-2])                      # -inf if T too short
```

- `lse(a,b) = max + ln(1+exp(min-max))`.
- Use only the leading `ceil((width-4)/8)` timesteps of a padded crop (`tensor::timesteps`),
  capped at the model's `T`. `width` is the crop's own resized width,
  `max(16, round(w*48/h))`, before right-padding to the batch width.
- Names: score the name as written, in upper case, and in lower case, and keep the max.
  Skip a variant with a character missing from the class list. A name with no scorable
  variant is not a candidate; log it once.
- `best(lp, lexicon)` returns `(word, score, margin)`. When the lexicon has one entry, the
  margin is `+inf`.

### 3.3 Acceptance rules

| Field | Lexicon | Accept when |
|---|---|---|
| Tag (left and right, read independently) | the two context tags | margin ≥ 6, score/len ≥ −3.0, and left ≠ right. Never infer one side by elimination, the same rule as the tracker's tag reads. |
| Score | the LR plate reader (`vision::plates`) | softmax ≥ 0.90 |
| Round | `{ROUND s, ROUND s+1}` when both scores are read (s = L+R), else `ROUND 1..50` | margin ≥ 3. Otherwise `null`. |
| Name, per row | the side's roster (from its accepted tag). If that tag is unread, use the union of both rosters, and the team comes from the roster entry. | margin ≥ 8, score/len(name) ≥ −3.0. If two rows on a side accept the same name, drop both. |
| K/D/A cell | `"0".."99"` | margin ≥ 4, score ≥ −1.5 |

- A player is emitted only when its name **and** all three cells are accepted. A partial
  K/D/A is dropped rather than emitted as zeros, because the `kda: [i64; 3]` schema cannot
  hold null.
- Output order: left rows 0–4, then right rows 0–4.
- Log every rejected field with its score and margin at debug level so thresholds can be
  tuned.

### 3.4 Canonical output (one JSON object per line on stdout)

```json
{"round":6,"teams":[{"name":"TL","score":3},{"name":"G2","score":2}],
 "players":[{"team":"TL","name":"kamo","kda":[6,3,3]}, ...]}
```

- Teams are listed left then right. An unread tag is `""` and an unread score or round is
  `null`.
- A frame with no table prints `null`.

### 3.5 Context file

```json
{"teams":[{"tag":"TL","players":["kamo","trexx","nAts","Kicks","GSR"]},
          {"tag":"G2","players":["BABYBAY","valyn","JAWGEMO","leaf","trent"]}]}
```

The caller supplies the rosters, which may include substitutes.

### 3.6 Fixtures

`scoreboard/testdata/`:

- `tl-g2-r6.png`: TL vs G2, Abyss, round 6, VOD 2890572310 at 15:54:45, 1080p.
- `tl-g2-context.json` and `tl-g2-r6.expected.json`:
  - round 6, TL 3 – G2 2;
  - TL: kamo 6/3/3, trexx 6/3/1, nAts 3/4/0, Kicks 2/5/2, GSR 1/4/1;
  - G2: BABYBAY 6/4/0, valyn 5/3/1, JAWGEMO 4/4/0, leaf 2/3/3, trent 2/4/2.
- `xlg-kc-r19.jpg`: XLG vs KC, Abyss, round 19, 720p.
- `xlg-kc-context.json` and `xlg-kc-r19.expected.json`:
  - round 19, XLG 6 – KC 12;
  - XLG: Lysoar 13/14/2, happywei 10/15/3, WsLeo 7/14/5, NoMan 7/14/2, Rarga 6/15/2;
  - KC: N4RRATE 24/7/5, avez 14/10/6, SUYGETSU 12/10/3, LewN 11/13/7, dos9 6/8/10.
- `abyss-start.png`: a non-table frame that must print `null`.
- The TL frame downscaled to 1280×720 must pass too; the live tracker decodes at
  `scale=1280:720:flags=fast_bilinear`.

## 4. Implementation

The reader is the `vbt-scoreboard` crate, wired into the tracker in-process behind the
`scoreboard` feature of `vbt`.

```toml
[dependencies]
vbt-vision = { path = "../vision" }
ort = "=2.0.0-rc.13"   # pin: the API changes between RCs. Default features statically link ORT via download-binaries.
image = { version = "0.25", default-features = false, features = ["png", "jpeg"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

The first build needs network access to fetch ORT (cached in `~/.cache/ort.pyke.io`).
After that ORT comes from the cache, and the binary has no `.so` dependency.

```
scoreboard/src/
  lib.rs        model constant, error type, re-exports
  main.rs       the CLI + --serve loop
  model.rs      the ORT session options and the pruned model's class list
  geometry.rs   §2 boxes, presence gate
  tensor.rs     crop -> CHW f32 (image::imageops::resize, FilterType::CatmullRom), BGR, padding
  ctc.rs        classes, log probs, score, best
  reader.rs     pipeline (§3.3) -> Scoreboard
  strip.rs      §6 map-strip name reader (map lexicon -> CTC -> accept rule)
  schema.rs     serde structs for the context (§3.5) and the canonical output (§3.4)
tools/
  prune_head.py slices the full model's head to the scoreboard alphabet (§1)
```

- Assets are embedded with `include_bytes!`:
  - the pruned model, which `scoreboard/build.rs` downloads from the asset host
    (or copies from `VBT_SCOREBOARD_MODEL`) into `OUT_DIR` after checking its sha256;
  - the plate classifier `assets/recognizer/digits_lr.bin` (i32 classes, i32 features, i32 ids, f32 intercepts, f32 coefficients row-major, all
    little-endian), embedded once by `vbt-vision` and shared with the main tracker.
- ort usage (rc.13):
  - Build the session with
    `Session::builder()?.with_intra_threads(n)?.with_inter_threads(1)?.with_intra_op_spinning(false)?.with_memory_pattern(true)?.with_execution_providers([CPU::default().with_arena_allocator(false).build()])?.commit_from_memory(MODEL_BYTES)?`.
  - Read classes from `session.metadata()?.custom("character")`.
  - Create inputs with `TensorRef::from_array_view(([n, 3, 48, w], data))`.
  - Call `session.run(ort::inputs!["x" => t])`, then
    `outputs["fetch_name_0"].try_extract_tensor::<f32>()`.
- Team scores come from the shared `vbt-vision` plate reader, not the model, so the
  scoreboard cannot drift from the tracker's own score reads.

CLI usage:

```
cargo run -p vbt-scoreboard -- --context ctx.json [--threads 2] [--bench N] frame.png...
cargo run -p vbt-scoreboard -- --context ctx.json --serve   # daemon
```

- `--serve` framing:
  - request: 4-byte big-endian length, then PNG or JPEG bytes;
  - response: 4-byte big-endian length, then canonical JSON or `null`.
- On EOF it exits 0. A malformed image produces `null` plus a message on stderr; it never
  panics the loop.
- Errors are `Result<_, Box<dyn Error>>` at the CLI boundary and a small `enum Error` in
  the library.

Wiring into the tracker:

```sh
target/release/vbt --file clip.mp4 --scoreboard-context ctx.json [--scoreboard-threads 2]
```

With a context configured, the tracker runs the table reader at its scoreboard cadence
(every 5 s) and records a `scoreboard_observation` ledger event per read. The model is
also loaded without that flag: the map-strip reader owns a second ORT session from
startup, because it answers the strip read on every map start and join. Measured at the
init boundary (median of 15 release runs on the 12th-gen i5 used for §6.2), the template
recognizer alone costs 0.9 ms and 9.2 MiB peak RSS, and the strip session adds 23 ms and
26 MiB. A strip session or model that fails to load is not fatal: the tracker logs a
warning and falls back to the template strip reader alone - the reader a build without
the `scoreboard` feature uses, with its 0.75 anchor floor - so a library or model problem
degrades the strip read instead of taking the process down. The `--scoreboard-context`
table reader still exits on failure, because that flag is explicit.

## 5. Verification

- `scoreboard/tools/prune_head.py` checks the full model's sha256 before pruning, and stops
  if blank is not class 0, space is not the last class, or a kept character is missing from
  the model's dict. `scoreboard/build.rs` checks the pruned model's sha256.
- `cargo test -p vbt-scoreboard` checks that the canonical JSON for TL 1080p, TL downscaled
  to 720p and XLG–KC 720p equals the `.expected.json` files, that
  `scoreboard/testdata/abyss-start.png` reads `null`, and that reading every field per crop
  (`BatchMode::PerCrop`) picks the same winning candidate as the batched read.
- `vbt-scoreboard --bench N` prints session init and image decode times and the read's p50,
  p95 and max over N runs.
- The reader covers the buy-phase table only, not the in-game Tab table.

## 6. The map strip's name (`strip.rs`)

The same model answers a second question: which map the broadcast's `CURRENT: <MAP>` strip
names. This replaces the per-map median templates as the tracker's primary strip reader
(they remain its fallback) because a name likelihood does not need a harvested asset per
map and does not collapse when a rendering changes.

### 6.1 Geometry and algorithm

- The vision recognizer locates `CURRENT:` (`vision::recognizer::Recognizer::anchor`, the
  same 12x48 template and correlation the templates use) and reports the best correlation
  even below `ANCHOR_THRESHOLD`.
- The name window is `anchor.x + current_w + 2` to `+64`, rows `anchor.y .. anchor.y + 12`
  (64 px covers `FRACTURE`, the longest name, at about 52 px of ink at 720p).
- The window goes through the same `Crop::prepare` as every other field: Catmull-Rom to
  48 px tall, BGR, `v/255*2-1`.
- All 13 map names are scored with `ctc::best` (upper, lower and as-written variants); the
  display names are `MAP_NAMES` in `strip.rs`.
- A read is accepted when the window's anchor correlation is at least `ANCHOR_LOW` **0.60**
  and the winning name beats the runner-up by `MAP_MARGIN` **16.0** nats with
  `score / characters` at least `MAP_PER_CHAR` **-3.0**.
- The lexicon score alone cannot refuse unknown text, because its only competitors are the
  known names: a window holding `SPLINT` still wins Split by 44.6 nats. The window is
  therefore also decoded freely - its first word, cut at a space or `FREE_GAP` **5** blank
  timesteps so the thumbnail icon that follows every name does not join it - and that word
  is scored as a CTC sequence: the read is kept only when it is the winning name's letters
  or explains the window within `FREE_TOLERANCE` **5.0** nats of the name's own likelihood.
- When the model's read is refused, `vision`'s template reader may still name the strip, but
  only if the free decode supports the model's name as above and the name the template picks
  is the one the model ranked first. `summit-12-4.jpg` is the measured case: Summit scores
  5.9 nats over Sunset (under the floor; free decode `5UMMT`, 2.4 nats from Summit) and the
  templates read Summit 0.99 / Sunset 0.76.
- The tracker calls the read on its 5-second map-check schedule.

### 6.2 Measured evidence

Corpus: 4152 real 720p frames from the 22 Aug - 16 Oct 2026 broadcasts. 48 of them are
repository fixtures (the synthetic `strip-unknown-splint.png` below is not counted), 130
are Haven frames, 3797 are harvest frames of Split, Sunset, Breeze, Summit and Ascent, and
186 are sparse samples. Measured against the template asset
`assets/recognizer/strip_templates.json` sha256
`59afea8c4c50c4f95b20fe4bd5ed61a7ae5fceaeae86c526ce3cc8c0260b003c` and model
`PP-OCRv6_rec_pruned.onnx` sha256
`8c857f945502eab54827376bee605c81e01faccbc269538ab3d096a8764500fc`; the reads were produced
by a probe over the same code as the tests.

| | template reader | CTC lexicon reader |
|---|---|---|
| Summit `f_0265` (worst accepted) | 0.093 gap | 20.98 nats |
| Sunset `f_067` (worst of the collision month) | 0.131 gap | 44.71 nats |
| frames it names | 1987 | 2116 (2115 accepted + 1 fallback) |
| frames it names that the other misses | 1 | 130 incl. the 5 unmapped names |

- Separation: over every frame with a located anchor (2148), the margin and
  per-character floors split cleanly: every read they admitted scored at least 20.98
  nats and -2.74 per character, and every candidate they refused topped out at 6.38 nats,
  so the 16.0 floor sits in an empty 14.6-nat band. The free-decode rule refuses one read
  further (the `ABYS5` window below). Per map, the worst accepted margins over all 13
  names are Abyss 43.6, Ascent 44.7, Bind 52.9, Breeze 53.6, Corrode 78.9, Fracture 100.2,
  Haven 40.2, Icebox 70.9, Lotus 21.8, Pearl 63.6, Split 32.1, Summit 21.0, Sunset 27.9.
- Unknown text is refused: `strip-unknown-splint.png` carries a synthetic `SPLINT`
  window: the lexicon scores Split at 44.6 nats with -1.34 per character, over both
  floors, and the free decode reads `SPLINT` 6.62 nats better than Split, so the read is
  refused. All 180 synthetic unknown-word renders (three fonts, four sizes, 15 words)
  refuse. No window changes to a different name.
- Reported collisions: Haven reads 40.2+ nats with no Haven template; on
  `summit-5-12.png` and `summit-13-5.jpg` - the fixtures whose Sunset gap was 0.022 and
  0.024, inside the templates' 0.05 margin - Summit wins by 36.2 and 40.3 nats, and on
  `summit-12-4.jpg` the model scores only 5.9 nats so the templates name it.
- Anchor miss: 130 of the 159 frames with `CURRENT:` between 0.60 and 0.75 are
  named (`summit3/f_0048.png`: anchor 0.700, Summit, 35.0 nats); the 29 refused in that
  band carry no text at all (their best candidate scores at most 6.4 nats), and below 0.60
  nothing is accepted.
- No regressions: the composite names 2116 frames: 1986 of the 1987 the templates
  name, every one of them with the same name, plus 130 frames more. The one template-named
  frame it refuses is `split2/f_049.png` (Abyss at a 42.3-nat margin) whose free decode is
  `ABYS5` 6.12 nats better than Abyss - the conservative side of the unknown-text rule,
  which refuses rather than guesses.
- Cost: 5.7 ms p50 / 6.8 ms p95 per read at 2 intra-op threads, 6.7/7.3 at 1
  (release build, 12th-gen i5, including the anchor and the fallback; the free decode adds
  one more short CTC recursion). The tracker calls it at most twice per 5 s, so the 1 fps
  budget is unaffected. Peak RSS on the same replay: 186.7 MB against 186.8 for the
  template-only build, and 194.4 against 191.7 with the scoreboard table reader also
  loaded; the second session shares the ORT runtime and the model mapping.

### 6.3 Failure modes

- A frame whose name window holds two names - which real broadcasts do not produce, but a
  synthetic double exposure does - can be named by the more likely one; the model scores a
  Lotus/Abyss template blend at Lotus +13.3 nats, under the floor, so the composite returns
  `""`, while a Summit/Sunset blend scores +26.8 and would read Sunset. No such frame
  appears in the corpus.
- An unreadable name degrades to `""`, exactly like the template reader's miss; the tracker
  then names the map from the veto when there is one, and never joins a strip that names
  nothing.
- A map name outside the lexicon is refused by the free decode rather than matched to its
  closest name (the `SPLINT` fixture above). The residual case is a window the model reads
  as an existing name within `FREE_TOLERANCE`: a rendering that far off is not separable
  from OCR noise, so naming it still requires an entry in `strip.rs`'s `MAP_NAMES`.
- The free decode also refuses frames where a word follows the name within five blank
  timesteps: the 2025 Icebox harvest's `ICEBOXNRG`/`ICEBOXNR0` windows (a team code hard
  against the name) and one 4,152-corpus frame, `split2/f_049.png` (`ABYS5`). Refusing is
  the intended side: a missed read only defers to the veto, while a wrong one starts a run
  under the wrong map.
