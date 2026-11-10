# vbt: Valorant broadcast tracker

Reads the live round score of VALORANT esports broadcasts off the stream at one frame per
second and sends each new score to
[val-esports-backend](https://github.com/akhilnarang/val-esports-backend).

The tracker is one native Rust binary. Stream decode, the score state machine, Riot
polling, the series map order and publishing run in one process. All image assets (score
glyphs, the digit classifier, strip and pause templates) and the optional scoreboard model
are embedded at build time. Nothing team-specific ships: team tags are read by OCR and by
tag crops the tracker learns while it runs.

## How it works

```text
streamlink (Twitch) ─► ffmpeg 1 fps 1280x720 RGB ─► vbt ─► val-esports-backend (Unix socket)
                                                      ▲
                                              Riot esports API (teams, sides, map results)
```

Every frame, vbt reads both score plates by template matching, with a small digit
classifier as a second opinion. A score must read the same twice and normally moves one
round at a time. Anything else waits for more evidence. Riot's esports API supplies the
teams and sides; the on-screen team tags only check that the video is this match. The map name comes from the series map order or the on-screen `CURRENT:`
strip. Tech pauses, timeouts and half-time are read from their overlays. When vbt cannot
trust what it sees it publishes `status: error`.

The stream is only opened while Riot lists a live match. `--events-file <path>` appends
every tracked change as one JSON line, so a replay and a live run can be diffed.

[docs/architecture.md](docs/architecture.md) has every rule, threshold and interface.
[docs/shadow-audit.md](docs/shadow-audit.md) covers the read-only visual audit, and
[docs/scoreboard-native.md](docs/scoreboard-native.md) the buy-phase table reader.

## Workspace

```text
vision/      vbt-vision: frames, regions, the exact resampler, plate classifier,
             map-strip anchor and pause recognizer, learned team tag crops
scoreboard/  vbt-scoreboard: readers over the PP-OCRv6 recognition model for the
             buy-phase table, the map strip's name and the team tags
vbt/         the tracker library, the `vbt` binary and `vbt-probe`
assets/      embedded image assets: glyphs, recognizer weights and templates
testdata/    test fixtures; the images are fetched by scripts/fetch-testdata.sh
scripts/     fetch-testdata.sh
deploy/      example systemd unit and launch scripts
docs/        architecture, shadow audit and scoreboard documentation
```

## Build and test

Runs on Linux only. Needs Rust 1.89 or newer (edition 2024), and `streamlink` plus `ffmpeg`
at runtime.

The test images and the scoreboard model are not in git. Both are downloaded from
`files.akhilnarang.dev/cdn/valorant/ocr/v1/`. Fetch the images once before running the tests;
the script needs `curl`, `zstd` and `sha256sum`:

```sh
scripts/fetch-testdata.sh                  # or: scripts/fetch-testdata.sh vbt-testdata-v1.tar.zst
```

`scoreboard/build.rs` downloads `PP-OCRv6_rec_pruned.onnx` on the first build and checks
its sha256. To skip the download, set `VBT_SCOREBOARD_MODEL` to the absolute path of a
local copy.

```sh
cargo test --workspace                     # scoreboard tests fetch the ORT runtime on first build
cargo test -p vbt --no-default-features    # tracker only, no ONNX, no model download
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo build --release -p vbt               # target/release/vbt
```

## Run

Any 720p or 1080p recording of the official VCT broadcast works with `--file`. A replay
uses no network and publishes nothing.

```sh
# Replay a file; append one JSON line per tracked event and print canonical lines.
target/release/vbt --file clip.mp4 --canonical --events-file events.jsonl

# Replay with the veto's series order standing in (--map-order is replay-only):
target/release/vbt --file clip.mp4 --map-order Sunset,Split,Ascent

# Live, publishing over BACKEND_SOCKET with the token in VIDEO_TOKEN_FILE
# (both from the environment or a .env in the working directory):
target/release/vbt

# Shadow: read everything, never write the token or PUT scores; the ledger goes to
# shadow.jsonl (to stdout without --events-file).
target/release/vbt --shadow --events-file shadow.jsonl

# In-process buy-phase scoreboard (needs a context file with two teams and rosters).
target/release/vbt --file clip.mp4 --scoreboard-context ctx.json
```

A `--canonical` line looks like this:

```text
sec=1834 kind=round map=Abyss score=G2=7,TL=5 winner=G2 source=
```

Publishing needs a running val-esports-backend whose `VIDEO_TOKEN_FILE` points at the same file
as the tracker's.

`--source <url>` overrides the stream URLs (default: twitch.tv/valorant,
twitch.tv/valorant_br, then YouTube) and `--verbose` raises the log level. `--glyphs DIR`
overrides the embedded glyph bank at runtime. `--tag-crops DIR` (default `data/tag-crops`)
is where the tracker keeps the team tag crops it learns; a build without the `scoreboard`
feature has no OCR to learn from, so it reads no team tags and never joins a map in
progress.

`vbt-probe` prints what each reader sees, for debugging:

```sh
target/release/vbt-probe --rgb DIR [--teams G2,TL]   # decoded .rgb + .json dumps; --teams adds the OCR tag read
target/release/vbt-probe --video clip.mp4 [--teams G2,TL]
```

## Deployment

[deploy/](deploy/) has a systemd user unit and launch scripts that pin streamlink and
ffmpeg. See [deploy/README.md](deploy/README.md) for the layout and the
`VIDEO_TOKEN_FILE` contract with val-esports-backend.

## Push contract

The tracker writes to val-esports-backend over its Unix socket; the backend's `VideoScore`
schema defines the body.

- `PUT /api/v1/video/score` with `{status, observed_at, map_number, teams: [{code, name, score, side}], pause: {kind, reason, since}}`, on every score change, on a pause starting or ending, and every 30 s. `side` is `blue` for the team on the broadcast's left score plate and `red` for the right. The tracker always reports the mapping it is publishing with, so a side correction changes the payload and is published as a new state even at a tied score. `pause` is present only while the broadcast shows a tech pause, naming its kind, reason and start.
- `status` is `ok` while a map is being read, no suspect score is held, and a read matched within 90 s, including through a detected pause. It is `error` once the map ended, while a suspect hold withholds a change, while a side reversal is pending, while a contender series (consistent reads that disagree with the tracked score) has lasted 20 s with 3 reads, or once no read has matched for 90 s.
- Auth: at startup the tracker writes a random token to `VIDEO_TOKEN_FILE` (mode 600) and sends it as `X-Video-Token`; the backend compares it with that file. A live run sends the same token when it fetches `GET /api/v1/video/context?codes=<riot codes>` for the series map order. A 404 leaves the strip as the name source, and the context is refreshed every minute.

## Known limits

- If Riot initially reports swapped sides, a wrong winner can be published until Riot
  reverses them (3 m 39 s in the worst observed reversal), or for the whole map if Riot
  never reverses a wrong attribution. The correction publishes the corrected score as a
  new state, which replaces the wrong one; it cannot take back the publications already
  sent. A reversal is applied only after Riot has reported it for
  120 s, so corrections land about two minutes after Riot reports them, and the tracker
  reports `error` while one is pending. The tracker never withholds scores at map start
  waiting for a reversal.
- A tracker that starts mid-map joins only after 60 s of corroborated reads (Riot's
  in-progress game with explicit sides, this match's tags in Riot's order, a stable non-zero
  score, and a strip that names the game and agrees with the series map order). Until then
  nothing is published for that map. A strip that names nothing cannot join, and a pause
  overlay resets the series, so a mid-map start around a pause can take up to 60 s of
  unpaused reads to adopt. A team whose tag the OCR cannot read on that broadcast, and has
  no learned crop for, cannot join either.
- Only one stream at a time, so one live match.

## Licence

The code is MIT licensed; see [LICENSE](LICENSE). The scoreboard model is Apache-2.0, and
the broadcast frames and crops belong to Riot Games; see [NOTICE](NOTICE).

vbt isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games
or anyone officially involved in producing or managing Riot Games properties. Riot Games,
and all associated properties are trademarks or registered trademarks of Riot Games, Inc.
