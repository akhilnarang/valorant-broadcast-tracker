# Shadow scoreboard audit

`--shadow` runs the whole tracker read-only: it never writes the token file or PUTs
`/video/score`, and the publisher stays disabled. Instead it emits visual audit JSONL:
`frame_observation`, `video_accepted` and `map_end` events with independently keyed
`video_score`, plus periodic `scoreboard_observation` lines when the native scoreboard is
enabled. Run it to validate what the video saw before trusting the publishing run.

```sh
cargo build --release -p vbt

# Live shadow. Same environment as the primary for GET /video/context only; the
# tracker never owns the token file or PUTs a score.
target/release/vbt --shadow --events-file shadow.jsonl

# Without --events-file, shadow JSONL goes to stdout (diagnostics to stderr).
# Replay with known competitors and the veto order, without Riot/network writes:
target/release/vbt --shadow --file clip.mp4 --teams G2,TL \
  --map-order Split,Abyss --events-file replay.jsonl

# Inspect a still directly, including scores and independent visual team tags:
target/release/vbt-probe --video frame.png --teams G2,TL
```

Replay ledgers use `match_id: "unknown"`. Replays start map numbering at 1, so replay the
complete series when comparing reference map numbers. The ledger is plain JSONL; keep the
saved VLR/Riot reference responses next to it and compare them offline with your own
tooling.

## Evidence and interpretation

- `frame_observation`: current plate reads/confidences, scene, raw left/right scores
  (null when unreadable), independently keyed `video_score`, Riot sides. Missing plates
  remain missing; they never inherit Riot's guess.
- `video_accepted`: a map start or accepted/corroborated score, with the last
  two-consecutive-frame-confirmed visual sides. Held changes are not accepted.
- `map_end`: final raw score, independently keyed `video_score`, visual sides, Riot
  sides, and the original end `source`/`exact` evidence.

The `score`/`winner` fields keep the tracker's publishing labels; the audit uses only
`video_score` for video results. The audit does not change what the publishing run sends.

A `map_end` whose final round was filled in rather than read says so in its `source`
(`riot`, `next_map`) and `exact` keys. Riot's sides can change after the map start, so the
Riot sides a line carries are the ones at that moment. If no `map_end` exists, the last
`video_accepted` score is partial.

## Reader scope

Team tags are read by crops the tracker learns at runtime and by OCR, each plate on its own and
only ever as one of the two expected competitors; a single readable tag never determines the
other. Expected codes come from the bound veto/context, falling back to Riot's team codes (not its
side order); `--teams` supplies them for replays. The rules, thresholds and when each reader runs
are in [architecture.md](architecture.md#reading-the-team-tags). The deploy's shadow script
keeps its own `--tag-crops` directory, so a shadow run never changes the live writer's crops.

## Reproducible regression checks

```sh
scripts/fetch-testdata.sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

The team-tag fixtures in `testdata/teamplates/` keep only the 80-pixel top band. The tests
learn crops from `abyss-start.png` and read `abyss-round5.png` with them, including with the
plates swapped, at both 1080p and 720p scales; `abyss-replay.png` must not identify broadcast
tags. Tracker coverage verifies a reversed Riot start produces correct visual audit scores and
a map end while making zero calls to an enabled Publisher. The Unix socket test verifies
read-only token ownership, GET auth, and blocked PUTs.

The readers were checked on 7,410 labelled frames of 15 VOD days and four archive broadcasts
(25 teams, gameplay frames at 1080p and 720p). Neither the OCR nor the learned crops named a
plate as the wrong team, including with every other real team code offered as a candidate,
and with the shown team's own crops removed. In a replay of each day at 720p with crops learned
in order from the OCR, TL read 180/180, T1 246/250, KC 29/29 and TYL 125/126, with no wrong
read.
