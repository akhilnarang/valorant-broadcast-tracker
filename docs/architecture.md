# Architecture

This document explains what the tracker does, in enough detail to review its rules without reading the
code. The Rust code in `vbt/`, `vision/` and `scoreboard/` is the source of truth. If this document and
the code disagree, the code wins, and this document should be fixed.

## What the tracker does

The tracker watches the official VALORANT esports broadcast and reads the round score off the on-screen
score bar once per second. It sends every new score to val-esports-backend.

The scope is narrow on purpose. The tracker reads one number per team: the rounds won on the current map.
It does not read kills, economy, timers or player names.

Team codes and sides come from Riot's esports API. The tracker reads the on-screen team tags only to check
that a mid-map join and the shadow audit are looking at this match.

The tracker never decides that a match is over. It reports each map's score.

When in doubt, it goes quiet. If it can't read the score, it publishes `status: error`.

## How the pieces fit

```text
                 ┌──────────────────────────── Riot esports API (poll every 30 s)
                 │                               live match, teams, sides per map, game states, maps won
                 ▼
streamlink ──► ffmpeg ──► 1 fps 1280x720 RGB ──► tracker ──► PUT /api/v1/video/score ──► val-esports-backend
(Twitch HLS)   (decode, scale)                   (per frame)   (HTTP over a Unix socket)
```

The runner tries Twitch first and then YouTube, unless `--source` gives other stream URLs.

| Component | Module | Responsibility |
|---|---|---|
| Runner | `vbt/src/main.rs`, `vbt/src/stream.rs` | Opens the stream only while Riot lists a live match, feeds frames, reconnects when the stream ends. |
| Readers | `vision/src/` | Stateless image functions: crop regions, classify the scene, read a score by template matching, classify digits, locate the strip's `CURRENT:` anchor, match team tags against crops learned at runtime. |
| Tracker | `vbt/src/tracker/` | The state machine: maps, scores, confirmation rules, hidden map ends, side corrections, publishing. It accepts an optional series map order: the video context when live, `--map-order` for a replay. |
| Map order | `vbt/src/context.rs` | Fetches the video context (the series' ordered map names) for the live match. |
| Riot | `vbt/src/riot.rs` | Background poller holding the live match's details. |
| Push | `vbt/src/push.rs` | Writes the score to the backend and owns the auth token. |
| Ledger | `vbt/src/ledger.rs` | Writes the JSON-lines event ledger and the canonical output. |
| Scoreboard and strip | `scoreboard/src/`, `vbt/src/scoreboard.rs` | Optional in-process readers over one ONNX model: the buy-phase table (CTC over a fixed lexicon), the strip's map name (CTC over the 13 map names, with the vision templates as its fallback) and the team tags (CTC over the match's two codes, which teaches the learned tag crops). |

## Frames and time

ffmpeg decodes the stream to 1 frame per second at 1280x720, RGB24, written as raw bytes. That is
`1280 × 720 × 3 = 2,764,800` bytes per frame. The filter is
`-vf fps=1,scale=1280:720:flags=fast_bilinear,format=rgb24 -f rawvideo`, and audio is dropped.

Every frame gets a `second`: the number of seconds since the tracker started. It goes up by one per frame,
but it never falls behind the wall clock. After a stall the pipe still holds old frames, so the tracker
sets `second = max(second, now - started)` and skips ahead. That way gaps are measured in real time. When
replaying a file, `second` is simply the frame index.

Every duration in the rules below is in these seconds.

## Where it looks

All regions are measured on a 1920x1080 frame and scaled to the frame's width (`scale = width / 1920`),
with each edge rounded to the nearest pixel. Boxes are `(left, top, right, bottom)`, with right and bottom
exclusive. The boxes live in `vision/src/regions.rs`, except the tag lines, which live with their reader in
`scoreboard/src/tags.rs`.

| Region | Box at 1080p | Used for |
|---|---|---|
| Left score plate | `(784, 4, 858, 56)` | Left team's rounds |
| Right score plate | `(1042, 2, 1118, 56)` | Right team's rounds |
| Left team tag | `(730, 8, 780, 36)` | Learned tag crops |
| Right team tag | `(1130, 8, 1200, 36)` | Learned tag crops |
| Left tag line | `(722, 10, 784, 36)` | Tag OCR: the tag only, without the seed line below it |
| Right tag line | `(1124, 10, 1204, 36)` | Tag OCR |
| Round timer | `(895, 24, 1002, 64)` | Scene detection only (brightness, spike icon) |
| REPLAY banner | `(1612, 996, 1852, 1044)` | Scene detection |
| Pause header | `(218, 818, 623, 881)` | The pause overlay's lower-third phrase |
| Pause card | `(75, 216, 300, 288)` | The half-time card's title line |
| Map strip | `(0, 0, 435, 18)` at 720p, not scaled | `CURRENT: <MAP>` text |

The pause header box was measured on the 2026-09-29 broadcast at 720p as `(145, 545)-(415, 587)`. It is
stored at the 1080p scale the other boxes use.

The broadcast shows Riot's blue team, which defends first, on the left plate, and the red team on the
right.

## Telling scenes apart

Before reading anything, the tracker decides what kind of shot the frame is (`scene_of` in
`vision/src/scene.rs`). It checks in this order:

1. The frame is a replay if more than 25 % of the REPLAY banner's pixels are brighter than 200 (grayscale).
2. The frame is gameplay if either score plate has more than 5 % of its pixels brighter than 195 (a score
   is visible), or the timer region has more than 8 % of its pixels brighter than 195, or the timer region
   has more than 15 % strongly red pixels (`R > 130`, `R > 1.8·G`, `R > 1.8·B`). The red pixels are the
   planted-spike icon.
3. Anything else has no HUD: desk shots, player cameras without the broadcast overlay, pauses, black
   frames.

Intermission footage can trip the timer rules with no score bar on screen at all. Crowd shots, arena
lighting, arena screens and highlight clips all do it. So the gameplay label alone never counts as HUD
evidence later on. The pending-end rule, for example, demands a readable score pair.

A score is read on gameplay frames, and on replay frames whose score plates are visible.

## Reading a score

Each plate is read by template matching first. A logistic-regression classifier, trained on real
score-bar glyphs, gives a second opinion. The two are then combined per plate.

### Template matching

For one plate crop:

1. Convert to 8-bit grayscale and stretch it with an autocontrast that maps the darkest pixel to 0 and the
   brightest to 255.
2. Binarize the stretched image with Otsu's method. If the mean is above 128, invert first, so text is
   always the brighter class. Pixels strictly above the chosen threshold are ink (1). The threshold is the
   level that maximizes between-class variance, scanning levels 0 to 255 in order and keeping the first
   maximum.
3. Find glyphs. Take runs of consecutive columns that contain any ink. Each run's box spans the first to
   the last row with ink in those columns. Drop runs shorter than 30 % of the crop's height.
4. If exactly one box was found, try to split it into two touching digits. A single digit is about 0.72 as
   wide as it is tall, so skip the split if the box is under 20 pixels wide or narrower than
   `1.6 × 0.72 × height`. Otherwise find the column with the least ink between 35 % and 70 % of the box's
   width. Skip the split if that minimum is more than half the busiest column's ink, or if the cut would
   fall within 8 pixels of either edge. Otherwise split there.
5. Drop specks: remove boxes whose ink count is below 30 % of the median box's ink.
6. Reject the read if no boxes remain, or if more than two do.
7. Normalize each glyph. Cut its box from the stretched grayscale image and center it on a black square
   canvas whose side is the box's longer side. Resize to 24 wide by 32 tall with Lanczos resampling, scale
   to 0 to 1, and flatten. Subtract the mean, then divide the mean-subtracted vector by its own L2 norm
   (plus `1e-6`), so it is a true unit vector.
8. Match. Take the dot product with every template vector, normalized the same way. Each digit's score is
   its best match among its templates. The best digit wins.
9. The confidence is `0.6 × (worst digit's best score) + 2 × (smallest lead of a winner over its runner-up
   digit)`, clamped to 0 to 1.

The value is the digits read left to right, so it is at most 99.

### Digit templates

`assets/glyphs/<digit>/*.png` holds 24×32 grayscale crops of real score-bar digits from recorded matches,
one folder per digit. At load time each template gets eight variants in addition to the original:

- six one-pixel shifts with wrap-around: up, down, left, right, up-left and down-right;
- two scalings about the center, by 0.92 and 1.08, using nearest-neighbor index clamping
  (`index = clip((i - n/2) / scale + n/2, 0, n - 1)`).

Every variant is normalized as in step 7. More real crops per digit improve accuracy directly.

### The digit classifier

The classifier is a multinomial logistic regression: a softmax over 10 logits. Its input is the 24x32
glyph canvas from steps 1 to 7, row-major, with each pixel divided by 255. It is trained on harvested,
labelled score-bar glyphs, balanced by class. The weights are float32: `coef` is 10x768, `intercept` has
10 elements, and `classes` has 10. They ship in `assets/recognizer/digits_lr.bin` and are loaded once. The
forward pass is `logits = coef @ x + intercept`, then `softmax(logits - max(logits))`.

For each plate, each glyph's digit is the argmax, and the plate's confidence is the lowest softmax
probability over its glyphs. A plate is read when its confidence is at least 0.90 (`CONFIDENCE` in
`vision/src/plates.rs`). Below that, the classifier reads nothing.

The 0.90 bar sits just above every wrong read that matters. On the four maps of the 26 Sep broadcast,
every wrong but confident read landed at 0.91 to 0.98. Those were frames where a sliding banner briefly
covered the plate and an 8 read as a 9, or where a graphic sat over a player-camera shot. Correct reads
centre at 0.99 and above. Raising the bar to 0.98 removes those wrong reads, but it also silences real
reads with lower confidence, such as the 8 on the GE vs VIT right plate (0.965 to 0.971), and delays real
rounds by minutes. So the bar stays at 0.90, and the tracker's rules deal with the remaining wrong reads
(see "Confirming a score").

### Combining the two readers

A template read is confident when it has a value and its confidence is at least 0.65
(`TEMPLATE_MIN_CONFIDENCE`). This bar was re-tuned when the normalization was corrected to true unit
vectors (see "Implementation exactness notes"). The old normalization was not unit length, which shrank
every confidence, and 0.35 was the tuned bar then. On the corrected scale, correct reads sit at 0.667 and
above (1st percentile), while non-score content such as player-camera portraits reads at 0.46 to 0.63. So
0.65 admits about 98 % of correct plate reads and keeps portraits untrusted.

The confidence bar does not stop real digits from other overlays. Round banners occasionally land exactly
on a plate and read at any confidence up to 1.0. The defence against those is in the tracker: the two-read
confirmation, the hold on suspect changes, and the one-round-at-a-time rule.

The classifier reads both plates, segmenting and classifying each crop directly, when any of these hold:

- the frame's `second` is a multiple of 5 (`RECOGNIZER_EVERY_SECOND`), as a periodic check;
- either template read is not confident while a map is being tracked;
- the read differs from the tracked score, since every change is confirmed by the classifier.

Each plate's value is then chosen like this:

1. Take the template value if it is confident, the read was not a forced periodic or change check, and the
   classifier agreed or read nothing.
2. Otherwise take the classifier's value, if it read one.
3. Otherwise take the template value only if it is confident. An unconfident template read never decides
   alone, because on player-camera shots the plates show agent portraits, which match digits weakly.

A frame's read is valid only if both plates produced a value.

## Following a map

The tracker follows one map at a time, called the active run. A run holds the map name, the Riot match
ID, the map number, the left and right team codes, the score, and a few timestamps. The rules in
`vbt/src/tracker/` decide when a run starts, how its score changes, and when it ends. Most of their
complexity comes from the broadcast cutting away at the wrong moment, and from Riot lagging behind the
picture.

### Starting a map

A run normally starts when the tracker sees a map begin at 0-0. It needs two things before it starts: the
game number, and a name for the map. A map that no source can name is not tracked, and no run is started for
that map.

The checks run on gameplay frames whose read is 0-0, every 5 seconds:

1. The tracker takes Riot's explicit in-progress game number, when Riot gives one.
2. It asks its optional map-order source for that game's name. Live, the source is
   `GET /api/v1/video/context` on val-esports-backend. It returns the series' ordered map names and keeps
   unnamed games as empty entries, so later names stay at their Riot game number.
3. A candidate starts after two 0-0 observations that agree on the game. Its name is the map order's name
   for that game, or the name read from the strip when the order has none.

Riot's in-progress game number decides which game a run is, but it never starts a run by itself. On the
2 Oct 2026 JDG vs T1 broadcast, a run started from Riot alone accepted a pre-game graphic's 1-0 and
rejected the real map for 22 minutes.

Any scheduled observation that is not a 0-0 gameplay read contradicts the pending confirmation and resets
it. So an interrupted sequence like 0-0, 1-0, 0-0 cannot start a map in the middle of a match.

#### Without a map order

Replays without `--map-order`, and live shadow runs without publishing configured, have no
map-order source. The map strip is then the only source of names. It must contain `CURRENT`, and the map
is the word right after it, read by the strip reader described under "Reading the strip name". A candidate
still needs two 0-0 checks in a row that agree on the game: the same name, and the same Riot game number
when both are known.

A strip that names nothing cannot start a run, even when Riot reports a game in progress. So a shadow with
no video context tracks every map the strip can name and starts no run for the rest. A run that already
tracks the candidate's game number is not restarted under a different name.

#### With a map order

A fallback name read from the strip must agree on consecutive observations before it can start a map. A
known map-order name can be confirmed by game number alone. An unnamed map-order entry is not a name, so that game
waits until the strip names the map.

Any change in Riot match ID resets the pending map-start confirmation and the hidden-end state, ends any
pause in progress, and quarantines the active run. The run is kept for the record but never published
against the new match (see "Match changes").

#### Moving to the next game

When the tracker sees a later map-order game while the current run has not ended, it accepts the move
only if the strip has changed from the name read at map start, or if Riot reports that later game in
progress. This stops a wrong strip that never changes, or a guessed fallback number, from advancing the
series.

A start never goes back to a game number at or below the active run's. At Game Changers 2026, Riot flipped
a game from `completed` back to `inProgress`, and the same 0-0 restarted map 3 as map 2, with game 2's
sides.

Riot also lags. It kept listing game N as `inProgress` for 2 to 5 minutes after map N+1's 0-0 was on
screen, so its number alone still pointed at the run's own game, and map N+1 was never started. To handle
this, once the active run has ended or is at match point, a 0-0 whose strip differs from the name read at
the run's start is taken as the next game, `map number + 1`, whatever Riot lists. A strip that names the
run's own game or an earlier veto game never counts as changed, because that is a replay or recap of a
past map. At such a boundary, a run at match point closes as a `next_map` hidden end (see "Ending a map"
for its winner), and the new run takes game N+1's sides. If the strip is unreadable, the number stays
Riot's, and the contender close under "Confirming a score" is the fallback.

A cut can hide more than the map-ending round. A run at 11-11 whose map really ended 13-11 is not at match
point. So a run whose leader has 11 or more rounds is treated as close enough to a win that a changed
strip at a 0-0 also marks the boundary. When the veto is known, the strip must name a later game in it.
With no veto, the changed strip and the usual two agreeing 0-0 checks are the evidence. No winner is known
here, so the run closes as a `next_map` hidden end with an empty `winner` and `exact: false`, its score
unchanged and no series credit. Riot's later credit for that game fills in the winner (see "Ending a
map"). Below 11 rounds, a changed strip still cannot start a game Riot does not list, so a wrong strip
alone never ends a map halfway. The start that closes the map also keeps the next map's opening from
resyncing it.

#### Numbering from the veto

When the veto is known, the strip's position in it gives the game number without Riot. If the strip names
exactly one veto game, and that game is above the number chosen so far, it becomes the number, as long as
no run of this match is still going. That means no run was tracked, the run has ended, or the boundary
rule above holds. This numbers a series whose first map was never tracked, a run that started without a
strip baseline once it has ended, and a map whose previous map went untracked, even while Riot lags. A
map the veto lists twice falls back to the rules above. Without veto names, nothing changes.

#### When the series is decided

Once Riot reports enough map wins for the best-of, the tracker ignores later 0-0 frames, with or without a
map order, rather than invent a decider. That check reads Riot's own map-win tally. The tracker's local
series tally is only a diagnostic.

#### What a start sets

The map number is Riot's explicit in-progress game number, whatever the strip says, unless it lags the
active run or the strip names a later veto game, as described above. Without one, a map name that was
already tracked keeps its number. Otherwise the tracker guesses one past the highest game number already
tracked in this match, so a run first seen at game 2 proposes game 3, never game 2 again.

The sides are Riot's blue and red team codes for that game. If Riot gives no sides yet, the teams are
taken in Riot's listed order. If Riot's match details have not arrived at all (`getEventDetails` has not
succeeded), the fallback is the `LEFT`/`RIGHT` placeholder pair and the run is marked provisional. A
provisional run adopts Riot's real codes as soon as Riot lists them, instead of publishing placeholders
for the rest of the map (see "Correcting sides").

If the previous run never ended and belongs to the same Riot match, it is closed as a hidden map end
confirmed by the next map (see "Ending a map"). The winner is the team Riot credited, if Riot already lists
that game completed. Otherwise it is the team at match point, as a guess. Otherwise there is no winner.

When a map starts, the tracker also snapshots Riot's map-win count for each team (`wins_baseline`). When
Riot later confirms a hidden end, the tracker compares Riot's results with that baseline, so a win from
before this map was observed is never taken as this map's result.

### Joining a map in progress

The tracker normally starts the maps it sees begin. It can also join a map that is already in progress,
for example after a start, a restart or a reconnect in the middle of a map. The join needs strong
evidence, so that a clip, a rogue score or a frame from a return-from-break is never adopted as this
match's state. Until a join lands, nothing is published for the current map.

On the same 5-second scheduled reads that can start a map, the tracker collects a join series of
corroborated mid-map reads. Every read counted in the series must pass all of the checks below. A read
that fails one resets the series, except where a check says the read is only neutral.

1. No pause is on. During a pause the broadcast may show a clip from anywhere. The 10-10 rogue score the
   tracker once followed came from a clip shown during a tech pause. The series is reset on every frame
   that shows a pause overlay, whether the pause is confirmed or still waiting for its second frame,
   independently of the scheduled reads.
2. Riot lists the live match, with a known match ID and an event, and the series is not decided.
3. Riot lists game `n` as `inProgress` and gives explicit sides for it. A joined run therefore never
   carries the provisional `LEFT`/`RIGHT` identity.
4. The game is new: no tracked run has this match ID with a map number at or past `n`.
5. The frame is gameplay with a confirmed score. A replay or an unreadable HUD is not evidence either way.
   The series is kept, but a gap between reads longer than 30 s (`JOIN_GAP_SECONDS`) starts it over.
6. The score is valid, unfinished and not zero. A 0-0 belongs to the ordinary start path.
7. The plates show this match's teams, in Riot's order. The tag readers only ever name the two expected
   team codes, and both plates must read, on the sides Riot lists. This is the main defence against a
   clip from another match. A plate read on the wrong side resets the series. A plate that reads nothing
   is neutral, like an unreadable HUD in check 5: the series is kept but the read is not counted. Before
   a team's crop is learned, the OCR reads a plate like TL only about three times in four, and a reset
   on every miss would rarely leave 8 reads in a row.
8. The strip names the game on this read, and when the veto has a name too, the two agree. An unreadable
   strip or a disagreement resets the series. When both agree, the veto's spelling names the run. When
   the veto entry is empty, the strip's name does.
9. The read continues the series: the same match, game, sides and name, a read at least every 30 s, and
   either the same score or a plausible, rate-limited advance. The series may advance at most one round
   per 45 s since its first read, exactly like the contender.

The strip check matters because the strip is the only independent evidence that the picture belongs to
the game Riot reports. The ordered plates prove which teams are playing and in what order, and
`inProgress` names only the game number, so nothing else confirms the map. A read that names nothing
therefore fails toward silence rather than accept a clip from another map. The model-backed strip reader
names all 13 maps of `MAP_NAMES` without a per-map asset, and all 13 also have a template for the fallback
path. The failures left are a rendering neither reader can read (an unseen HUD revision, a covered strip)
and a read refused as ambiguous.

A series that reaches 8 reads (`JOIN_MIN_READS`) over 60 seconds (`JOIN_MIN_SECONDS`) opens the run at the
on-screen score. It emits a `map_join` event instead of `map_start` and publishes `status: ok` in that same
frame. A later pause publishes along with it, because the run now exists to publish through.

After that, the adopted score gets no special treatment. A read that contradicts it goes through the
ordinary contender path, and a rerun, a side reversal or a move to the next game is handled by the rules
above. A match whose tags neither the OCR nor a learned crop can read never joins. Neither does a map
whose strip names nothing on every read. Neither gets a run for that map. A build without the
`scoreboard` feature has no OCR to learn crops from, so it reads no tags and never joins.

### Reading the team tags

Nothing team-specific ships. Two readers name each tag plate as one of the match's two expected codes
(the veto context's, else Riot's), each plate on its own, so one readable plate never determines the
other by elimination:

- Learned crops (`vision/src/learned.rs`). The white tag foreground is matched by shape and aspect ratio
  against crops the tracker learned earlier. A code is named when its best crop scores at least 0.93
  (`MATCH`) and beats the best crop of every other learned code by at least 0.08 (`MARGIN`). With the
  shown team's own crops removed, no other code matched over about 8,000 labelled gameplay plates at any
  setting from 0.90/0.05 upwards; the best wrong-code score was 0.878, and TL and T1 cross at 0.814 at
  most. Own crops read TL 240/240, T1 246/247, KC 98/98 and GE 103/109 at 720p. Crops are kept per frame
  height, because they do not transfer across resolutions: a 1080p crop scores 0.84 at the 5th
  percentile on the same team's 720p plate.
- OCR (`scoreboard/src/tags.rs`). The tag line is run through the scoreboard model, both codes are scored
  as CTC lexicon entries, and the line is also decoded freely. The better code is accepted only when the
  free decode spells it exactly (upper-cased, folding only `6` to G, `0` to O and `Ü` to U), it leads the
  other code by at least 3 nats (`TAG_MARGIN`) and it scores at least -1.5 per character
  (`TAG_PER_CHAR`). A 5-nat tolerance on the free decode, as the strip uses, accepted GEN on SEN, T1 on
  TL and VT on VIT, so there is none, and `L` is not folded to `1` because TL would then spell T1. TL
  against TYL legitimately leads by about 5 nats, and false accepts were the same at every margin from 0
  to 10. The worst accepted read on the 2026 HUD scored -1.04 per character; wrong codes that passed the
  spelling check scored -2.90 or less. The condensed HUD font makes the model drop a narrow `T`, so TL
  decodes as `L` and T1 as `1`; a side that is not accepted on the raw crop is retried on a gray crop
  stretched 1.3 and then 1.8 times wide and edge-padded by 8 px. At 720p that lifts TL from 7/240 to
  181/240, T1 from 1/247 to 240/248, KC from 87/98 to 94/98 and TYL from 21/125 to 125/125, with no
  wrong-team accept on the labelled frames of 23 teams.

When the OCR reads both expected codes on the same gameplay frame, the tracker learns a crop of each
plate the crops did not read. It keeps 3 per code and frame height (`KEEP`), and a code that already has
3 loses its oldest, so crops of a HUD that changed age out. They are saved as `CODE_<height>_<n>.png` in
`--tag-crops` (default `data/tag-crops`), through a temporary file so a crash never leaves a torn crop;
a file that does not decode is skipped at startup, and a failed save keeps learning in memory only.
Requiring both tags keeps single-side noise, such as a `T1` on a caster camera, out of the store.

The crops are matched on every gameplay frame: the visual sides, `plates_agree` and the shadow
`video_score` need a read per frame. The OCR runs only when the crops leave a plate unread and the frame
needs tags: while an expected code has no crop at the frame's height, while the visual sides are not yet
settled, on every 5-second read (the reads a join counts), and on a read that jumps the run by more than
one round. It always reads both plates, so the frame can teach the crops. If the model cannot load, the
tracker logs a warning and reads tags from the crops it already has.

### Reading the strip name

The strip at the top left of the broadcast says `CURRENT: <MAP>`. With the `scoreboard` build feature,
which is on by default, the name is read by a small model in `scoreboard/src/strip.rs`. Without it, the
name is read by per-map median templates in `vision/src/recognizer.rs`. The template reader also stays in
place as the model reader's fallback.

#### How the model reads it

The vision recognizer first finds `CURRENT:` by normalized correlation, searching rows 0 to 3, with an
`ANCHOR_THRESHOLD` of 0.75. The name window starts 2 px (`NAME_GAP`) after the anchor template's own width,
and is 12 rows from the anchor's row and 64 px wide. The model is `PP-OCRv6_rec_pruned.onnx`, the same one
the buy-phase table uses. It resizes the window to 48 px tall and emits a softmax per timestep. Each of
the 13 map names is scored as a CTC label sequence over those timesteps. Nothing is stored per map, so a
map can render however it likes.

The best name is accepted only when it passes three checks:

1. It leads the runner-up by a log-likelihood margin of at least 16 nats (`MAP_MARGIN` in
   `scoreboard/src/strip.rs`).
2. It scores at least -3.0 per character (`MAP_PER_CHAR`).
3. A free decode of the window agrees that the window's first word really is that name.

The third check is needed because unknown text cannot fail the margin: the only competitors are known
names, and `SPLINT` scores Split at 44.6 nats. So the window is also decoded freely, and the read is kept
only when that word is the winning name's letters, or explains the window within 5.0 nats
(`FREE_TOLERANCE`) of the name's own likelihood. The free decode ends at a space or after 5 blank
timesteps (`FREE_GAP`), so the thumbnail icon after every name is not read as part of it. All 180
synthetic unknown-word renders are refused, including the `SPLINT` fixture.

These floors were set on a corpus of 4152 frames from four broadcasts, all at 720p. Every read the floors
admitted scored at least 20.98 nats and -2.74 per character. Every candidate they refused scored at most
6.38 nats. The floor sits inside a 14.6-nat empty band.

#### Dim anchors

A `CURRENT:` match below 0.75 is still usable when the name separates clearly. Down to 0.60
(`ANCHOR_LOW`), the same accept rule applies. These are the dimmer 720p renders that the template reader
refuses. 16 Summit strips on the 10 Oct broadcast sat at 0.700. The model names 130 of the corpus's 159
frames between 0.60 and 0.75, and the frames it refuses in that band carry no text at all: their best
candidate tops out at 6.4 nats. Below 0.60, nothing is accepted.

#### Falling back to the templates

When the model's read is under the margin floor, the template read is kept only if the model also ranked
that same name first and the free decode supports it. On the corpus this fires for exactly one case, the
oldest Summit rendering (`summit-12-4.jpg`, 5.9 nats, with Summit at 0.99 and Sunset at 0.76 in the templates). It can never bring
back the missing-template failure, because a template name that the model did not rank first is refused.

#### Startup and cost

The strip reader's ORT session loads with the process, even when `--scoreboard-context` is not passed.
Measured at the init boundary, the template recognizer alone costs 0.9 ms and 9.2 MiB peak RSS, and the
strip session adds 23 ms and 26 MiB (median of 15 release runs). If the session or model cannot load, the
tracker logs a warning and falls back to the template reader alone instead of exiting.

A strip read costs 5.7 ms p50 and 6.8 ms p95 at two threads, and 6.7 ms and 7.3 ms at one thread (release
build, 12th-gen i5, covering the anchor, the fallback and the model). The tracker calls it at most twice
per 5 s, so the 1 fps budget is not affected. Peak RSS for the same replay is 186.7 MB against the
template reader's 186.8 MB (one thread; ffmpeg and the frame buffers dominate it). With the scoreboard
table reader also loaded it is 194.4 MB against 191.7 MB. The second session shares the ORT runtime and
the 2.2 MB model mapping.

#### The template reader

The template reader relies on the strip's font and positions being fixed. It slides the stored `CURRENT:`
anchor (12x48 grayscale) over the strip by normalized correlation, and the best position at 0.75 or more
is the anchor. The map is the window after it, matched against one stored 12x50 template per map. The
best map needs a correlation of at least 0.55 (`MAP_THRESHOLD`) and must beat the runner-up by at least
0.05 (`MAP_MARGIN` in `vision/src/recognizer.rs`).

`assets/recognizer/strip_templates.json` holds the anchor and one template for each of the 13 names in
`MAP_NAMES`. Each template is the per-pixel median of the fixed window over readable 0-0 frames, so
compression noise never reaches the template. The templates came from these broadcasts:

- Abyss, Ascent, Lotus and Summit came from the 26 Sep broadcast.
- Haven came from map 3 of the 16 Oct VIT vs LOUD broadcast.
- Split, Sunset and Breeze were added next.
- Summit was then refreshed from map 2 of the same VIT vs LOUD broadcast. The new Sunset template had
  brought the 26 Sep Summit template to 0.855 against Sunset's 0.833 on current-production Summit strips.
  That 0.022 gap is inside the 0.05 runner-up margin, so those strips read empty and a real Summit could
  not join.
- The five names outside the current rotation (Bind, Corrode, Fracture, Icebox, Pearl) are not on any
  Twitch VOD still inside its retention window. They were harvested from the YouTube archives that the VLR
  match pages link to, each on 62 to 112 confirmed 0-0 frames: Bind from map 1 of VCT 2026 Americas
  Stage 1 LOUD vs LEVIATÁN (11 May, 361 s), Corrode from map 1 of the Masters Santiago 2026 grand final
  (15 Mar, 476 s), Fracture from map 1 of the Masters London 2026 grand final (21 Jun, 128 s), Pearl from
  map 1 of the Masters London 2026 VIT vs FUT swiss match (8 Jun, 136 s), and Icebox from map 2 of VCT 2025
  Americas Stage 2 SEN vs NRG (31 Aug 2025, 5 s).

Bind, Corrode, Fracture and Pearl read on current production. The Icebox template comes from the 2025 HUD,
so it was checked on an independent 2025 broadcast, the Masters Toronto 2025 grand final, rather than a
2026 one.

A missing template lets a similar name misread. With no Sunset template, a `CURRENT: SUNSET` strip
correlated 0.67 to 0.73 with the Summit template and started a phantom map under the wrong name, although
the scores still tracked. With no Haven template, a Haven strip correlated 0.58 to 0.72 with Abyss, which
named all 15 live 0-0 checks of the 16 Oct map 3 `Abyss` until the template landed. No threshold can
separate a similar-name misread from a true read: the Abyss/Haven collision scored 0.58 to 0.72, the same
range a true Abyss strip sat in.

Summit and Sunset look alike in both directions, and that stays true with both templates in place, so
only the margin separates them. The refreshed Summit template scores 0.87 to 1.00 on a true Summit strip
against Sunset's 0.73 to 0.86 (worst gap 0.12). The Sunset template scores 0.83 to 1.00 on a true Sunset
strip against Summit's 0.72 to 0.86 (worst gap 0.09). With all 13 names covered, the failure that remains
is a rendering no template reads, such as an unseen HUD revision or a covered strip.

Each template is trimmed at its trailing separator before matching. The harvest's fixed window had caught
the emblem that the strip draws after the map name (`CURRENT: ABYSS <emblem>`), which the plain ticker's
`CURRENT: <MAP>` entry does not have. With those columns included, the 29 Sep ticker's name correlated
0.42 to 0.64 with its own template and never beat the runner-up by the margin, so the strip-only path
could not name any map. The name alone correlates 0.78 to 0.89 and clears the margin on every gameplay
frame of that broadcast. With the trim and the margin, the reader names the shared
`prx-g2-lotus-ticker.jpg` fixture `Lotus`. Untrimmed, it read `Ascent` there, and while Sunset had no
template, a `CURRENT: SUNSET` strip misread as Summit.

### Match changes

When Riot's match ID changes, the tracker treats it as a new match. The new match may start at game one
even if the previous match also ended on game one. The change clears pending map-start and hidden-end
confirmation, and ends any pause in progress (`pause_end` with source `match_change`). Starting the new
match does not close the previous match's map as a `next_map` hidden end.

The previous run is quarantined. It is kept for the record, but it is never published while its match ID
differs from Riot's live match, and the new match's sides cannot relabel it.

### Confirming a score

A changed read is only accepted once it is confirmed. On every frame where the active run has not ended:

1. If the read is missing or equals the tracked score, there is no candidate.
2. If the read equals the previous frame's candidate, it is confirmed (two identical reads in a row) and
   recorded.
3. Otherwise the read becomes the new candidate.

#### Suspect reads and holds

A candidate is suspect when its first read came right after a frame where the plate was visible but
unreadable. A partly covered plate looks exactly like this: on the next frame, the covered digit wears
another digit's shape.

A suspect change is recorded like any other, but its publication is held back. It is published once a
periodic recognizer check (a frame whose `second` is a multiple of 5), at least 5 seconds after the
change was recorded, reads the new score again. A different suspect score restarts the hold window and
keeps the last corroborated score. While a score is held, the tracker publishes that earlier score with
`status: error`. At match point, the map end and
the series tally also wait for the same corroborating check, so a suspect match-point score cannot end
the map until it is corroborated or rolled back.

The hold covers what the confidence bar cannot. When the Lotus banner covered an 8, it read as a 9 at
0.969 to 0.972, which no per-frame signal can tell apart from a real 9. The two-read confirmation alone
would have published a false 9-3 round 152 s early. The hold costs nothing for changes that follow a clean read or a
hidden bar. It costs a few seconds for the four real rounds per broadcast whose first appearance
follows an unreadable plate.

A held score is rolled back when two confirmed reads return exactly to the score from before the hold.
The hold is cleared, the timestamps are restored, a `score_rollback` event is emitted with
`source: confirmed_held_score`, and later real rounds advance normally.

The event stream and JSON ledger mark held rounds and their corroboration separately. Rejected reads
include the observed pair next to the score that was kept. Together this keeps the evidence needed to
explain a withheld or rejected score in the replay record.

#### Recording a confirmed score

A confirmed score is recorded as one of the following.

A rollback, when a score is held and the read returns exactly to the score from before the hold. This
clears the hold instead of rejecting the move down.

A round, when the total went up by exactly one. The team whose score rose won it.

A sync, when the total went up by more than one and the gain is within `1 + unpaused hidden seconds / 45`.
That is one round, plus one per 45 seconds (`MIN_ROUND_SECONDS`) of play since the score bar was last on
screen with no change pending. Any frame with the plate on screen restarts this clock, even when it is
unreadable or its read was rejected. So a sync only covers time when the score bar was really gone, and a
steady misread that stays on screen can never pass as one. Pause seconds do not count as play, so a tech
pause cannot buy the time for a jump. The map's team plates must also agree. They are read on gameplay
frames, and a jump is rejected if this map's plates have been readable before but this frame's are not.
The target score must also be possible: regulation to 13, overtime won by two.

A reject, in every other case: a team's score went down, the target score cannot happen on a map, the jump
outran the play time, or the plates disagree. This is almost always a misread.

A resync, when rejected reads build up a consistent series that then replaces the run's score. Read
errors on a highlight that cuts away can spike the score for minutes, and this series is the way back. It
is also the only way up after rounds pass behind an unreadable plate or rejected reads, since those stop
the sync's clock from running. The next section describes it.

#### Recovering through a contender series

Rejected reads are collected into a `Contender`, which records the first score, the latest score, when
the series started, the last read, the read count, and the rounds advanced. A read extends the series if
it matches `latest`, or advances it by a plausible amount: at most one round per 45 seconds between
reads. On top of that, the series as a whole may advance at most one round per 45 seconds of unpaused play
since its first read, so neither a pause nor wall time can ratchet it. Any other read starts a new
series. A read of the run's own score cancels it. A series with no supporting read for 180 seconds
(`CONTENDER_GAP_SECONDS`) starts over.

Once a series is at least 60 seconds old (`CONTENDER_MIN_SECONDS`) with at least 8 reads
(`CONTENDER_MIN_READS`), it replaces the run's score:

- immediately if it advanced, with `source: contender_series`;
- after 120 seconds of holding still (`CONTENDER_STABLE_SECONDS`), with `source: contender_stable`.

It never replaces the score while a pause is on. A series that is ahead of the run's score must also fit a
jump allowance, so `contender_stable` cannot adopt a rogue jump. That allowance is
`1 + unpaused seconds / 45` since the run's score was last read back or accepted. Unreadable and rejected
reads do not restart that clock, so rounds played while the plate could not be trusted still count toward
the recovery.

A resync clears any hold, emits a `resync` event carrying the replaced score as `previous_score`, and ends
the map if the new score completes it.

Some series are not corrections but the next map's opening. This applies to a series that opened at 0-0
and earned its place while the run was at match point or had a side on 11 or more (the strip rule's late
threshold, for a cut that hid two rounds), but holds at most half the run's rounds. Here the map-ending
round was never read, Riot has not moved on, and no readable strip started the next map. Instead of a
resync, the map closes as a `next_map` hidden end, with the next-map winner rules under "Ending a map".
There is no winner when no side is at match point. The next game number then starts at the series' latest
score, when the veto names that game and the series is not decided.

A series that opened mid-map is a correction and resyncs. Examples are a false 12-2 read back as 2-2, or
11-5 read back as 4-4.

A series during which a 0-0 map check read the strip of the run's own game or an earlier veto game, a
replay or recap of a past map's opening, neither closes the map nor resyncs it. This counts from the
series' first read, even while that read was still unconfirmed. If the last strip name read at a 0-0 is
that veto name, the new run keeps it as its start strip, so the next boundary has a baseline. With no veto
name the map only closes, since an unnamed map is never started. The next one can only start through the
mid-map join.

Once a series is 20 seconds old (`CONTENDER_DOUBT_SECONDS`) with 3 reads (`CONTENDER_DOUBT_READS`),
publication switches to `status: error`, because the run's own score may be the wrong one. It stays
`error` until the resync happens or the series is contradicted.

#### Timestamps a run keeps

A run keeps three timestamps:

1. The last second the score bar was on screen with no pending candidate. This is the sync's allowance.
2. The last second the run's score was read back, applied, rolled back or resynced. This is the
   contender's allowance.
3. The last second a read matched the tracked score, used for health.

The first two each store the pause total at that moment, so pause seconds can be subtracted.

### Ending a map

A map normally ends as soon as a recorded score has one team at 13 or more with a two-round lead. This
also covers overtime. The harder cases come from the broadcast cutting away before the final score is
read, and they are handled below.

#### A map-winning read seen only once

A deciding read can be on screen for only about a second, so the two consecutive 1 fps reads that
normally confirm a change may never both land inside it. So when a gameplay read is a one-round,
map-winning score for the run's match-point leader, the run remembers it. That means the leader reaches 13
or more, with a margin of two, on a valid map score, with one round gained.

The first later frame on which the score bar has actually disappeared adopts the remembered read once,
through the ordinary `round`/`map_end` path, if it is no more than 20 s old (`PENDING_END_SECONDS`). Any
later readable read that is not the remembered one drops it, including a return to the run's own score.
So do a rollback and a resync that replaces the run's score.

If the remembered read followed a plate that was visible but unreadable, it keeps the suspect flag through
adoption. The HUD disappearing never turns suspect evidence clean: the round stays held, and the map waits
for corroboration or for Riot.

One risk remains. A single false 13-x read on the very last gameplay frame can still end the map a round
early. The match-point, valid-score, margin and one-round-gain checks, plus the reset on contradiction,
are the only guards, because no further read is possible once the HUD has left.

#### A hidden final score

Series-deciding rounds are often cut away from before the final score shows. A run is at match point when
one team is on 12 and the other on 11 or less, or, in overtime, one team leads by one at 12 or more. When
the run is at match point and no gameplay frame with a readable score pair has been seen for 20 seconds,
a pending end is armed.

Only a gameplay frame with a readable score pair clears the pending end, and only such a frame counts as
the HUD returning. Its pair must also be at or above the run's score on both sides. A pair below it on
either side is the next map's opening, a replay or another match's HUD, not this map continuing. At Game
Changers 2026, the next map's 0-0 cleared every armed end before Riot completed the game. Such a read now
feeds the contender instead. It leaves an armed end armed, does not restart the 20 s clock, and lets the
Riot check below run on that frame.

Frames without a readable pair have no effect on the pending end. A clip from which the broadcast cuts
back into the arena trips the scene timer rules without showing a score bar. On VIT vs LOUD map 2, this
happened for 114 of the 435 intermission seconds, in clusters of up to 28 s. Those clips cleared and
re-armed the end five times, which could swallow Riot's completion inside a gap. So frames the classifier
calls gameplay but that have no readable pair neither clear an armed end nor reset the 20 s clock. The
single-read adoption above still acts only on frames where the HUD is actually gone.

While the end is armed, the tracker checks Riot. When Riot marks the game completed and exactly one of the
two teams' map-win counts has gone up, that team won. The final round is filled in only if the winner was
the team at match point. Otherwise the last read score is kept.

#### The next map starts first

The previous map also ends when the next map starts, by its strip or through the contender, before the
previous one ended. If Riot already lists the previous game completed, and exactly one team gained a map
win since that map's own series baseline, that team won. The final round is filled in (`exact: true`) only
if it was the team at match point.

Otherwise the team at match point is credited as a guess (`exact: false`, score unchanged). That guess can
be wrong: a team on 12 can still lose. At Game Changers 2026, M8 sat on 12 on Sunset against KRU and lost
the map 12-14. With neither Riot's credit nor a team at match point, the map closes with no winner.

#### Riot's late credit

Riot credits game N some minutes after map N+1 can start. So map N+1's series baseline, taken at its
start, can miss game N's win. When different teams win the two maps, map N+1's own end then sees two teams
gain and finds no winner.

Riot's game list carries no winner per game. `games[]` has only `number`, `state`, `teams[].id/side` and
`vods`. So once Riot lists the game before the active run's as `completed`, and the run's own game is not,
the run's baseline takes Riot's current tally, once. If exactly one team gained exactly one win since the
baseline, that win belongs to game N:

- A map N that closed with no winner gets it as a second `map_end` with `source: riot`, `exact: false` and
  its score unchanged, and the series tally credits it.
- A map N whose guessed winner was the other team is corrected the same way, and the tally moves the win
  from the guessed team to Riot's.
- A guess that Riot confirms is never re-emitted, and neither is a map N whose winner was read or
  confirmed by Riot.

A closed map can stay the active run because no next map opened (there was no veto name). It settles
against its own baseline instead, once Riot lists its game completed. The next map's start settles it
first, so the next map's baseline cannot absorb that win.

### Correcting sides

Riot lists placeholder sides for games that haven't started, and it may reverse them minutes into a map.
In the live G2 vs PRX incident it reversed its starting sides 3 m 39 s in. On the NRG vs KC maps it never
reversed them. The API has no flag that says a side assignment is final, and waiting for a reversal that
often never comes would silence the tracker for the whole map. So the tracker uses the sides as soon as it
reads them, and corrects them later if needed.

A map start takes Riot's explicit sides for the game, or the teams in Riot's listed order when Riot gives
none. It publishes from the first round with every team's `side`. Nothing is held back while a correction
might still arrive.

A run that started before `getEventDetails` succeeded carries the `LEFT`/`RIGHT` placeholders, which are not
real team codes. On every frame while the run is provisional, the tracker tries to adopt Riot's
explicit sides for the game, or Riot's listed teams once the details exist. When it can, it replaces the
placeholders, relabels any held score, and logs the same `sides_corrected` line a reversal logs, which
forces the next publication to carry the real codes. This is never delayed: it happens on the first frame
after Riot's details exist. A replay has no Riot identity to adopt, so it keeps `LEFT`/`RIGHT`.

#### Reversals

Riot's responses can be cached for up to 120 s (`max-age=60`, `stale-while-revalidate=60`). On the 2 Oct
2026 Abyss broadcast, one game's sides flipped three times, 90 s apart. So a reversal is applied only when
Riot's latest successful `getEventDetails` reports the exact reverse of the run's sides on every frame for
120 seconds (`REVERSAL_STABLE_SECONDS`):

1. The first such frame logs `side reversal pending` and records the second.
2. Any frame whose sides are missing, or are not the exact reverse, abandons the pending reversal and logs
   `side reversal abandoned`. A later reversal starts a fresh window.
3. When the window has passed, the tracker swaps its team codes, relabels any held (suspect) score to the
   corrected teams, and logs `sides_corrected` with how long the reversal was pending. The score by side
   is unchanged. Only the labels move.

While a reversal is pending, either attribution may be wrong. So the tracker repeats the last publication
with `status: error` until the reversal settles. Corrections therefore
land `REVERSAL_STABLE_SECONDS` later than Riot reports them. On the 1 Oct 2026 broadcast, the delay moved
corrections from 96 s to 216 s and from 154 s to 274 s. On the 2 Oct Abyss broadcast, one flip at about
382 s replaced the three that immediate swaps would have made.

This is an accepted tradeoff. A reversal still pending when the map ends is never applied. Between the map start and Riot's reversal, a wrong winner can be published. If
Riot never reverses a wrong initial attribution, the wrong winner is published for the whole map. The
correction publishes the corrected score as a new state, which replaces the wrong one, but it cannot take
back the publications already sent.

The reversal window is the only timer here, and it exists only to relabel a run that Riot reverses. A map
whose sides never change publishes normally for its whole length. A replay has no Riot to correct its
sides, so it keeps the sides it read and logs its rounds as before.

#### Retracting a wrong tie

When the teams are tied, the per-team score map can look identical before and after a correction. So the
sides are part of the published score. The tracker treats a changed broadcast side as a changed
publication, so it publishes a corrected tie at once as a new state, which retracts the wrong tie. The tracker applies
the correction before it publishes for the same frame.

#### What the tracker logs about Riot's sides

Riot's first sides are not always its final ones. On the 29 Sep to 1 Oct 2026 broadcasts, it reversed a
game's sides 42 to 218 s into the map in 5 of 15 map starts (maps 1 to 3, always a swap of the same two
teams). Nobody knows yet whether Riot's game state marks the point after which sides are final. To find
out whether `inProgress` means final sides, the tracker logs Riot's `state` and explicit sides for the game
at every map start and whenever they change:

- the `map start` log line's `riot` attribute;
- a `riot game` log line on each change;
- the `riot_state`, `riot_blue` and `riot_red` keys on the `map_start`, `map_join` and `sides_corrected`
  ledger events.

The explicit sides drive only the correction rules above. Riot's `state` is read in exactly three other
places: a mid-map join requires the game to be `inProgress` (see "Joining a map in progress"), a hidden
map end waits for `completed`, and the late-credit rule under "Ending a map" waits for the previous game's
`completed`. The state never starts a run by itself, and otherwise publishing does not depend on it.

## Pause state

During a tech pause the broadcast shows a lower third with a phrase on a dark panel: `TECH PAUSE`, with an
optional ` - <REASON>` suffix. The tracker reads that phrase, holds the pause state, and keeps the map
healthy through the HUD gap the pause explains. Timeouts and half-time are read the same way.

### Reading the phrase

The tracker crops the phrase's own region: the lower-third header band, or the title line of the
half-time card. It binarizes the crop at luma above 150 (`PAUSE_MASK_LEVEL`). This is a fixed level, not
Otsu, because the phrase's shape is what has to survive another production's gamma. It then slides one
tight phrase template over the crop, using the same normalized correlation the map strip uses. A phrase
matches at 0.60 (`PAUSE_THRESHOLD`) with a 0.05 lead over the runner-up (`PAUSE_MARGIN`), the same margin
rule the strip's names use.

The templates live in `assets/recognizer/pause_templates.json`. Each holds the phrase's text, its kind, its
reason, its region (the header when empty, `card` for the full-screen card), and its binarized rows,
harvested from a real frame. A template for a full string carries the reason. A template for the bare
phrase also matches a header with a reason the asset does not know. Then only the phrase is reported, so
an unseen reason degrades to no reason rather than a miss.

### Starting and ending a pause

Two matched frames in a row start a pause, so a single transitional frame cannot. Two unmatched frames in
a row with the gameplay HUD visible end it. The overlay's absence alone must not end a pause: in the
measured pause, the overlay vanished at t=2639 but play only returned at t=2647.

While paused, the map's `status` stays `ok` across the HUD gap, instead of the 90-second silence rule
turning it to `error`. The hidden-map-end rule is switched off while a pause is on. Without that, a HUD gap
at match point during a pause would arm a false map end.

### What was measured

On the 2026-09-29 broadcast's 79 pause frames, the phrase scored 0.726 at minimum and 0.787 at the median.
Across 2973 frames of gameplay, ROUND WIN, replays, player cameras, crowd, sponsor and transition shots,
the best score was 0.583, and none reached the 0.60 threshold (`p99` 0.330). That best non-matching frame
is the pause overlay's own first second, one frame before its phrase cleared the threshold. Among frames
the pause ground truth does not call a pause, the best was 0.351.

The map strip's reader does not gate the pause. `scene_of` classifies 3 of those 79 pause frames as
gameplay (red arena lighting) and 21 frames before the overlay as replay, so the pause is judged on the
phrase alone.

Templates are specific to a production, just like the strip's (0.67 to 0.73 across days). The Shanghai
template scored 0.754 against the uploaded PRX vs G2 broadcast's `TECH PAUSE - GEAR` header. It scored
0.028 against that broadcast's ROUND WIN card, 0.037 against a `WIN CONDITION` sponsor card and 0.032
against a gameplay header. So a pause from another production may have its phrase read but not its
reason.

The `TIMEOUT` and `HALFTIME` kinds are measured too. The Champions `TIMEOUT - TEAM VITALITY` frame reads
1.000 on its harvested lower-third template (`timeout-8385.jpg`), and the 2026-09-26 `HALFTIME` frame reads
1.000 (`halftime-6000.jpg`).

The half-time break's full-screen card draws its title line in a region of its own. The pause card box is
`(75, 216, 300, 288)` at 1080p, which is `(50, 144)-(200, 192)` at 720p. It is a band rather than a tight
box, because broadcasts place the card's title at different heights (rows 152-167 and 165-180 at 720p have
both been seen), and the 25-row template slides through it. The card is read by the `HALFTIME` template
whose region is `card`. The card's three measured frames read 0.878 to 1.000 (`lotus-half-time.jpg`,
`halftime-card-2790.jpg`, `halftime-card-5955.jpg`). The countdown card, whose line reads
`CHAMPIONS SHANGHAI`, stays at 0.460.

Across the 480-frame false-positive sweep, no phrase reached the 0.60 threshold on a frame that is not its
own. The best negative per template was 0.498 (`HALFTIME` against the timeout frame), 0.460 (the card
template against the countdown card), 0.385 (tech pause against a live frame) and 0.366 (timeout against a
tech pause).

Harvesting a production's phrase is a development step, not something that happens at runtime.

## Event ledger

With `--events-file <path>`, a run appends every tracked change to a JSON-lines file. Each line is one
JSON object. The file is append-only and written unbuffered, so a lifecycle line reaches disk before a
reconnect can stall. Replays and live runs write the same format, so the two can be diffed line by line.

Every line carries these keys. A key the line's kind does not know gets a zero value.

| Key | Type | Meaning |
| --- | --- | --- |
| `at` | string | wall-clock time, RFC3339, UTC |
| `second` | int | tracker-relative second the change was seen |
| `kind` | string | event kind, snake_case (vocabulary below) |
| `map` | string | map name, `""` when unknown |
| `map_number` | int | game number in the series, `0` when unknown |
| `match_id` | string | Riot match id, `""` when unknown |
| `score` | object or null | current score keyed by team code, `null` when no score applies |
| `winner` | string | winning team code, `""` when the event has none (a sync, a reject) |
| `source` | string | what settled the event (`normal`, `riot`, `next_map`, `overlay_gone`, `match_change`, `confirmed_held_score`, `contender_series`, `contender_stable`), `""` when none |
| `held` | bool | whether a suspect hold withholds the evidence this line reports: the `score` on `round`, `sync`, `reject`, `sides_corrected`, `pending_end` and `pending_cleared`, and the `read_score` on `score_held` |

While a suspect hold withholds a change, every event that reports the run's tracked score carries
`held: true`. Those are `round`, `sync`, `reject`, `sides_corrected`, `pending_end` and `pending_cleared`.
Publication still carries the corroborated score from before the hold, so the line's `score` is the
tracked score, not the published one. `score_held` is the one hold event that states the corroborated
score, and it puts the withheld read in `read_score`.

These optional keys appear only when the event carries them:

| Key | Type | Meaning |
| --- | --- | --- |
| `read_score` | object | the observed read keyed by team code, on `reject`, `score_held`, `score_corroborated` and `score_rollback` |
| `previous_score` | object | on `resync`: the score the new one replaced, keyed by team code |
| `exact` | bool | on a hidden `map_end`: Riot confirmed the team at match point as the winner, so the filled-in final round is exact. `false` with the score unchanged otherwise: an empty `winner` when a map closed with no known winner, a `next_map` guess of the team at match point before Riot completed the game, Riot's winner when it was not the team at match point, and the later `map_end` (`source: riot`) that fills in or corrects that map's winner |
| `pause` | string | the pause overlay's kind (`tech_pause`, ...) |
| `reason` | string | the overlay's reason suffix, when a template read one |
| `pause_since` | int | the second the pause's first overlay frame was seen, which can come before its confirmation |
| `riot_state` | string | on `map_start`, `map_join` and `sides_corrected`: Riot's state for the game when the event fired (`unstarted`, `inProgress`, `completed`), `""` when Riot lists none |
| `riot_blue` | string | on the same events: Riot's explicit blue team code for the game, `""` when Riot lists no explicit sides |
| `riot_red` | string | on the same events: Riot's explicit red team code for the game |
| `visual_left`, `visual_right` | string | audit: the team codes read from the left and right tag plates, `""` when unread. On `frame_observation`, and in a shadow run on every event of a known run |
| `left_score`, `right_score` | int or null | audit, on the same lines: the left and right plate scores, `null` when unreadable |
| `video_score` | object or null | audit, on the same lines: the plate scores keyed by the visual team codes, `null` unless both tags were read |
| `plate_ok`, `scene` | bool, string | on `frame_observation`: whether both tags were identified as different teams, and the frame's scene (`gameplay`, `replay`, `no_hud`) |
| `left_confidence`, `right_confidence` | float | on `frame_observation`: the better expected code's learned-crop similarity for each plate, `0` before a crop of either is learned; a plate the OCR named keeps this value |
| `scoreboard` | object | on `scoreboard_observation`: the buy-phase table read |

The set of kinds is fixed:

| Group | Kinds |
| --- | --- |
| Score and map | `map_start`, `map_join`, `round`, `sync`, `reject`, `resync`, `score_held`, `score_corroborated`, `score_rollback`, `map_end`, `sides_corrected`, `pending_end`, `pending_cleared` |
| Pause | `pause_start`, `pause_end` |
| Stream lifecycle | `stream_connect`, `stream_failed` |
| Audit | `frame_observation`, `video_accepted` (shadow only), `scoreboard_observation` (with `--scoreboard-context`) |

Stream lifecycle lines have no run or score: `map` and `match_id` are empty and `score` is `null`. They
carry one extra key, `stream`, the URL the runner connected to. This is where a shadow process records
that it started reading. A connection attempt that fails to open emits `stream_failed` before the runner
moves on, so every attempt has an outcome in the ledger, including the ones that bench their source.

`pause_end` carries a `source` only when a rule ends the pause (`overlay_gone` or `match_change`).

The audit kinds are described in [shadow-audit.md](shadow-audit.md) and
[scoreboard-native.md](scoreboard-native.md).

## Riot esports API

The tracker polls Riot's public esports API every 30 seconds in the background (`vbt/src/riot.rs`). It
uses it to know whether a match is live, which teams are playing, which side each team is on, each game's
state, and how many maps each team has won.

The base URL is `https://esports-api.service.valorantesports.com/persisted/val/<endpoint>?hl=en-US&sport=val`.
Requests send the `x-api-key` header with the public key that valorantesports.com sends from the browser
(`riot::API_KEY`).

| Endpoint | Fields used |
|---|---|
| `getLiveDetails` | `data.schedule.events[]`: events with `type == "match"` or a `match` object. Any such event means a match is live. The ID comes from `match.id`, else `id`. |
| `getEventDetails&id=<match>` | `data.event.match.strategy.count`: the series length (best of N). `data.event.match.teams[]`: `id`, `code`, `name`, `result.gameWins`. `data.event.match.games[]`: `number`, `state` (`unstarted`, `inProgress`, `completed`), and `teams[]` with `id` and `side` (`blue` or `red`). |

The stream is opened only while `getLiveDetails` lists a live match. That saves bandwidth and CPU between
matches, and it means a rerun on the channel is never read as live.

## Publishing to val-esports-backend

The tracker sends its score to val-esports-backend with one HTTP call. This section covers the transport,
the body and when the tracker writes.

### Transport and auth

The tracker sends `PUT /api/v1/video/score` over the backend's Unix socket, as HTTP/1.1 with
`Content-Type: application/json`. The timeout is 2 seconds, so a hung backend can't stall the frame loop.

At startup the tracker generates a random token (32 random bytes, URL-safe base64) and writes it to
`VIDEO_TOKEN_FILE` with mode `0600`. It forces the mode even if the file already exists. It sends the token
as `X-Video-Token`, and the backend compares it with the file's content.

Before writing the token, a live run takes an exclusive lock on `<VIDEO_TOKEN_FILE>.lock` and holds it for
the life of the process. A second live writer on the same token file exits at startup with `another live
writer holds <path>.lock`. Shadow runs and replays never take the lock or write the token.

`BACKEND_SOCKET` and `VIDEO_TOKEN_FILE` come from the environment or a `.env` file. If either is empty,
nothing is published.

### Body

```json
{
  "status": "ok",
  "observed_at": 1790501615,
  "map_number": 2,
  "teams": [
    {"code": "LOUD", "name": "LOUD", "score": 8, "side": "blue"},
    {"code": "EDG", "name": "EDward Gaming", "score": 5, "side": "red"}
  ],
  "pause": {"kind": "tech_pause", "reason": "GEAR", "since": 35}
}
```

`teams` holds both teams' Riot codes and names with their rounds, and the score-bar side each is shown on:
`blue` is the left plate and `red` the right. The tracker reports the sides it is publishing with. A side
correction changes these sides, so the corrected score is published as a new state even when the score
is tied.

`pause` is present only while the broadcast shows a pause overlay. It holds the `kind` (`tech_pause`,
`timeout` or `halftime`), the `reason` suffix when a template read one, and `since`, the
tracker-relative second the pause began. That is seconds since the tracker started, not a Unix timestamp,
and it tells two pauses apart. `pause` is absent while play is running.

`observed_at` is the Unix time in seconds when the write was made.

The backend responds `204` when it stored the score, `401` for a bad or missing token, and `422` for an
invalid body.

### When to write

The tracker writes only while Riot lists a live match and a map is active. It writes at once whenever
`(map_number, score, status, side)` changes, or when a pause starts or ends. Otherwise it writes every 30
seconds as a heartbeat. A failed write is logged and retried with the next heartbeat, never on every
frame.

`status` is `ok` while the map hasn't ended, no suspect score is held, and a read matched the tracked score
within the last 90 seconds (`UNREAD_SECONDS`). It is also `ok` while a pause is on, because the overlay
says play has stopped, not that the tracker lost the broadcast.

`status` is `error` in these cases:

- the map has ended;
- a suspect hold is withholding a change;
- a side reversal is pending;
- a contender series has lasted 20 seconds with 3 reads (`CONTENDER_DOUBT_SECONDS`, `CONTENDER_DOUBT_READS`), so the run's own score may be wrong;
- 90 seconds have passed without a matching read, for example during a camera shot the HUD doesn't cover.

The payload still carries a score during `error`. Normally it is the last tracked score. During a hold it
is the score from before the hold, and while a reversal is pending it is the last publication's score and
sides.

## Failure behaviour

| Situation | Behaviour |
|---|---|
| Stream ends or stalls | A stall is 30 s without a frame. A stall, or an end while Riot still lists the match live, counts as a failure of that source. Kill both processes, wait 10 s, reconnect. Three failures in a row bench the source for 120 s and the next source is tried. The active run is kept across reconnects. |
| No match live on Riot | Don't open the stream; check again every 30 s. |
| Riot API fails | Keep the last known details; log and retry on the next poll. |
| Backend unreachable or rejects | Log; retry with the next heartbeat. |
| Score unreadable for 90 s | Publish `error`. |
| No map name from veto or strip | No run is started for the map. |
| Riot reverses sides | `error` for up to 120 s, then the corrected score is published. |
| Tracker starts mid-map | Joins after 60 s of corroborated play: Riot's in-progress game with explicit sides, this match's tags in Riot's order, a steady non-zero score, and a strip that names the game and agrees with the veto. A pause overlay resets the series, and a strip that names nothing does not join. Until then nothing is published. |

## Performance

The score path uses no OCR model. It runs scene classification, the template readers, and one 10-class
logistic-regression forward pass per score glyph. With the default `scoreboard` feature, the strip name is
read by a small CTC model (5.7 ms p50, at most twice per 5 s).

Replaying the 38-minute Abyss clip at 1 fps on a 16-core x86_64 machine, the equivalent classifier stack
measured 47 s wall time and 468 s CPU time at 203 MB peak RSS, including the ffmpeg decode.

In steady state a frame costs about 2 ms CPU. Scene detection and template reads take about 2 ms, and a
forced classifier check of both plates takes about 2.5 ms. The learned tag crops add about 0.3 ms with
23 codes' crops learned at the frame height, since every learned code is a rival for the margin; the
cost grows with the store. A
two-tag OCR read took 13.4 ms on the raw crops, 19.4 ms with the 1.3 stretch and 26.7 ms with the 1.8
stretch on the deploy host; it runs only on the frames listed in "Reading the team tags". In a build without the `scoreboard` feature,
the template map-strip read takes about 13 ms, and it runs only on the scheduled 5-second reads that are
starting or joining a map.

The optional in-process scoreboard adds the embedded ONNX model, with a session and activations of about
20 MB. It is documented separately in [scoreboard-native.md](scoreboard-native.md).

## Implementation exactness notes

Both recognition tasks read one line of known text from a fixed crop, so only recognition is needed, not
text detection. But small numerical differences move confidences near the bars, so these details must
match exactly:

- Normalization: glyph and template vectors are zero-mean unit vectors (the mean-subtracted vector divided
  by its own L2 norm). The 0.65 confidence bar assumes unit vectors.
- Autocontrast: map the histogram's lowest and highest used levels to 0 and 255 with a lookup table, not a
  percentile stretch.
- Otsu: scan levels in order and keep the first maximum. Ink is strictly above the threshold. Other Otsu
  variants give a different mask on about 1 % of plate crops.
- Lanczos: kernel implementations differ slightly between libraries, so check with a replay rather than
  assume they are equal.
- Template variants: shifts wrap around, and scalings use clamped nearest-neighbor indices, not
  interpolation.

## How the assets are made

The embedded assets are built offline from broadcast frames. The scripts that made them are not in this
repository.

- `assets/glyphs/<digit>/` holds 24x32 grayscale crops of real score-bar digits.
- `assets/recognizer/digits_lr.bin` is the digit classifier, trained on harvested, labelled glyphs by
  `vision/tools/train_digits.py`. Its layout is in [scoreboard-native.md](scoreboard-native.md).
- `strip_templates.json` holds grayscale map-name templates, each the per-pixel median of a fixed window
  over confirmed 0-0 frames. `pause_templates.json` holds binarized phrases, each from one real frame.
  Their sources are listed in `assets/recognizer/README.md`.
- The scoreboard model is built by `scoreboard/tools/prune_head.py`.

## Testing

The test images are not in git. `scripts/fetch-testdata.sh` unpacks them from the asset host.

`testdata/frames/` holds real frames with known scores, plus `strip-unknown-splint.png`, a synthetic
`SPLINT` strip the map reader must refuse. The reader tests track `summit-12-4.jpg` from 11-4 to 12-4: the
templates read 12-4 confidently and the classifier confirms the change. They change nothing on
`lotus-half-time.jpg`. `summit-5-12.png` and the `falsegame-0840xx.png` frames replay the VIT vs LOUD map 2
cut. The HUD frame corroborates gameplay, while the break frames read no pair and must not clear or defer
a pending end. `testdata/teamplates/` holds the team-tag fixtures: a TL vs G2 Abyss map start the tests learn
crops from, a later round they must read, and a replay they must refuse.

The tracker tests in `vbt/tests/` replay captured frames through the state machine. They assert the ledger
events, the published payloads, the pause paths and the shadow audit.

`scoreboard/testdata/` holds two broadcast tables with expected canonical outputs. The reader reproduces
them at 1080p and 720p.

To compare against a real map, record a full map, ideally one whose final score is cut away, and replay
it with `--file`. Then inspect the logged map starts, rounds (second and score), syncs and hidden map
ends. `--events-file` writes the same format live and on replay, so the two can be diffed line by line.

Before switching a change live, run the tracker with `--shadow` next to the publishing tracker during a
live match and compare their logs.
