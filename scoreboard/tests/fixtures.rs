//! §5 verification on the shared fixtures: the three inputs, the null frame, the padding
//! check, and the `--serve` stdio protocol. The strip name and team tag readers'
//! contracts run on the shared frames too.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use image::imageops::FilterType;
use vbt_scoreboard::reader::{BatchMode, Reader};
use vbt_scoreboard::schema::{Context, Scoreboard};
use vbt_scoreboard::strip::{self, MAP_MARGIN};
use vbt_scoreboard::tags::{self, TAG_MARGIN};
use vbt_vision::frame::Frame;
use vbt_vision::recognizer::{ANCHOR_THRESHOLD, Recognizer};
use vbt_vision::regions::MAP_STRIP;

include!("../../testdata/fixture.rs");

const TL_CONTEXT: &[u8] = include_bytes!("../testdata/tl-g2-context.json");
const TL_EXPECTED: &[u8] = include_bytes!("../testdata/tl-g2-r6.expected.json");
const XLG_CONTEXT: &[u8] = include_bytes!("../testdata/xlg-kc-context.json");
const XLG_EXPECTED: &[u8] = include_bytes!("../testdata/xlg-kc-r19.expected.json");

fn image_bytes(name: &str) -> Vec<u8> {
    std::fs::read(fixture(&format!("scoreboard/testdata/{name}"))).unwrap()
}

fn frame_from(bytes: &[u8]) -> Frame {
    let image = image::load_from_memory(bytes).expect("fixture decodes");
    Frame::from_rgb_image(&image.to_rgb8())
}

fn context(bytes: &[u8]) -> Context {
    serde_json::from_slice(bytes).expect("context parses")
}

fn assert_matches(output: &Option<Scoreboard>, expected: &[u8]) {
    let expected: serde_json::Value = serde_json::from_slice(expected).unwrap();
    let value = serde_json::to_value(output).unwrap();
    assert_eq!(value, expected, "canonical JSON differs");
    let Some(scoreboard) = output else {
        panic!("expected a scoreboard, got null");
    };
    let expected_players: BTreeMap<(String, String), Vec<i64>> = expected["players"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            (
                (
                    p["team"].as_str().unwrap().to_string(),
                    p["name"].as_str().unwrap().to_string(),
                ),
                p["kda"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap())
                    .collect(),
            )
        })
        .collect();
    let players: BTreeMap<(String, String), Vec<i64>> = scoreboard
        .players
        .iter()
        .map(|p| ((p.team.clone(), p.name.clone()), p.kda.to_vec()))
        .collect();
    assert_eq!(players, expected_players, "player sets differ");
    let teams: BTreeMap<String, Option<i64>> = scoreboard
        .teams
        .iter()
        .map(|t| (t.name.clone(), t.score))
        .collect();
    let expected_teams: BTreeMap<String, Option<i64>> = expected["teams"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| (t["name"].as_str().unwrap().to_string(), t["score"].as_i64()))
        .collect();
    assert_eq!(teams, expected_teams, "team sets differ");
}

fn analyze(bytes: &[u8], context_bytes: &[u8]) -> Option<Scoreboard> {
    let mut reader = Reader::new(2).unwrap();
    reader
        .analyze(&frame_from(bytes), &context(context_bytes))
        .unwrap()
}

#[test]
fn tl_1080p_and_xlg_720p_match_expected() {
    assert_matches(
        &analyze(&image_bytes("tl-g2-r6.png"), TL_CONTEXT),
        TL_EXPECTED,
    );
    assert_matches(
        &analyze(&image_bytes("xlg-kc-r19.jpg"), XLG_CONTEXT),
        XLG_EXPECTED,
    );
}

#[test]
fn tl_downscaled_to_720p_matches_expected() {
    let full = image::load_from_memory(&image_bytes("tl-g2-r6.png"))
        .unwrap()
        .to_rgb8();
    for filter in [FilterType::Triangle, FilterType::CatmullRom] {
        let small = image::imageops::resize(&full, 1280, 720, filter);
        let frame = Frame::from_rgb_image(&small);
        let mut reader = Reader::new(2).unwrap();
        let scoreboard = reader.analyze(&frame, &context(TL_CONTEXT)).unwrap();
        assert_matches(&scoreboard, TL_EXPECTED);
    }
}

#[test]
fn non_table_frame_is_null() {
    let frame = frame_from(&image_bytes("abyss-start.png"));
    let mut reader = Reader::new(2).unwrap();
    let scoreboard = reader
        .analyze(&frame, &context(TL_CONTEXT))
        .expect("analyze succeeds");
    assert!(scoreboard.is_none(), "start frame must not report a table");
}

/// Padding must not change which candidate wins each field (§3.1). Absolute CTC
/// scores rise with padding (extra timesteps admit more paths), so a score-normalised
/// threshold can flip between runs; on the TL fixture the 1080p/720p left tag only clears
/// `score/len >= -3.0` in the batched run, which is the run §3.1 prescribes and the one
/// whose output matches the expected JSON. The winner itself is invariant everywhere.
#[test]
fn padding_does_not_change_winning_candidates() {
    let full = image::load_from_memory(&image_bytes("tl-g2-r6.png"))
        .unwrap()
        .to_rgb8();
    let tl_small = Frame::from_rgb_image(&image::imageops::resize(
        &full,
        1280,
        720,
        FilterType::Triangle,
    ));
    let cases: [(&Frame, &[u8]); 3] = [
        (&frame_from(&image_bytes("tl-g2-r6.png")), TL_CONTEXT),
        (&tl_small, TL_CONTEXT),
        (&frame_from(&image_bytes("xlg-kc-r19.jpg")), XLG_CONTEXT),
    ];
    for (frame, context_bytes) in cases {
        let ctx = context(context_bytes);
        let mut auto = Reader::with_mode(2, BatchMode::Auto).unwrap();
        auto.analyze(frame, &ctx).unwrap();
        let batched: Vec<_> = auto
            .decisions()
            .iter()
            .map(|d| (d.field.clone(), d.value.clone(), d.accepted))
            .collect();
        let mut per_crop = Reader::with_mode(2, BatchMode::PerCrop).unwrap();
        per_crop.analyze(frame, &ctx).unwrap();
        let single: Vec<_> = per_crop
            .decisions()
            .iter()
            .map(|d| (d.field.clone(), d.value.clone(), d.accepted))
            .collect();
        assert_eq!(batched.len(), single.len());
        for (a, b) in batched.iter().zip(&single) {
            assert_eq!(
                (a.0.clone(), a.1.clone()),
                (b.0.clone(), b.1.clone()),
                "winning candidate differs between batch modes"
            );
        }
    }
}

#[test]
fn serve_speaks_the_stdio_protocol() {
    let context_path =
        std::env::temp_dir().join(format!("vbt-scoreboard-{}.json", std::process::id()));
    std::fs::write(&context_path, TL_CONTEXT).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_vbt-scoreboard"))
        .arg("--context")
        .arg(&context_path)
        .arg("--serve")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let mut roundtrip = |payload: &[u8]| -> Vec<u8> {
        stdin
            .write_all(&(payload.len() as u32).to_be_bytes())
            .unwrap();
        stdin.write_all(payload).unwrap();
        stdin.flush().unwrap();
        let mut header = [0u8; 4];
        stdout.read_exact(&mut header).unwrap();
        let size = u32::from_be_bytes(header) as usize;
        let mut body = vec![0u8; size];
        stdout.read_exact(&mut body).unwrap();
        body
    };

    let body = roundtrip(&image_bytes("tl-g2-r6.png"));
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let expected: serde_json::Value = serde_json::from_slice(TL_EXPECTED).unwrap();
    assert_eq!(value, expected, "daemon response differs");

    let body = roundtrip(b"this is not an image");
    assert_eq!(body, b"null", "malformed payload must answer null");

    drop(stdin);
    assert!(child.wait().unwrap().success(), "daemon exits 0 on EOF");
    let _ = std::fs::remove_file(&context_path);
}

/// The repository's `testdata/frames/`, shared with the vision tests.
fn frames_dir() -> PathBuf {
    fixture("testdata/frames")
}

/// `testdata/frames/<name>` cropped to the strip the tracker reads.
fn strip_of(name: &str) -> Frame {
    let path = frames_dir().join(name);
    let frame = frame_from(&std::fs::read(&path).unwrap_or_else(|e| panic!("read {name}: {e}")));
    frame.crop_raw(MAP_STRIP)
}

/// The strip name reader names every map from the thirteen-name lexicon on four
/// broadcasts, including the dimmer 720p renders the template reader refuses. The
/// Summit/Sunset pair, whose template margin is as low as 0.022 on the fixtures below,
/// separates by tens of nats.
#[test]
fn strip_names_the_map_without_templates() {
    let vision = Recognizer::new().expect("recognizer");
    let reader = strip::Reader::new(2).expect("strip reader");
    let cases = [
        // The Abyss template scores 0.719 on Haven's strip, as high as on a true Abyss
        // strip.
        ("haven-0-0.jpg", "Haven"),
        // Summit renders whose template gap to Sunset is 0.022/0.024, inside the margin.
        ("summit-5-12.png", "Summit"),
        ("summit-13-5.jpg", "Summit"),
        // A second broadcast (NRG vs LOUD, map 3) with a SUNSET panel beside the name.
        ("summit-nrg-loud.jpg", "Summit"),
        ("sunset-0-0.jpg", "Sunset"),
        ("sunset-nrg-loud.jpg", "Sunset"),
        ("split-0-0.jpg", "Split"),
        ("breeze-0-0.jpg", "Breeze"),
        ("ge-vit-abyss-4322.jpg", "Abyss"),
        ("prx-g2-lotus-ticker.jpg", "Lotus"),
        ("lotus-occlusion-15978.jpg", "Lotus"),
    ];
    for (name, want) in cases {
        assert_eq!(reader.map(&vision, &strip_of(name)), want, "{name}");
    }
    // The confusable pair's margin, in nats, on both sides of the collision.
    for (name, want) in [
        ("summit-5-12.png", "Summit"),
        ("summit-nrg-loud.jpg", "Summit"),
        ("sunset-0-0.jpg", "Sunset"),
        ("sunset-nrg-loud.jpg", "Sunset"),
    ] {
        let read = reader.read(&vision, &strip_of(name)).expect("anchor found");
        assert_eq!(read.name, want);
        assert!(
            read.accepted && read.margin >= 20.0,
            "{name} margin = {:.2}, want a decisive one",
            read.margin
        );
    }
}

/// A `CURRENT:` anchor can be a false positive on a blurry strip; the name must then stay
/// empty. `strip-blur.jpg` correlates 0.761 with `CURRENT:` (above the template floor) but
/// carries no text, and the model's best candidate wins a 4-nat margin with -4.3 nats per
/// character, under the floor on both counts. `strip-unknown-splint.png` carries text the
/// lexicon does not know: `SPLINT` scores Split at 44.6 nats with -1.3 nats per character,
/// clearing the margin and the per-character floor, but the window's free decode is not
/// Split, so the read is refused instead of being named after its closest match.
#[test]
fn strip_stays_silent_without_a_name() {
    let vision = Recognizer::new().expect("recognizer");
    let reader = strip::Reader::new(2).expect("strip reader");
    for name in [
        "strip-blur.jpg",
        "lotus-half-time.jpg",
        "stage-3712.jpg",
        "countdown-card-5835.jpg",
        "strip-unknown-splint.png",
    ] {
        assert_eq!(reader.map(&vision, &strip_of(name)), "", "{name}");
    }
    let read = reader
        .read(&vision, &strip_of("strip-blur.jpg"))
        .expect("anchor is locatable");
    assert!(
        !read.accepted && read.margin < MAP_MARGIN && read.per_char < -3.0,
        "blur read = {read:?}, want it refused on the margin and the per-character floor"
    );
    let read = reader
        .read(&vision, &strip_of("strip-unknown-splint.png"))
        .expect("anchor is locatable");
    assert_eq!(read.name, "Split", "the lexicon's closest name");
    assert_eq!(read.free, "SPLINT", "the window's free decode");
    assert!(
        !read.known && !read.accepted,
        "unknown text read = {read:?}, want it refused on the free decode"
    );
    assert!(
        read.margin >= MAP_MARGIN,
        "the lexicon margin alone clears the floor; the free decode is the only refusal"
    );
}

/// The model is weakest on the oldest renderings: this Summit frame scores 5.9 nats
/// over Sunset, under the floor, so the read is refused and the templates name it instead
/// (Summit 0.99 / Sunset 0.76). The fallback keeps the strip readable where the model's
/// likelihood drops, and the tracker's strip agreement paths are unchanged.
#[test]
fn strip_names_the_weak_render_from_the_templates() {
    let vision = Recognizer::new().expect("recognizer");
    let reader = strip::Reader::new(2).expect("strip reader");
    let strip = strip_of("summit-12-4.jpg");
    let read = reader.read(&vision, &strip).expect("anchor found");
    assert_eq!(read.name, "Summit");
    assert!(
        !read.accepted && read.margin < MAP_MARGIN,
        "read = {read:?}, want it under the floor"
    );
    assert_eq!(reader.map(&vision, &strip), "Summit", "template fallback");
}

/// `CURRENT:` under the template reader's floor is still usable when the name separates:
/// this Summit render correlates 0.700, and the template read is empty, but the name
/// scores 20+ nats over Sunset.
#[test]
fn strip_names_a_map_under_the_template_anchor_floor() {
    let vision = Recognizer::new().expect("recognizer");
    let reader = strip::Reader::new(2).expect("strip reader");
    let strip = strip_of("summit-low-anchor.jpg");
    let anchor = vision.anchor(&strip);
    assert!(
        anchor.score < ANCHOR_THRESHOLD,
        "anchor = {anchor:?}, want it under the template floor"
    );
    assert_eq!(vision.map(&strip), "", "template read");
    let read = reader.read(&vision, &strip).expect("anchor found");
    assert_eq!(read.name, "Summit");
    assert!(read.accepted, "read = {read:?}");
    assert_eq!(reader.map(&vision, &strip), "Summit");
}

/// The tag reader needs no per-team asset: NS and VIT never had a shipped crop, and both
/// read on the production 720p decode.
#[test]
fn tags_read_teams_without_any_asset() {
    let reader = tags::Reader::new(2).expect("tag reader");
    let frame = frame_from(&std::fs::read(frames_dir().join("split-0-0.jpg")).unwrap());
    let [left, right] = reader.read(&frame, ["VIT", "NS"]);
    assert_eq!((left.code.as_str(), right.code.as_str()), ("NS", "VIT"));
}

/// T1 out-scores G2 on TL's plate by 14.6 nats, so the lexicon alone would name it; the
/// free decode spells TL, so the side is refused rather than read as the closest code
/// (no tolerance and no L-for-1 fold). The other plate still reads on its own.
#[test]
fn tags_refuse_a_code_the_plate_does_not_spell() {
    let reader = tags::Reader::new(2).expect("tag reader");
    let frame =
        frame_from(&std::fs::read(fixture("testdata/teamplates/abyss-round5.png")).unwrap());
    let [left, right] = reader.read(&frame, ["T1", "G2"]);
    assert_eq!((left.best.as_str(), left.free.as_str()), ("T1", "TL"));
    assert!(
        left.code.is_empty() && left.margin >= TAG_MARGIN,
        "{left:?}"
    );
    assert_eq!(right.code, "G2");
}
