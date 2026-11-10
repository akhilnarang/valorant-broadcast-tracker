//! Shared fakes and frame helpers for the tracker tests.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use vbt::tracker::{
    Event, MapOrder, MapRun, Pause, Published, Publisher, PushError, Riot, Score, ScoreMap, Sides,
    Status, Team, Tracker, VisionRecognizer,
};
use vbt_vision::frame::{Frame, rgba_luma};
use vbt_vision::learned::Crops;
use vbt_vision::recognizer::PauseRead;
use vbt_vision::regions::{SCORE_LEFT, SCORE_RIGHT, TEAM_LEFT, TEAM_RIGHT};
use vbt_vision::templates::Templates;

include!("../../../testdata/fixture.rs");

pub fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(relative)
}

pub fn frames_dir() -> PathBuf {
    fixture("testdata/frames")
}

pub fn glyphs_dir() -> PathBuf {
    repo_path("assets/glyphs")
}

/// Decodes an image file into an RGB frame.
pub fn load_frame_path(path: &Path) -> Frame {
    let image = image::open(path).unwrap_or_else(|e| panic!("decode {}: {e}", path.display()));
    Frame::from_rgb_image_owned(image.to_rgb8())
}

/// Decodes `testdata/frames/<name>` into an RGB frame.
pub fn load_frame(name: &str) -> Frame {
    load_frame_path(&frames_dir().join(name))
}

/// Decodes a 1280-wide score-bar band and pastes it at a 1280x720 frame's top-left.
pub fn load_band_frame(name: &str) -> Frame {
    let band = image::open(frames_dir().join(name)).expect("decode band");
    let band = band.to_rgb8();
    let mut frame = Frame::from_rgb(1280, 720, vec![0; 1280 * 720 * 3]).unwrap();
    for y in 0..band.height() as usize {
        for x in 0..band.width() as usize {
            let offset = (y * 1280 + x) * 3;
            let pixel = band.get_pixel(x as u32, y as u32);
            frame.pix[offset] = pixel[0];
            frame.pix[offset + 1] = pixel[1];
            frame.pix[offset + 2] = pixel[2];
        }
    }
    frame
}

/// Pastes a 1920-wide team-plate fixture into a 1920x1080 frame's top-left.
pub fn load_teamplate_frame(name: &str) -> Frame {
    let image = image::open(fixture("testdata/teamplates").join(name))
        .unwrap_or_else(|e| panic!("decode {name}: {e}"));
    let image = image.to_rgb8();
    let mut frame = Frame::from_rgb(1920, 1080, vec![0; 1920 * 1080 * 3]).unwrap();
    for y in 0..image.height() as usize {
        for x in 0..image.width() as usize {
            let offset = (y * 1920 + x) * 3;
            let pixel = image.get_pixel(x as u32, y as u32);
            frame.pix[offset] = pixel[0];
            frame.pix[offset + 1] = pixel[1];
            frame.pix[offset + 2] = pixel[2];
        }
    }
    frame
}

/// A crop store that learned TL (left) and G2 (right) from the Abyss map start, as the
/// tracker would from an OCR read of both tags.
pub fn learned_plates() -> Crops {
    let mut crops = Crops::open(None).expect("in-memory crops");
    crops
        .learn(&load_teamplate_frame("abyss-start.png"), ["TL", "G2"])
        .expect("learn TL and G2");
    crops
}

/// A 1920-wide `frame` with both tag plate boxes blacked out and `left` and `right`
/// centred in them; `None` leaves a box blank.
pub fn paste_plates(frame: &Frame, left: Option<&Frame>, right: Option<&Frame>) -> Frame {
    let mut out = frame.clone();
    for (source, dest) in [(left, TEAM_LEFT), (right, TEAM_RIGHT)] {
        let w = dest.right - dest.left;
        for y in dest.top..dest.bottom {
            for x in 0..w {
                let i = ((y * out.w as i64 + dest.left + x) * 3) as usize;
                let pixel = source
                    .and_then(|source| {
                        let sx = x - (w - source.w as i64) / 2;
                        let sy = (y - dest.top) as usize;
                        (sx >= 0 && sx < source.w as i64 && sy < source.h).then(|| {
                            let si = (sy * source.w + sx as usize) * 3;
                            [source.pix[si], source.pix[si + 1], source.pix[si + 2]]
                        })
                    })
                    .unwrap_or([0; 3]);
                out.pix[i..i + 3].copy_from_slice(&pixel);
            }
        }
    }
    out
}

/// A gameplay frame whose score plates are visible but unreadable, as a partial
/// occlusion is: bright blocks sit where the digits should be.
pub fn occluded_plate_frame() -> Frame {
    let mut frame = Frame::from_rgb(1280, 720, vec![0; 1280 * 720 * 3]).unwrap();
    let scale = 1280.0 / 1920.0;
    for b in [SCORE_LEFT, SCORE_RIGHT] {
        let x0 = (b.left as f64 * scale) as i64;
        let x1 = (b.right as f64 * scale) as i64;
        let y0 = (b.top as f64 * scale) as i64;
        let y1 = (b.bottom as f64 * scale) as i64;
        for y in y0..y1 {
            for x in x0..x1 {
                let offset = ((y * 1280 + x) * 3) as usize;
                frame.pix[offset] = 255;
                frame.pix[offset + 1] = 255;
                frame.pix[offset + 2] = 255;
            }
        }
    }
    frame
}

/// A 720p gameplay frame whose score plates show 0-0, drawn from a zero glyph
/// template, plus a bright timer strip so the scene reads as gameplay.
pub fn zero_zero_frame() -> Frame {
    let (_, _, data) = vbt_vision::GLYPHS
        .iter()
        .find(|(_, digit, _)| *digit == 0)
        .expect("a zero glyph");
    let glyph = image::load_from_memory_with_format(data, image::ImageFormat::Png)
        .expect("decode zero glyph")
        .to_rgba8();
    let mut frame = Frame::from_rgb(1280, 720, vec![0; 1280 * 720 * 3]).unwrap();
    let draw = |frame: &mut Frame, dx: i64, dy: i64| {
        for y in 0..glyph.height() as i64 {
            for x in 0..glyph.width() as i64 {
                let pixel = glyph.get_pixel(x as u32, y as u32);
                let luma = rgba_luma(pixel[0], pixel[1], pixel[2], pixel[3]);
                let offset = (((dy + y) * 1280 + dx + x) * 3) as usize;
                frame.pix[offset] = luma;
                frame.pix[offset + 1] = luma;
                frame.pix[offset + 2] = luma;
            }
        }
    };
    draw(&mut frame, 535, 4); // ScoreLeft's plate
    draw(&mut frame, 708, 3); // ScoreRight's plate
    for y in 20..32 {
        for x in 600..620 {
            let offset = (y * 1280 + x) * 3;
            frame.pix[offset] = 255;
            frame.pix[offset + 1] = 255;
            frame.pix[offset + 2] = 255;
        }
    }
    frame
}

/// The scriptable Riot state the fake holds.
#[derive(Debug, Clone, Default)]
pub struct FakeRiotState {
    pub live: bool,
    pub event: bool,
    pub match_id: String,
    pub teams: Vec<Team>,
    pub sides: BTreeMap<i64, Sides>,
    pub order: Sides,
    pub states: BTreeMap<i64, String>,
    pub wins: ScoreMap,
    pub best_of: i64,
    pub game: i64,
}

/// A scripted Riot source.
#[derive(Clone, Default)]
pub struct FakeRiot {
    pub state: Arc<Mutex<FakeRiotState>>,
}

impl FakeRiot {
    pub fn new() -> FakeRiot {
        FakeRiot::default()
    }

    pub fn get(&self) -> MutexGuard<'_, FakeRiotState> {
        self.state.lock().unwrap()
    }
}

impl Riot for FakeRiot {
    fn live_now(&self) -> bool {
        self.get().live
    }

    fn has_event(&self) -> bool {
        self.get().event
    }

    fn match_id(&self) -> String {
        self.get().match_id.clone()
    }

    fn teams(&self) -> Vec<Team> {
        self.get().teams.clone()
    }

    fn sides(&self, _match_id: &str, map_number: i64) -> Option<Sides> {
        self.get().sides.get(&map_number).cloned()
    }

    fn listed_order(&self, _match_id: &str) -> Sides {
        self.get().order.clone()
    }

    fn map_number(&self, played: i64) -> i64 {
        let state = self.get();
        if state.game > 0 {
            state.game
        } else {
            played + 1
        }
    }

    fn game_state(&self, map_number: i64) -> Option<String> {
        self.get().states.get(&map_number).cloned()
    }

    fn game_wins(&self) -> ScoreMap {
        self.get().wins.clone()
    }

    fn best_of(&self) -> i64 {
        self.get().best_of
    }
}

/// Builds the tracker most tests start from.
pub fn test_tracker() -> (Tracker, FakeRiot) {
    let templates = Templates::load_dir(&glyphs_dir()).expect("load templates");
    let riot = FakeRiot::new();
    {
        let mut state = riot.get();
        state.match_id = "match".to_string();
        state.teams = vec![
            Team {
                id: "1".into(),
                code: "LOUD".into(),
                name: "LOUD".into(),
            },
            Team {
                id: "2".into(),
                code: "EDG".into(),
                name: "EDward Gaming".into(),
            },
        ];
        state.sides.insert(
            2,
            Sides {
                blue: "LOUD".into(),
                red: "EDG".into(),
            },
        );
        state.order = Sides {
            blue: "LOUD".into(),
            red: "EDG".into(),
        };
    }
    let tracker = Tracker::new(templates, None, Box::new(riot.clone()), None, None);
    (tracker, riot)
}

/// Installs a run as the active one and returns its index, the Rust form of
/// assigning `tracker.Active`.
pub fn set_active(tracker: &mut Tracker, run: MapRun) -> usize {
    tracker.maps.push(run);
    let index = tracker.maps.len() - 1;
    tracker.active = Some(index);
    index
}

pub fn active(tracker: &Tracker) -> &MapRun {
    tracker.active_run().expect("an active run")
}

pub fn update_active(tracker: &mut Tracker, update: impl FnOnce(&mut MapRun)) {
    let index = tracker.active.expect("an active run");
    update(&mut tracker.maps[index]);
}

/// Captures every emitted event into a shared vector.
pub fn capture(tracker: &mut Tracker) -> Arc<Mutex<Vec<Event>>> {
    let events = Arc::new(Mutex::new(Vec::new()));
    let sink = events.clone();
    tracker.on_event = Some(Box::new(move |event| {
        sink.lock().unwrap().push(event.clone());
    }));
    events
}

/// The scriptable publisher state.
#[derive(Debug, Clone, Default)]
pub struct PublisherState {
    pub enabled: bool,
    pub scores: Vec<Published>,
    pub blues: Vec<String>,
    pub pauses: Vec<Option<Pause>>,
    pub seconds: Vec<i64>,
    pub current_second: i64,
}

/// Records every publish.
#[derive(Clone, Default)]
pub struct FakePublisher {
    pub state: Arc<Mutex<PublisherState>>,
}

impl FakePublisher {
    pub fn new(enabled: bool) -> FakePublisher {
        let publisher = FakePublisher::default();
        publisher.get().enabled = enabled;
        publisher
    }

    pub fn get(&self) -> MutexGuard<'_, PublisherState> {
        self.state.lock().unwrap()
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.get().enabled = enabled;
    }
}

impl Publisher for FakePublisher {
    fn enabled(&self) -> bool {
        self.get().enabled
    }

    fn publish(
        &self,
        map_number: i64,
        score: &ScoreMap,
        _teams: &[Team],
        blue: &str,
        status: Status,
        pause: Option<&Pause>,
    ) -> Result<(), PushError> {
        let mut published = Published {
            map_number,
            score: score.clone(),
            status,
            blue: blue.to_string(),
            ..Default::default()
        };
        if let Some(pause) = pause {
            published.pause = pause.kind.clone();
            published.pause_reason = pause.reason.clone();
        }
        let mut state = self.get();
        let second = state.current_second;
        state.scores.push(published);
        state.blues.push(blue.to_string());
        state.pauses.push(pause.cloned());
        state.seconds.push(second);
        Ok(())
    }
}

/// Scripts the map-strip reads.
#[derive(Clone, Default)]
pub struct StubRecognizer {
    pub name: Arc<Mutex<String>>,
}

impl StubRecognizer {
    pub fn new(name: &str) -> StubRecognizer {
        let recognizer = StubRecognizer::default();
        *recognizer.name.lock().unwrap() = name.to_string();
        recognizer
    }

    pub fn set_name(&self, name: &str) {
        *self.name.lock().unwrap() = name.to_string();
    }
}

impl vbt::tracker::Recognizer for StubRecognizer {
    fn score(&self, _crop: &Frame) -> Option<i32> {
        Some(0)
    }

    fn map(&self, _strip: &Frame) -> String {
        self.name.lock().unwrap().clone()
    }

    fn pause(&self, _header: &Frame, _card: &Frame) -> (PauseRead, bool) {
        (PauseRead::default(), false)
    }
}

/// Answers both plates with the scripted pair.
#[derive(Debug, Clone, Default)]
pub struct FixedPairState {
    pub pair: Score,
    pub call: usize,
    pub pause: PauseRead,
    pub on: bool,
    /// The scripted map-strip read.
    pub map: String,
}

#[derive(Clone, Default)]
pub struct FixedPairRecognizer {
    pub state: Arc<Mutex<FixedPairState>>,
}

impl FixedPairRecognizer {
    pub fn new(pair: Score) -> FixedPairRecognizer {
        let recognizer = FixedPairRecognizer::default();
        recognizer.get().pair = pair;
        recognizer
    }

    pub fn get(&self) -> MutexGuard<'_, FixedPairState> {
        self.state.lock().unwrap()
    }
}

impl vbt::tracker::Recognizer for FixedPairRecognizer {
    fn score(&self, _crop: &Frame) -> Option<i32> {
        let mut state = self.get();
        let call = state.call;
        state.call += 1;
        Some(if call.is_multiple_of(2) {
            state.pair.left
        } else {
            state.pair.right
        })
    }

    fn map(&self, _strip: &Frame) -> String {
        self.get().map.clone()
    }

    fn pause(&self, _header: &Frame, _card: &Frame) -> (PauseRead, bool) {
        let state = self.get();
        if !state.on {
            return (PauseRead::default(), false);
        }
        (state.pause.clone(), true)
    }
}

/// Reads nothing for two calls, then 13/11 alternating.
#[derive(Clone, Default)]
pub struct TerminalScoreRecognizer {
    pub calls: Arc<Mutex<usize>>,
}

impl vbt::tracker::Recognizer for TerminalScoreRecognizer {
    fn score(&self, _crop: &Frame) -> Option<i32> {
        let mut calls = self.calls.lock().unwrap();
        let call = *calls;
        *calls += 1;
        if call < 2 {
            return None;
        }
        Some(if (call - 2).is_multiple_of(2) { 13 } else { 11 })
    }

    fn map(&self, _strip: &Frame) -> String {
        String::new()
    }

    fn pause(&self, _header: &Frame, _card: &Frame) -> (PauseRead, bool) {
        (PauseRead::default(), false)
    }
}

/// A fixed map order by match id.
#[derive(Clone, Default)]
pub struct FakeMapOrder {
    pub names: BTreeMap<String, Vec<String>>,
}

impl FakeMapOrder {
    pub fn new(entries: &[(&str, &[&str])]) -> FakeMapOrder {
        FakeMapOrder {
            names: entries
                .iter()
                .map(|(match_id, names)| {
                    (
                        (*match_id).to_string(),
                        names.iter().map(|name| (*name).to_string()).collect(),
                    )
                })
                .collect(),
        }
    }
}

impl MapOrder for FakeMapOrder {
    fn map_name(&self, match_id: &str, map_number: i64) -> String {
        let Some(order) = self.names.get(match_id) else {
            return String::new();
        };
        if map_number < 1 || map_number as usize > order.len() {
            return String::new();
        }
        order[map_number as usize - 1].clone()
    }
}

/// The real recognizer behind the tracker boundary.
pub fn vision_recognizer() -> VisionRecognizer {
    VisionRecognizer(vbt_vision::recognizer::Recognizer::new().expect("load recognizer"))
}

/// The two team tag plate boxes, re-exported for tests that paste plates.
pub const PLATE_LEFT: vbt_vision::frame::Rect = TEAM_LEFT;
pub const PLATE_RIGHT: vbt_vision::frame::Rect = TEAM_RIGHT;
