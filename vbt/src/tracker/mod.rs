//! The 1 fps tracker state machine. It turns frames into a map score, at most one
//! round at a time, and decides map ends, hidden finals and side corrections.

pub mod observations;

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use vbt_vision::frame::Frame;
use vbt_vision::recognizer::PauseRead;
use vbt_vision::{learned, regions, scene, templates};

/// The decoded frame size the runner produces.
pub const FRAME_WIDTH: usize = 1280;
pub const FRAME_HEIGHT: usize = 720;
/// The size of one RGB24 frame.
pub const FRAME_BYTES: usize = FRAME_WIDTH * FRAME_HEIGHT * 3;

/// Below this the recognizer re-reads the score.
pub const TEMPLATE_MIN_CONFIDENCE: f64 = 0.65;
/// The recognizer also re-reads scores and the map strip this often.
pub const RECOGNIZER_EVERY_SECOND: i64 = 5;
/// Fastest spacing between two score changes: 30 s buy + 7 s end phase + ~8 s of play.
pub const MIN_ROUND_SECONDS: i64 = 45;
/// A rejected series must stand this long, uncontested, before it can replace the run's score.
pub const CONTENDER_MIN_SECONDS: i64 = 60;
/// Without advancing, a rejected series must stand this long before it replaces the run's score.
pub const CONTENDER_STABLE_SECONDS: i64 = 120;
/// Confirmed reads a series needs before it can replace the run's score.
pub const CONTENDER_MIN_READS: i64 = 8;
/// A series with no supporting read for this long starts over.
pub const CONTENDER_GAP_SECONDS: i64 = 180;
/// A series this old with this many reads makes the run's score doubtful, so it publishes as an error.
pub const CONTENDER_DOUBT_SECONDS: i64 = 20;
pub const CONTENDER_DOUBT_READS: i64 = 3;
/// At match point, this long without the gameplay HUD arms a hidden end.
pub const PENDING_END_SECONDS: i64 = 20;
/// This long without a matching read, the tracker reports an error.
pub const UNREAD_SECONDS: i64 = 90;
/// An unchanged score is written again this often as a heartbeat.
pub const HEARTBEAT_SECONDS: u64 = 30;
/// How long Riot must keep reporting the exact reverse of a run's sides before the run swaps.
pub const REVERSAL_STABLE_SECONDS: i64 = 120;
/// A pause starts on two consecutive overlay frames, and ends on two consecutive
/// frames without one, each showing the gameplay HUD.
pub const PAUSE_CONFIRM_FRAMES: i64 = 2;
pub const PAUSE_CLEAR_FRAMES: i64 = 2;
/// How often the experimental table reader may look at a gameplay frame.
pub const SCOREBOARD_EVERY_SECONDS: i64 = 5;
/// Confirmed scheduled reads a mid-map join needs before it adopts the on-screen score.
pub const JOIN_MIN_READS: i64 = 8;
/// Seconds (no pause allowed) a join series must stand before it is adopted.
pub const JOIN_MIN_SECONDS: i64 = 60;
/// A join series with no supporting read for this long starts over.
pub const JOIN_GAP_SECONDS: i64 = 30;

/// The match id the tracker uses while Riot lists none.
pub const UNKNOWN_MATCH: &str = "unknown";

/// The event kinds the ledger carries.
pub mod kind {
    pub const MAP_START: &str = "map_start";
    /// A run opened mid-map from corroborated reads rather than a 0-0 start.
    pub const MAP_JOIN: &str = "map_join";
    pub const ROUND: &str = "round";
    pub const SYNC: &str = "sync";
    pub const REJECT: &str = "reject";
    pub const SCORE_HELD: &str = "score_held";
    pub const SCORE_ROLLBACK: &str = "score_rollback";
    pub const SCORE_CORROBORATED: &str = "score_corroborated";
    pub const RESYNC: &str = "resync";
    pub const MAP_END: &str = "map_end";
    pub const SIDES_CORRECTED: &str = "sides_corrected";
    pub const PENDING_END: &str = "pending_end";
    pub const PENDING_CLEARED: &str = "pending_cleared";
    pub const PAUSE_START: &str = "pause_start";
    pub const PAUSE_END: &str = "pause_end";
    pub const FRAME_OBSERVATION: &str = "frame_observation";
    pub const VIDEO_ACCEPTED: &str = "video_accepted";
    pub const SCOREBOARD_OBSERVATION: &str = "scoreboard_observation";
}

/// The event sources the tracker attributes changes to.
pub mod source {
    pub const NORMAL: &str = "normal";
    pub const RIOT: &str = "riot";
    pub const NEXT_MAP: &str = "next_map";
    pub const OVERLAY_GONE: &str = "overlay_gone";
    pub const MATCH_CHANGE: &str = "match_change";
    pub const CONFIRMED_HELD_SCORE: &str = "confirmed_held_score";
    /// The contender series advanced its way to the run's new score.
    pub const CONTENDER_SERIES: &str = "contender_series";
    /// The contender series held still long enough to replace the run's score.
    pub const CONTENDER_STABLE: &str = "contender_stable";
}

/// A score keyed by team code. `BTreeMap` keeps the serialized order sorted.
pub type ScoreMap = BTreeMap<String, i32>;

/// Rounds won on the current map, by side of the score bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Score {
    pub left: i32,
    pub right: i32,
}

/// The teams by side of the score bar.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Sides {
    pub blue: String,
    pub red: String,
}

/// A team as Riot lists it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Team {
    pub id: String,
    pub code: String,
    pub name: String,
}

/// Riot's view of one game, logged for diagnosing side reversals.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RiotGame {
    pub state: String,
    pub sides: Sides,
    /// False when Riot gives no explicit sides for the game.
    pub listed: bool,
}

/// Whether the tracker is reading a map, published as `status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    #[default]
    Ok,
    Error,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            Status::Error => "error",
        }
    }
}

/// A broadcast pause in progress: what its overlay says and when it began. It lives
/// on the tracker, not on a map run, because a pause can precede the first map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pause {
    pub kind: String,
    pub reason: String,
    /// The tracker's second the first overlay frame was seen.
    pub since: i64,
}

/// What was last published.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Published {
    pub map_number: i64,
    pub score: ScoreMap,
    pub status: Status,
    /// The team code on the broadcast's left score plate.
    pub blue: String,
    /// The pause overlay's kind, `""` while play is running.
    pub pause: String,
    /// The reason suffix that overlay carries, when a template read one.
    pub pause_reason: String,
}

/// The experimental full-table read, as the ledger records it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScoreboardObservation {
    pub round_displayed: Option<i64>,
    pub layout: String,
    pub teams: Vec<ScoreboardTeam>,
    pub players: Vec<ScoreboardPlayer>,
}

/// One side of the scoreboard observation's header.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScoreboardTeam {
    pub name: String,
    pub score: Option<i64>,
}

/// One table row of the scoreboard observation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScoreboardPlayer {
    pub team: String,
    pub name: String,
    pub kda: [i64; 3],
}

/// A push failure, carried to the tracker's boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushError(pub String);

impl std::fmt::Display for PushError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PushError {}

/// Identity and results source the tracker needs.
pub trait Riot {
    fn live_now(&self) -> bool;
    fn has_event(&self) -> bool;
    fn match_id(&self) -> String;
    fn teams(&self) -> Vec<Team>;
    fn sides(&self, match_id: &str, map_number: i64) -> Option<Sides>;
    fn listed_order(&self, match_id: &str) -> Sides;
    fn map_number(&self, maps_played: i64) -> i64;
    fn game_state(&self, map_number: i64) -> Option<String>;
    fn game_wins(&self) -> ScoreMap;
    fn best_of(&self) -> i64;
}

/// The optional series map-order source: the backend's video context, which names
/// each game's map from the veto, ahead of the broadcast.
pub trait MapOrder {
    fn map_name(&self, match_id: &str, map_number: i64) -> String;
    /// The veto/context's competitors, when the source knows them.
    fn team_codes(&self, _match_id: &str) -> Option<Vec<String>> {
        None
    }
}

impl<T: MapOrder + ?Sized> MapOrder for std::sync::Arc<T> {
    fn map_name(&self, match_id: &str, map_number: i64) -> String {
        (**self).map_name(match_id, map_number)
    }

    fn team_codes(&self, match_id: &str) -> Option<Vec<String>> {
        (**self).team_codes(match_id)
    }
}

/// The second opinion the templates defer to.
pub trait Recognizer {
    fn score(&self, crop: &Frame) -> Option<i32>;
    fn map(&self, strip: &Frame) -> String;
    fn pause(&self, header: &Frame, card: &Frame) -> (PauseRead, bool);
}

/// Hands the score to the backend.
pub trait Publisher {
    fn enabled(&self) -> bool;
    fn publish(
        &self,
        map_number: i64,
        score: &ScoreMap,
        teams: &[Team],
        blue: &str,
        status: Status,
        pause: Option<&Pause>,
    ) -> Result<(), PushError>;
}

/// Reads team tags by OCR, for the plates the learned crops cannot, and teaches them.
pub trait TagReader {
    /// Each plate (left, right) as one of `codes`, `""` when refused.
    fn read(&self, frame: &Frame, codes: [&str; 2]) -> [String; 2];
}

/// The experimental full-table reader. Its observations are logged and emitted,
/// never merged into the score.
pub trait ScoreboardReader {
    fn analyze(&mut self, frame: &Frame) -> Result<Option<ScoreboardObservation>, String>;
}

/// The recognizer adapter over the vision crate.
pub struct VisionRecognizer(pub vbt_vision::recognizer::Recognizer);

impl Recognizer for VisionRecognizer {
    fn score(&self, crop: &Frame) -> Option<i32> {
        self.0.score(crop)
    }

    fn map(&self, strip: &Frame) -> String {
        self.0.map(strip)
    }

    fn pause(&self, header: &Frame, card: &Frame) -> (PauseRead, bool) {
        self.0.pause(header, card)
    }
}

/// One tracked change, emitted for logging and for replay comparison.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Event {
    pub kind: String,
    pub second: i64,
    /// Map, MatchID and MapNumber identify the run a change belongs to.
    pub map: String,
    pub match_id: String,
    pub map_number: i64,
    /// The score keyed by team code.
    pub score: Option<ScoreMap>,
    pub read_score: Option<ScoreMap>,
    /// resync: the score it replaced.
    pub previous_score: Option<ScoreMap>,
    pub winner: String,
    pub source: String,
    pub held: bool,
    pub exact: Option<bool>,
    /// pause_start/pause_end: the overlay's kind.
    pub pause: String,
    /// Its reason suffix, when a template read one.
    pub reason: String,
    /// pause_start/pause_end: the tracker second the pause's first overlay frame was seen.
    pub pause_since: i64,
    /// map_start/sides_corrected: Riot's state and explicit sides for the game.
    pub riot_state: String,
    pub riot_blue: String,
    pub riot_red: String,
    /// Shadow evidence, independent of `score`.
    pub visual_left: String,
    pub visual_right: String,
    pub left_score: Option<i32>,
    pub right_score: Option<i32>,
    pub video_score: Option<ScoreMap>,
    pub plate_ok: bool,
    pub left_confidence: f64,
    pub right_confidence: f64,
    pub scene: String,
    pub scoreboard: Option<ScoreboardObservation>,
}

/// A run of rejected but mutually consistent reads that may replace the run's score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Contender {
    /// The first read of the series.
    pub first: Score,
    /// The series' current score.
    pub latest: Score,
    /// The second of `first`.
    pub since: i64,
    /// The second of the unconfirmed candidate read that `first` confirmed: the
    /// series' strip evidence counts from here.
    pub opened: i64,
    /// The second of the latest supporting read.
    pub last_second: i64,
    /// Confirmed supporting reads (`record_score` calls).
    pub reads: i64,
    /// Rounds the series itself gained since `first`.
    pub advances: i64,
    /// `Tracker::paused_total` when the series started. A later cleared pause
    /// must not retroactively inflate the series' unpaused play time.
    pub paused_at_start: i64,
}

/// A run of corroborated mid-map reads that may open a run at a non-zero score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinCandidate {
    pub match_id: String,
    pub map_number: i64,
    pub name: String,
    pub sides: Sides,
    pub latest: Score,
    /// The second of the first read of the series.
    pub since: i64,
    /// The second of the latest supporting read.
    pub last_second: i64,
    /// Confirmed supporting reads.
    pub reads: i64,
    /// Rounds the series itself gained since `since`.
    pub advances: i64,
}

/// One detected map: identity from Riot, score from the broadcast.
#[derive(Debug, Clone, Default)]
pub struct MapRun {
    pub name: String,
    pub match_id: String,
    pub map_number: i64,
    /// Riot's blue team, which the broadcast shows on the left.
    pub left_code: String,
    pub right_code: String,
    pub score: Score,
    pub ended: bool,
    /// The second the score plate was last on screen with no change pending: the
    /// clock a sync's jump allowance runs on, so only hidden play can earn one.
    pub last_hud_second: Option<i64>,
    /// `Tracker::paused_total` at the moment `last_hud_second` was set.
    pub paused_at_hud: i64,
    /// The second the run's score was last read back or accepted: the clock a
    /// contender's jump allowance runs on, which unreadable and rejected reads do
    /// not reset.
    pub last_known_second: Option<i64>,
    /// `Tracker::paused_total` at the moment `last_known_second` was set.
    pub paused_at_known: i64,
    pub last_read_second: i64,
    pub candidate: Option<Score>,
    /// The second `candidate` was first read.
    pub candidate_since: i64,
    /// Suspect marks a candidate that emerged from a visible but unreadable plate.
    pub suspect: bool,
    /// A one-round, map-winning gameplay read of the match-point leader, while no
    /// second read has confirmed it: the second it was read, the pair, and whether
    /// it emerged from an unreadable plate (suspect, so adoption keeps the hold).
    pub final_read: Option<(i64, Score, bool)>,
    /// While a suspect change's publication is withheld: the second it was last recorded,
    /// and the last corroborated score keyed by team code, before the change.
    pub held: Option<(i64, ScoreMap)>,
    /// A rejected but consistent read series that may replace the run's score.
    pub contender: Option<Contender>,
    /// What the strip said when this map began.
    pub strip_at_start: String,
    /// Riot's series tally when this map began, settled once Riot completes the game before it.
    pub wins_baseline: ScoreMap,
    /// Whether `wins_baseline` has been settled against the previous game's completion.
    pub baseline_settled: bool,
    /// The team the map ended for, `""` while it runs or when its end named none.
    pub winner: String,
    /// `winner` is the match-point leader credited at the next map's start before Riot
    /// completed the game: a guess Riot's later credit confirms or corrects.
    pub guessed: bool,
    /// A run bound before Riot's details arrived, whose codes are the placeholders.
    pub provisional: bool,
    /// Riot's view of the game when it was last logged.
    pub riot: RiotGame,
    /// The second Riot was first seen reporting the exact reverse of the run's sides.
    pub reversal_since: Option<i64>,
    /// The last independently confirmed plate mapping.
    pub visual: Sides,
}

impl MapRun {
    /// The current score keyed by team code.
    pub fn score_by_code(&self) -> ScoreMap {
        let mut score = ScoreMap::new();
        score.insert(self.left_code.clone(), self.score.left);
        score.insert(self.right_code.clone(), self.score.right);
        score
    }

    /// The held score as a left/right pair.
    fn held_pair(&self) -> Option<Score> {
        self.held.as_ref().map(|(_, held)| Score {
            left: held.get(&self.left_code).copied().unwrap_or(0),
            right: held.get(&self.right_code).copied().unwrap_or(0),
        })
    }

    /// The team one round from winning the map, or `""`.
    pub fn match_point_leader(&self) -> String {
        let Score { left, right } = self.held_pair().unwrap_or(self.score);
        let high = left.max(right);
        let low = left.min(right);
        if (high == 12 && low <= 11) || (high >= 12 && high - low == 1) {
            if left > right {
                return self.left_code.clone();
            }
            return self.right_code.clone();
        }
        String::new()
    }
}

/// The observer callback the tracker feeds every event to.
pub type EventCallback = Box<dyn FnMut(&Event) + Send>;

/// The causal 1 fps tracker for the live broadcast.
pub struct Tracker {
    pub templates: templates::Templates,
    pub recognizer: Option<Box<dyn Recognizer + Send>>,
    pub riot: Box<dyn Riot + Send>,
    pub publish: Option<Box<dyn Publisher + Send>>,
    pub map_order: Option<Box<dyn MapOrder + Send>>,

    /// `maps` holds every run, `active` is an index into it. Runs are never removed.
    pub maps: Vec<MapRun>,
    pub active: Option<usize>,
    pub series_wins: BTreeMap<String, ScoreMap>,
    pub veto_candidate_num: i64,
    pub veto_candidate_n: i64,
    pub veto_candidate_name: String,
    /// The last name a 0-0 map check read from the strip, and the second it was read.
    pub last_strip: String,
    pub last_strip_second: i64,
    pub last_match_id: String,
    pub last_gameplay_full: Option<i64>,
    pub pending_since: Option<i64>,
    pub published: Option<Published>,
    pub published_at: Instant,
    /// Injectable clock for heartbeat tests.
    pub now: Box<dyn Fn() -> Instant + Send>,
    /// Whether the last frame's plate was visible but unreadable.
    pub prev_plate_withheld: bool,
    /// The broadcast pause in progress, or none. It precedes any map run.
    pub pause: Option<Pause>,

    pub pause_seen: i64,
    pub pause_first: i64,
    pub unpause_seen: i64,
    /// Seconds spent in pauses that have ended.
    pub paused_seconds: i64,
    /// This frame's team plate read; empty off gameplay.
    pub plate: learned::Read,
    /// A corroborated mid-map read series that may open a run at a non-zero score.
    pub join: Option<JoinCandidate>,

    /// Called for every tracked change.
    pub on_event: Option<EventCallback>,
    /// Shadow forbids publishing even if an enabled Publisher is supplied.
    pub shadow: bool,
    /// The learned tag crops. Without them no tag is read and no map in progress is joined.
    pub plates: Option<learned::Crops>,
    /// The OCR that reads what the crops cannot and teaches them.
    pub tag_reader: Option<Box<dyn TagReader + Send>>,
    /// The experimental full-table reader.
    pub scoreboard: Option<Box<dyn ScoreboardReader + Send>>,
    /// A replay-only substitute for the match context's team tags.
    pub expected_teams: Vec<String>,
    pub plate_candidate: Sides,
    pub plate_count: i64,
    pub visual: Sides,
    pub last_scoreboard_second: i64,
}

impl Tracker {
    /// Builds a tracker. `recognizer`, `publish` and `map_order` may be `None`;
    /// `riot` must not be.
    pub fn new(
        templates: templates::Templates,
        recognizer: Option<Box<dyn Recognizer + Send>>,
        riot: Box<dyn Riot + Send>,
        publish: Option<Box<dyn Publisher + Send>>,
        map_order: Option<Box<dyn MapOrder + Send>>,
    ) -> Tracker {
        Tracker {
            templates,
            recognizer,
            riot,
            publish,
            map_order,
            maps: Vec::new(),
            active: None,
            series_wins: BTreeMap::new(),
            veto_candidate_num: 0,
            veto_candidate_n: 0,
            veto_candidate_name: String::new(),
            last_strip: String::new(),
            last_strip_second: 0,
            last_match_id: String::new(),
            last_gameplay_full: None,
            pending_since: None,
            published: None,
            published_at: Instant::now(),
            now: Box::new(Instant::now),
            prev_plate_withheld: false,
            pause: None,
            pause_seen: 0,
            pause_first: 0,
            unpause_seen: 0,
            paused_seconds: 0,
            plate: learned::Read::default(),
            join: None,
            on_event: None,
            shadow: false,
            plates: None,
            tag_reader: None,
            scoreboard: None,
            expected_teams: Vec::new(),
            plate_candidate: Sides::default(),
            plate_count: 0,
            visual: Sides::default(),
            last_scoreboard_second: -SCOREBOARD_EVERY_SECONDS,
        }
    }

    /// The active run, or `None`.
    pub fn active_run(&self) -> Option<&MapRun> {
        self.active.map(|i| &self.maps[i])
    }

    /// The active run mutably, or `None`.
    pub fn active_run_mut(&mut self) -> Option<&mut MapRun> {
        self.active.map(|i| &mut self.maps[i])
    }

    /// Pause seconds so far, including the pause in progress.
    pub fn paused_total(&self, second: i64) -> i64 {
        self.paused_seconds + self.pause.as_ref().map_or(0, |p| (second - p.since).max(0))
    }

    /// Unpaused seconds from `since` (when the pause total was `paused_at`) to `second`.
    fn unpaused_since(&self, since: Option<i64>, paused_at: i64, second: i64) -> i64 {
        since.map_or(0, |last| {
            ((second - last) - (self.paused_total(second) - paused_at)).max(0)
        })
    }

    /// Seconds the HUD was hidden while play could run.
    fn play_seconds(&self, idx: usize, second: i64) -> i64 {
        let run = &self.maps[idx];
        self.unpaused_since(run.last_hud_second, run.paused_at_hud, second)
    }

    /// Seconds since the run's score was last read back or accepted while play could run.
    fn known_play_seconds(&self, idx: usize, second: i64) -> i64 {
        let run = &self.maps[idx];
        self.unpaused_since(run.last_known_second, run.paused_at_known, second)
    }

    /// Marks the HUD seen at `second`, remembering how much of the elapsed time was pause.
    fn mark_hud(&mut self, idx: usize, second: i64) {
        let paused = self.paused_total(second);
        self.maps[idx].last_hud_second = Some(second);
        self.maps[idx].paused_at_hud = paused;
    }

    /// Marks the run's score known at `second`, remembering how much of the elapsed time was pause.
    fn mark_known(&mut self, idx: usize, second: i64) {
        let paused = self.paused_total(second);
        self.maps[idx].last_known_second = Some(second);
        self.maps[idx].paused_at_known = paused;
    }

    /// False only when this map's plates have been readable but this frame's are not:
    /// the evidence a broadcast cut to another match.
    fn plates_agree(&self, idx: usize) -> bool {
        self.maps[idx].visual.blue.is_empty() || self.plate.ok
    }

    /// Riot's live match id, or `UNKNOWN_MATCH` while Riot lists none.
    fn match_id_or_unknown(&self) -> String {
        let match_id = self.riot.match_id();
        if match_id.is_empty() {
            UNKNOWN_MATCH.to_string()
        } else {
            match_id
        }
    }

    /// Processes one frame and applies every live rule.
    pub fn step(&mut self, second: i64, frame: &Frame) {
        let match_id = self.match_id_or_unknown();
        if !self.last_match_id.is_empty() && self.last_match_id != match_id {
            self.visual = Sides::default();
            self.plate_candidate = Sides::default();
            self.plate_count = 0;
            self.reset_map_candidate();
            self.last_strip.clear();
            self.join = None;
            self.pending_since = None;
            self.last_gameplay_full = None;
            // The pause belongs to the previous match's broadcast; it must not keep
            // the next match's maps healthy or suppress their hidden ends.
            if self.pause.is_some() {
                self.clear_pause(second, source::MATCH_CHANGE);
            } else {
                self.pause_seen = 0;
            }
        }
        self.last_match_id = match_id.clone();
        let (scene, read) = self.process_frame(second, frame);
        self.observe_scoreboard(second, frame, scene);
        self.correct_sides(second);
        self.settle_baseline(second);
        // A pause explains a missing HUD, so the hidden-end rule stays out of the way.
        if self.pause.is_none() {
            self.check_pending_end(second, scene, scene::score_visible(frame), read);
        }
        // Nothing is published without a live Riot match, and only while the active run
        // is that match's own.
        let publishes = self.publish.as_ref().is_some_and(|p| p.enabled())
            && !self.shadow
            && self.riot.live_now()
            && self.riot.has_event();
        if publishes
            && let Some(idx) = self.active
            && self.maps[idx].match_id == match_id
        {
            self.publish_run(idx, second);
        }
    }

    /// Reads one frame and advances map and score state, returning the frame's scene
    /// and its score pair, when its read produced one.
    fn process_frame(&mut self, second: i64, frame: &Frame) -> (scene::Scene, Option<Score>) {
        let frame_scene = scene::scene_of(frame);
        let prev_withheld = self.prev_plate_withheld; // the previous frame's plate state
        self.update_pause(second, frame);
        let score_on_screen = frame_scene == scene::Scene::Gameplay
            || (frame_scene == scene::Scene::Replay && scene::score_visible(frame));
        let mut pair = Score::default();
        let mut have_pair = false;
        if score_on_screen {
            let read = self.read_pair(frame, second % RECOGNIZER_EVERY_SECOND == 0);
            pair = read.0;
            have_pair = read.1;
        }
        if have_pair
            && let Some(idx) = self.active
            && !self.maps[idx].ended
            && pair != self.maps[idx].score
        {
            // every change is confirmed by the recognizer
            let read = self.read_pair(frame, true);
            pair = read.0;
            have_pair = read.1;
        }
        self.observe_frame(second, frame, frame_scene, pair, have_pair);
        // A plate that is visible but unreadable marks how partial occlusions present.
        self.prev_plate_withheld = score_on_screen && !have_pair;

        if frame_scene == scene::Scene::Gameplay
            && have_pair
            && pair == Score::default()
            && second % RECOGNIZER_EVERY_SECOND == 0
        {
            let observed = self.read_map_strip(frame);
            tracing::debug!(
                second,
                observed = %observed,
                candidate = %self.veto_candidate_name,
                number = self.veto_candidate_num,
                "map check"
            );
            let match_id = self.match_id_or_unknown();
            if self.series_decided(&match_id) {
                // The series is over; a 0-0 post-match HUD cannot reopen the run.
                self.reset_map_candidate();
            } else {
                // A run starts only under a name: the veto's, or the strip's when the
                // veto has none. Riot's explicit in-progress game number decides which
                // game it is unless it lags the tracked run, never whether one starts.
                let mut number = self.next_map_number(&observed, &match_id);
                let (in_progress, has_in_progress) = self.in_progress_game(&match_id);
                if has_in_progress {
                    number = in_progress;
                }
                if !observed.is_empty() {
                    self.last_strip = observed.clone();
                    self.last_strip_second = second;
                }
                let veto_number = self.veto_number(&match_id, &observed);
                // Riot can list the previous game in progress for minutes after the
                // next map's 0-0 is on screen, and can flap back to an earlier game.
                // A changed strip once the run has ended or reached match point is
                // the next game whatever Riot says, and a start never goes back to a
                // game this match has already tracked.
                let run = self
                    .active
                    .map(|i| &self.maps[i])
                    .filter(|r| r.match_id == match_id);
                // A strip naming the run's own or an earlier veto game (a replay or recap
                // of a past map) is never evidence of the next one.
                let strip_changed = run.is_some_and(|r| {
                    !r.strip_at_start.is_empty()
                        && !observed.is_empty()
                        && !observed.eq_ignore_ascii_case(&r.strip_at_start)
                        && (veto_number == 0 || veto_number > r.map_number)
                });
                // A run within two rounds of a win can have ended behind a cut that hid
                // its last rounds. Its boundary then also needs the strip to name a later
                // game of a known veto, since no winner proves the map over.
                let boundary = run.is_some_and(|r| {
                    let late = r.score.left.max(r.score.right) >= 11
                        && (veto_number > r.map_number
                            || self.veto_name(&match_id, r.map_number + 1).is_empty());
                    strip_changed && (r.ended || !r.match_point_leader().is_empty() || late)
                });
                if let Some(r) = run
                    && boundary
                    && number <= r.map_number
                {
                    number = r.map_number + 1;
                }
                // The strip's place in the veto numbers the game without Riot, so it
                // overrides a lagging number once nothing tracked is still running.
                if veto_number > number && run.is_none_or(|r| r.ended || boundary) {
                    number = veto_number;
                }
                // A changed strip or Riot's explicit in-progress game can corroborate
                // a new map during a run. A persistent wrong strip cannot.
                let new_game = run.is_none_or(|r| {
                    number > r.map_number && (r.ended || strip_changed || has_in_progress)
                });
                let veto = self.veto_name(&match_id, number);
                let name = if veto.is_empty() {
                    observed.clone()
                } else {
                    veto.clone()
                };
                if !name.is_empty() && new_game {
                    let candidate_name = if veto.is_empty() && !observed.is_empty() {
                        observed.clone()
                    } else {
                        String::new()
                    };
                    if self.veto_candidate_num == number
                        && self
                            .veto_candidate_name
                            .eq_ignore_ascii_case(&candidate_name)
                    {
                        self.veto_candidate_n += 1;
                    } else {
                        self.veto_candidate_num = number;
                        self.veto_candidate_n = 1;
                        self.veto_candidate_name = candidate_name;
                    }
                    if self.veto_candidate_n >= 2 {
                        self.start_map(&name, number, &observed, second);
                        self.reset_map_candidate();
                    }
                } else {
                    self.reset_map_candidate();
                }
            }
        } else if second % RECOGNIZER_EVERY_SECOND == 0 {
            // A scheduled observation that is not a 0-0 gameplay read contradicts a
            // pending map-start confirmation.
            self.reset_map_candidate();
            tracing::debug!(
                second,
                scene = %frame_scene,
                has_pair = have_pair,
                pair = ?(pair.left, pair.right),
                "frame"
            );
        }
        // A scheduled read with the score on screen also feeds the mid-map join:
        // `read_pair` forced the recognizer on this second, so the score is confirmed.
        if second % RECOGNIZER_EVERY_SECOND == 0 {
            self.consider_join(second, frame, frame_scene, pair, have_pair);
        }

        let read = have_pair.then_some(pair);
        let Some(idx) = self.active else {
            return (frame_scene, read);
        };
        if self.maps[idx].ended {
            return (frame_scene, read);
        }
        if !have_pair || pair == self.maps[idx].score {
            self.maps[idx].candidate = None;
            self.maps[idx].suspect = false;
            if have_pair {
                // A readable return to the run's own score contradicts any
                // remembered map-winning read.
                self.maps[idx].final_read = None;
                self.mark_known(idx, second);
            }
        } else if self.maps[idx].candidate == Some(pair) {
            let suspect = self.maps[idx].suspect;
            self.maps[idx].candidate = None;
            self.maps[idx].suspect = false;
            // The read is confirmed now, so the ordinary path owns it; the
            // remembered copy must not survive to double-record it.
            self.maps[idx].final_read = None;
            self.record_score(idx, pair, second, suspect);
        } else {
            self.maps[idx].candidate = Some(pair);
            self.maps[idx].candidate_since = second;
            self.maps[idx].suspect = prev_withheld;
            // A map-winning read may be the last frame before the HUD vanishes;
            // keep it in case no second read can confirm it. Only gameplay evidence
            // qualifies: a replay frame must not both create and consume it within
            // this step, and a remembered final that this read does not match is
            // contradicted.
            let leader = self.maps[idx].match_point_leader();
            let leader_rounds = if leader == self.maps[idx].left_code {
                pair.left
            } else {
                pair.right
            };
            let winning = !leader.is_empty()
                && leader_rounds >= 13
                && (pair.left - pair.right).abs() >= 2
                && valid_map_score(pair)
                && advance_of(self.maps[idx].score, pair) == Some(1);
            if frame_scene == scene::Scene::Gameplay && winning {
                self.maps[idx].final_read = Some((second, pair, prev_withheld));
            } else if self.maps[idx]
                .final_read
                .is_some_and(|(_, final_pair, _)| final_pair != pair)
            {
                self.maps[idx].final_read = None;
            }
        }
        if have_pair
            && pair == self.maps[idx].score
            && second % RECOGNIZER_EVERY_SECOND == 0
            && self.maps[idx]
                .held
                .as_ref()
                .is_some_and(|(since, _)| second >= since + RECOGNIZER_EVERY_SECOND)
        {
            // A periodic recognizer re-check after the hold window has re-read the
            // score, so the change outlasted the transient overlays.
            self.maps[idx].held = None;
            tracing::info!(
                second,
                map = %self.maps[idx].name,
                score = ?self.maps[idx].score_by_code(),
                "score corroborated for publish"
            );
            let event = Event {
                read_score: Some(self.maps[idx].score_by_code()),
                ..self.run_event(idx, kind::SCORE_CORROBORATED, second)
            };
            self.emit(event);
            self.end_map_if_complete(idx, second);
        }
        if have_pair && pair == self.maps[idx].score {
            self.maps[idx].last_read_second = second;
            if self.maps[idx].contender.take().is_some() {
                tracing::info!(
                    second,
                    map = %self.maps[idx].name,
                    "contender contradicted by the run's score"
                );
            }
        }
        // Any plate on screen with no change pending restarts the sync's allowance, so
        // a steady visible misread can never pass as a sync. Recovering from one is the
        // contender's job, on the `last_known_second` clock instead.
        if score_on_screen && self.maps[idx].candidate.is_none() {
            self.mark_hud(idx, second);
        }
        (frame_scene, read)
    }

    fn reset_map_candidate(&mut self) {
        self.veto_candidate_num = 0;
        self.veto_candidate_n = 0;
        self.veto_candidate_name.clear();
    }

    fn read_map_strip(&self, frame: &Frame) -> String {
        let Some(recognizer) = &self.recognizer else {
            return String::new();
        };
        recognizer.map(&frame.crop_raw(regions::MAP_STRIP))
    }

    /// Settles one side's score: a confident template wins unless the recognizer was
    /// forced or disagrees; an unsure template never decides alone.
    fn read_side(
        &self,
        frame: &Frame,
        region: regions::Rect,
        template: vbt_vision::templates::Read,
        need_recognizer: bool,
        force_recognizer: bool,
    ) -> (i32, bool) {
        let mut recognized: Option<i32> = None;
        if need_recognizer && let Some(recognizer) = &self.recognizer {
            recognized = recognizer.score(&frame.crop(region));
        }
        let confident = template.value >= 0 && template.confidence >= TEMPLATE_MIN_CONFIDENCE;
        if confident
            && !force_recognizer
            && (recognized.is_none() || recognized == Some(template.value))
        {
            return (template.value, true);
        }
        if let Some(value) = recognized {
            return (value, true);
        }
        if confident {
            return (template.value, true);
        }
        (0, false)
    }

    /// Reads both score plates, asking the recognizer when forced or when a
    /// template is unsure.
    fn read_pair(&self, frame: &Frame, force_recognizer: bool) -> (Score, bool) {
        let boxes = [regions::SCORE_LEFT, regions::SCORE_RIGHT];
        let mut templates_read = [vbt_vision::templates::Read::default(); 2];
        let mut unsure = false;
        for (i, region) in boxes.iter().enumerate() {
            templates_read[i] = self.templates.read_score(&frame.crop(*region));
            if templates_read[i].value < 0 || templates_read[i].confidence < TEMPLATE_MIN_CONFIDENCE
            {
                unsure = true;
            }
        }
        let run_continues = self.active_run().is_some_and(|run| !run.ended);
        let need_recognizer = force_recognizer || (unsure && run_continues);
        let (left, ok_left) = self.read_side(
            frame,
            boxes[0],
            templates_read[0],
            need_recognizer,
            force_recognizer,
        );
        let (right, ok_right) = self.read_side(
            frame,
            boxes[1],
            templates_read[1],
            need_recognizer,
            force_recognizer,
        );
        if !ok_left || !ok_right {
            return (Score::default(), false);
        }
        (Score { left, right }, true)
    }

    /// Reads the pause overlay and advances its confirm and clear counters.
    fn update_pause(&mut self, second: i64, frame: &Frame) {
        let Some(recognizer) = &self.recognizer else {
            return;
        };
        let (read, ok) = recognizer.pause(
            &frame.crop(regions::PAUSE_HEADER),
            &frame.crop(regions::PAUSE_CARD),
        );
        if !ok {
            self.pause_seen = 0;
            if self.pause.is_none() {
                return;
            }
            if !scene::score_visible(frame) {
                // The overlay can disappear before gameplay does; the pause holds
                // until the HUD is back.
                self.unpause_seen = 0;
                return;
            }
            self.unpause_seen += 1;
            if self.unpause_seen >= PAUSE_CLEAR_FRAMES {
                self.clear_pause(second, source::OVERLAY_GONE);
            }
            return;
        }
        self.unpause_seen = 0;
        // An overlay frame can precede a cut to a clip from anywhere, so no join
        // series survives it, even before the pause itself is confirmed (see
        // `consider_join`, which also refuses a read while `pause_seen > 0`).
        self.join = None;
        if self.pause.is_none() {
            if self.pause_seen == 0 {
                self.pause_first = second;
            }
            self.pause_seen += 1;
            if self.pause_seen >= PAUSE_CONFIRM_FRAMES {
                self.pause = Some(Pause {
                    kind: read.kind.clone(),
                    reason: read.reason.clone(),
                    since: self.pause_first,
                });
                tracing::info!(
                    second,
                    kind = %read.kind,
                    reason = %read.reason,
                    since = self.pause_first,
                    score = read.score,
                    "pause start"
                );
                let mut event = Event {
                    kind: kind::PAUSE_START.to_string(),
                    second,
                    pause: read.kind,
                    reason: read.reason,
                    pause_since: self.pause_first,
                    ..Default::default()
                };
                if let Some(idx) = self.active {
                    event.map = self.maps[idx].name.clone();
                    event.match_id = self.maps[idx].match_id.clone();
                    event.map_number = self.maps[idx].map_number;
                }
                self.emit(event);
            }
        }
    }

    /// Ends the pause in progress, whatever cleared it.
    pub fn clear_pause(&mut self, second: i64, source_name: &str) {
        let pause = self.pause.take();
        self.pause_seen = 0;
        self.unpause_seen = 0;
        let Some(pause) = pause else {
            return;
        };
        self.paused_seconds += (second - pause.since).max(0);
        tracing::info!(
            second,
            kind = %pause.kind,
            reason = %pause.reason,
            source = source_name,
            "pause end"
        );
        let mut event = Event {
            kind: kind::PAUSE_END.to_string(),
            second,
            pause: pause.kind.clone(),
            reason: pause.reason.clone(),
            pause_since: pause.since,
            source: source_name.to_string(),
            ..Default::default()
        };
        if let Some(idx) = self.active {
            event.map = self.maps[idx].name.clone();
            event.match_id = self.maps[idx].match_id.clone();
            event.map_number = self.maps[idx].map_number;
        }
        self.emit(event);
    }

    /// The veto's name for a game, or `""` without a map-order source or a known name.
    pub fn veto_name(&self, match_id: &str, map_number: i64) -> String {
        match &self.map_order {
            Some(order) => order.map_name(match_id, map_number),
            None => String::new(),
        }
    }

    /// The game the veto order lists `name` at, or 0 when the name is empty, unlisted,
    /// or listed twice.
    fn veto_number(&self, match_id: &str, name: &str) -> i64 {
        let mut games = (1..=5).filter(|n| {
            !name.is_empty() && self.veto_name(match_id, *n).eq_ignore_ascii_case(name)
        });
        match (games.next(), games.next()) {
            (Some(n), None) => n,
            _ => 0,
        }
    }

    /// The game number of the map about to start.
    pub fn next_map_number(&self, name: &str, match_id: &str) -> i64 {
        let (in_progress, has_in_progress) = self.in_progress_game(match_id);
        if has_in_progress {
            return in_progress;
        }
        for run in &self.maps {
            if run.match_id == match_id && !name.is_empty() && run.name.eq_ignore_ascii_case(name) {
                return run.map_number;
            }
        }
        // Advance past the highest game number this match has tracked.
        let highest = self
            .maps
            .iter()
            .filter(|r| r.match_id == match_id)
            .map(|r| r.map_number)
            .max()
            .unwrap_or(0);
        self.riot.map_number(highest)
    }

    /// Accumulates corroborated mid-map reads into a join series, and opens a run at
    /// the on-screen score once enough of them agree. Called on the scheduled reads
    /// only, where `read_pair` forced the recognizer and so confirmed the score.
    ///
    /// Every gate fails safe: a failing read resets the series rather than letting a
    /// rogue clip accumulate. A replay or a hidden HUD is not evidence either way and
    /// keeps the series, bounded by `JOIN_GAP_SECONDS`; a read whose strip cannot name
    /// the game, or that contradicts the veto's name, resets it.
    fn consider_join(
        &mut self,
        second: i64,
        frame: &Frame,
        frame_scene: scene::Scene,
        pair: Score,
        have_pair: bool,
    ) {
        // A pause means the broadcast may be showing a clip from anywhere: the
        // tracked 10-10 rogue came from one shown during a tech pause. A first
        // overlay frame is the same evidence one frame before the pause is
        // confirmed, so it also blocks — and resets — a join (see `update_pause`,
        // which invalidates the series on every frame that shows an overlay).
        if self.pause.is_some() || self.pause_seen > 0 {
            self.join = None;
            return;
        }
        let match_id = self.riot.match_id();
        // Riot must list the game itself as `inProgress` with explicit sides, so the
        // run is never provisional.
        let (number, in_progress) = self.in_progress_game(&match_id);
        let sides = in_progress
            .then(|| self.riot.sides(&match_id, number))
            .flatten();
        let eligible = self.riot.live_now()
            && self.riot.has_event()
            && !self.series_decided(&match_id)
            // A tracked map is never re-joined, and the tracker never joins a game
            // it has already moved past.
            && !self
                .maps
                .iter()
                .any(|run| run.match_id == match_id && run.map_number >= number);
        let Some(sides) = sides.filter(|_| eligible) else {
            self.join = None;
            return;
        };
        // The plates are the main defence against a clip from another match: the
        // reader only ever names the match's expected codes, and here they must sit in
        // Riot's order. A plate read on the wrong side resets the series.
        let plate = &self.plate;
        if (!plate.left.is_empty() && !plate.left.eq_ignore_ascii_case(&sides.blue))
            || (!plate.right.is_empty() && !plate.right.eq_ignore_ascii_case(&sides.red))
        {
            self.join = None;
            return;
        }
        if frame_scene != scene::Scene::Gameplay || !have_pair || !plate.ok {
            // A replay, a hidden HUD or an unread plate is not evidence either way: keep
            // the series, but a read gap longer than the series' patience starts it over.
            if self
                .join
                .as_ref()
                .is_some_and(|join| second - join.last_second > JOIN_GAP_SECONDS)
            {
                self.join = None;
            }
            return;
        }
        // 0-0 belongs to the existing start path; a finished or impossible score is
        // another map's end state.
        let finished = pair.left.max(pair.right) >= 13 && (pair.left - pair.right).abs() >= 2;
        if pair == Score::default() || !valid_map_score(pair) || finished {
            self.join = None;
            return;
        }
        let veto = self.veto_name(&match_id, number);
        let strip = self.read_map_strip(frame);
        // The strip is the only independent evidence that the picture is this game:
        // the plates prove the competitors and their displayed order, and Riot's
        // `inProgress` names the game number, but neither proves the video belongs to
        // that game. A strip that names nothing - the model read and the templates
        // were both inconclusive - or that disagrees with the veto resets the series:
        // a read that names nothing fails toward silence rather than let a clip from
        // another map join, and a contradictory picture is never accepted as true.
        // All 13 map names have a strip template; the five that are
        // out of the current rotation (Bind, Corrode, Fracture, Icebox, Pearl) were
        // harvested from the YouTube archives VLR links to, so the template fallback
        // covers them too.
        if strip.is_empty() || (!veto.is_empty() && !veto.eq_ignore_ascii_case(&strip)) {
            if !strip.is_empty() {
                tracing::warn!(
                    second,
                    r#match = %match_id,
                    number,
                    veto = %veto,
                    strip = %strip,
                    "join: strip disagrees with the veto; resetting the join"
                );
            }
            self.join = None;
            return;
        }
        let name = if veto.is_empty() { strip.clone() } else { veto };
        // A supporting read is the same score or a plausible advance of the series,
        // rate-limited exactly like the contender. Anything else starts a new series
        // on this read; gate 1 guarantees no pause ever overlaps one.
        let continued = self
            .join
            .take()
            .filter(|join| {
                join.match_id == match_id
                    && join.map_number == number
                    && join.sides == sides
                    && join.name.eq_ignore_ascii_case(&name)
                    && second - join.last_second <= JOIN_GAP_SECONDS
            })
            .and_then(|mut join| {
                if pair != join.latest {
                    let gain = advance_of(join.latest, pair)
                        .filter(|gain| *gain <= max_gain(second - join.last_second))?;
                    if join.advances + i64::from(gain) > (second - join.since) / MIN_ROUND_SECONDS {
                        return None;
                    }
                    join.latest = pair;
                    join.advances += i64::from(gain);
                }
                join.reads += 1;
                join.last_second = second;
                Some(join)
            });
        let join = continued.unwrap_or_else(|| JoinCandidate {
            match_id: match_id.clone(),
            map_number: number,
            name: name.clone(),
            sides: sides.clone(),
            latest: pair,
            since: second,
            last_second: second,
            reads: 1,
            advances: 0,
        });
        // Every counted read above required the strip to name this game, so the
        // series needs only time and reads.
        if join.reads >= JOIN_MIN_READS && second - join.since >= JOIN_MIN_SECONDS {
            tracing::warn!(
                second,
                map = %join.name,
                number,
                score = ?(join.latest.left, join.latest.right),
                reads = join.reads,
                "joining map in progress"
            );
            self.begin_run(
                &join.name,
                number,
                &strip,
                second,
                join.latest,
                kind::MAP_JOIN,
            );
        } else {
            self.join = Some(join);
        }
    }

    /// Starts tracking a map at 0-0, closing a previous map left running.
    pub fn start_map(&mut self, name: &str, map_number: i64, strip_at_start: &str, second: i64) {
        self.begin_run(
            name,
            map_number,
            strip_at_start,
            second,
            Score::default(),
            kind::MAP_START,
        );
    }

    /// Opens a run under `name` at `score`, recording `event_kind`. A run is an
    /// ordinary run from here on, whatever opened it.
    fn begin_run(
        &mut self,
        name: &str,
        map_number: i64,
        strip_at_start: &str,
        second: i64,
        score: Score,
        event_kind: &str,
    ) {
        if map_number < 1 {
            tracing::warn!(
                map = name,
                number = map_number,
                "refusing map start with invalid map number"
            );
            return;
        }
        let previous = self.active;
        let match_id = self.match_id_or_unknown();
        let mut name = name.to_string();
        let veto = self.veto_name(&match_id, map_number);
        if !veto.is_empty() {
            name = veto;
        }
        let (sides, ok) = match self.riot.sides(&match_id, map_number) {
            Some(sides) => (sides, true),
            None => (self.riot.listed_order(&match_id), false),
        };
        // Before Riot's details arrive, the listed order is the LEFT/RIGHT placeholder
        // pair. The run must not keep that identity for the rest of the map.
        let provisional = !ok && !self.riot.has_event();
        // The run being replaced settles against Riot before this baseline can absorb its win.
        self.settle_baseline(second);
        let wins_baseline = self.riot.game_wins();
        let run = MapRun {
            name: name.clone(),
            match_id: match_id.clone(),
            map_number,
            left_code: sides.blue.clone(),
            right_code: sides.red.clone(),
            score,
            last_hud_second: Some(second),
            paused_at_hud: self.paused_total(second),
            last_known_second: Some(second),
            paused_at_known: self.paused_total(second),
            last_read_second: second,
            strip_at_start: strip_at_start.to_string(),
            wins_baseline,
            provisional,
            ..Default::default()
        };
        let idx = self.maps.len();
        self.maps.push(run);
        self.active = Some(idx);
        // The run now owns the map: a join series that opened it is spent, and a 0-0
        // start must not leave one behind for the next read.
        self.join = None;
        // A previous map's last-known plates are not evidence for a new map's sides.
        if self.plate_count >= 2 {
            self.maps[idx].visual = self.visual.clone();
        }
        let riot_game = self.riot_game(idx);
        self.maps[idx].riot = riot_game.clone();
        if !self.series_wins.contains_key(&match_id) {
            let mut wins = ScoreMap::new();
            wins.insert(sides.blue.clone(), 0);
            wins.insert(sides.red.clone(), 0);
            self.series_wins.insert(match_id.clone(), wins);
        }
        if let Some(previous) = previous {
            let prev_match = self.maps[previous].match_id.clone();
            let prev_ended = self.maps[previous].ended;
            if prev_match == match_id && !prev_ended {
                self.close_at_next_map(previous, second);
            }
        }
        self.pending_since = None;
        tracing::info!(
            second,
            map = %name,
            number = map_number,
            r#match = %match_id,
            score = ?(score.left, score.right),
            sides = ?(sides.blue, sides.red),
            riot = ?riot_game,
            "map start"
        );
        let mut event = Event {
            riot_state: riot_game.state.clone(),
            ..self.run_event(idx, event_kind, second)
        };
        if riot_game.listed {
            event.riot_blue = riot_game.sides.blue.clone();
            event.riot_red = riot_game.sides.red.clone();
        }
        self.emit(event);
    }

    /// Applies a confirmed score: one round, a sync after a long hidden gap, or a rejection.
    /// An applied or rolled-back score is marked known; a rejected one feeds the contender.
    pub fn record_score(&mut self, idx: usize, pair: Score, second: i64, suspect: bool) {
        if let Some(held) = self.maps[idx].held_pair()
            && pair == held
        {
            self.maps[idx].score = held;
            self.maps[idx].held = None;
            self.maps[idx].candidate = None;
            self.maps[idx].suspect = false;
            // The baseline moved back, so a remembered final no longer applies.
            self.maps[idx].final_read = None;
            self.maps[idx].last_read_second = second;
            self.mark_hud(idx, second);
            self.mark_known(idx, second);
            let recovered = self.maps[idx].score_by_code();
            tracing::info!(
                second,
                map = %self.maps[idx].name,
                score = ?recovered,
                "suspect score rolled back to prior read"
            );
            let event = Event {
                read_score: Some(recovered),
                source: source::CONFIRMED_HELD_SCORE.to_string(),
                ..self.run_event(idx, kind::SCORE_ROLLBACK, second)
            };
            self.emit(event);
            return;
        }
        let run_score = self.maps[idx].score;
        let gained = Score {
            left: pair.left - run_score.left,
            right: pair.right - run_score.right,
        };
        let total = gained.left + gained.right;
        let play = self.play_seconds(idx, second);
        let plausible = gained.left >= 0
            && gained.right >= 0
            && valid_map_score(pair)
            && (total <= 1 || (total <= max_gain(play) && self.plates_agree(idx)));
        if !plausible {
            tracing::info!(
                second,
                from = ?(run_score.left, run_score.right),
                to = ?(pair.left, pair.right),
                play,
                map = %self.maps[idx].name,
                "rejected"
            );
            let mut read_score = ScoreMap::new();
            read_score.insert(self.maps[idx].left_code.clone(), pair.left);
            read_score.insert(self.maps[idx].right_code.clone(), pair.right);
            let event = Event {
                read_score: Some(read_score),
                held: self.maps[idx].held.is_some(),
                ..self.run_event(idx, kind::REJECT, second)
            };
            self.emit(event);
            if !suspect {
                self.track_contender(idx, pair, second);
            }
            return;
        }
        self.maps[idx].contender = None;
        let mut winner = String::new();
        if gained.left + gained.right == 1 {
            if gained.left > 0 {
                winner = self.maps[idx].left_code.clone();
            } else {
                winner = self.maps[idx].right_code.clone();
            }
        }
        if suspect {
            let held_score = match self.maps[idx].held.take() {
                Some((_, held_score)) => held_score,
                None => self.maps[idx].score_by_code(),
            };
            self.maps[idx].held = Some((second, held_score.clone()));
            let mut read_score = ScoreMap::new();
            read_score.insert(self.maps[idx].left_code.clone(), pair.left);
            read_score.insert(self.maps[idx].right_code.clone(), pair.right);
            let event = Event {
                score: Some(held_score),
                read_score: Some(read_score),
                held: true,
                ..self.run_event(idx, kind::SCORE_HELD, second)
            };
            self.emit(event);
        }
        self.maps[idx].score = pair;
        self.mark_known(idx, second);
        if suspect {
            tracing::info!(
                second,
                map = %self.maps[idx].name,
                score = ?self.maps[idx].score_by_code(),
                "score recorded from an unreadable plate; publication withheld"
            );
        }
        let held = self.maps[idx].held.is_some();
        if !winner.is_empty() {
            tracing::info!(
                second,
                map = %self.maps[idx].name,
                winner = %winner,
                score = ?self.maps[idx].score_by_code(),
                round = pair.left + pair.right,
                "round won"
            );
            let event = Event {
                winner,
                held,
                ..self.run_event(idx, kind::ROUND, second)
            };
            self.emit(event);
        } else {
            tracing::info!(
                second,
                map = %self.maps[idx].name,
                play,
                score = ?self.maps[idx].score_by_code(),
                "sync"
            );
            let event = Event {
                held,
                ..self.run_event(idx, kind::SYNC, second)
            };
            self.emit(event);
        }
        if self.maps[idx].held.is_none() {
            self.end_map_if_complete(idx, second);
        }
    }

    /// Accumulates a rejected but consistent series of reads. A series that stands
    /// long enough, within its play-time and jump allowance, replaces the run's
    /// score (`resync`).
    fn track_contender(&mut self, idx: usize, pair: Score, second: i64) {
        let run_score = self.maps[idx].score;
        let mirror = Score {
            left: run_score.right,
            right: run_score.left,
        };
        // Invalid scores, other matches' frames, and side flips (correct_sides owns
        // those) never count.
        if !valid_map_score(pair) || !self.plates_agree(idx) || pair == mirror {
            return;
        }
        let continued = self.maps[idx].contender.and_then(|mut c| {
            if second - c.last_second > CONTENDER_GAP_SECONDS {
                return None;
            }
            if pair == c.latest {
                c.reads += 1;
            } else {
                // A contradicting read starts over on this read (`?` returns None).
                let g = advance_of(c.latest, pair)
                    .filter(|g| *g <= max_gain(second - c.last_second))?;
                // The series may have gained at most one round per MIN_ROUND_SECONDS of
                // unpaused play since it started: neither a pause nor wall time may
                // ratchet a rogue jump into a legitimate advance. `paused_at_start` is
                // the pause total recorded when the series started, so a pause cleared
                // in between cannot shrink the pause it is credited against.
                let unpaused_play =
                    ((second - c.since) - (self.paused_total(second) - c.paused_at_start)).max(0);
                if c.advances + i64::from(g) > unpaused_play / MIN_ROUND_SECONDS {
                    return None;
                }
                c.latest = pair;
                c.advances += i64::from(g);
                c.reads += 1;
            }
            c.last_second = second;
            Some(c)
        });
        let c = continued.unwrap_or_else(|| {
            tracing::info!(
                second,
                map = %self.maps[idx].name,
                read = ?(pair.left, pair.right),
                "contender started"
            );
            Contender {
                first: pair,
                latest: pair,
                since: second,
                opened: self.maps[idx].candidate_since.min(second),
                last_second: second,
                reads: 1,
                advances: 0,
                paused_at_start: self.paused_total(second),
            }
        });
        self.maps[idx].contender = Some(c);
        let age = second - c.since;
        if self.pause.is_some()
            || age < CONTENDER_MIN_SECONDS
            || c.reads < CONTENDER_MIN_READS
            || (c.advances < 1 && age < CONTENDER_STABLE_SECONDS)
        {
            return;
        }
        // A series whose 0-0 showed the strip of the run's own or an earlier veto game
        // is a replay or recap of a past map's opening: neither a boundary nor a correction.
        let strip_game = self.veto_number(&self.maps[idx].match_id, &self.last_strip);
        if self.last_strip_second >= c.opened
            && (1..=self.maps[idx].map_number).contains(&strip_game)
        {
            return;
        }
        // At match point, or with a side on 11 or more (the strip gate's `late`), a
        // series that opened at 0-0 and is at most half the run's rounds is the next
        // map's opening, not a correction: the map ends as it would at the next map's
        // start, and that map starts at the series' score when the veto names it. A
        // series that opened mid-map corrects a false high score.
        if (!self.maps[idx].match_point_leader().is_empty()
            || run_score.left.max(run_score.right) >= 11)
            && c.first == Score::default()
            && 2 * (c.latest.left + c.latest.right) <= run_score.left + run_score.right
        {
            self.close_at_next_map(idx, second);
            let match_id = self.maps[idx].match_id.clone();
            let number = self.maps[idx].map_number + 1;
            let name = self.veto_name(&match_id, number);
            if !name.is_empty() && !self.series_decided(&match_id) {
                // A strip that named this map gives the next boundary a baseline.
                let strip = if self.last_strip.eq_ignore_ascii_case(&name) {
                    self.last_strip.clone()
                } else {
                    String::new()
                };
                self.begin_run(&name, number, &strip, second, c.latest, kind::MAP_START);
            }
            return;
        }
        // A series with more total rounds than the run's score is an upward jump and
        // must fit a jump allowance, so CONTENDER_STABLE cannot launder a rogue jump.
        // The allowance runs from the last second the run's score was read back or
        // accepted, so play spent behind an unreadable plate or rejected reads counts:
        // this, not a sync, is how the score recovers after such a stretch. Counting
        // net rounds bounds a mixed score like 1-2 to 10-1 too, where the rising side
        // would otherwise hide the falling one; a series with no more total rounds is
        // the recovery from a wrong run score and needs no allowance.
        let total_gain = (c.latest.left + c.latest.right) - (run_score.left + run_score.right);
        if total_gain > max_gain(self.known_play_seconds(idx, second)) {
            return;
        }
        self.resync(idx, c, second);
    }

    /// Replaces the run's score with a contender series that earned it.
    fn resync(&mut self, idx: usize, c: Contender, second: i64) {
        let previous = self.maps[idx].score_by_code();
        {
            let run = &mut self.maps[idx];
            run.score = c.latest;
            run.contender = None;
            run.candidate = None;
            run.suspect = false;
            // The baseline is replaced, so a remembered final no longer applies.
            run.final_read = None;
            run.held = None;
            run.last_read_second = second;
        }
        self.mark_hud(idx, second);
        self.mark_known(idx, second);
        self.pending_since = None;
        let source = if c.advances >= 1 {
            source::CONTENDER_SERIES
        } else {
            source::CONTENDER_STABLE
        };
        let score = self.maps[idx].score_by_code();
        tracing::warn!(
            second,
            map = %self.maps[idx].name,
            from = ?previous,
            to = ?score,
            since = c.since,
            reads = c.reads,
            advances = c.advances,
            source,
            "score resynced to a consistent series"
        );
        let event = Event {
            read_score: Some(score),
            previous_score: Some(previous),
            source: source.to_string(),
            ..self.run_event(idx, kind::RESYNC, second)
        };
        self.emit(event);
        self.end_map_if_complete(idx, second);
    }

    /// Ends the map when the score reaches 13 with a two-round lead.
    pub fn end_map_if_complete(&mut self, idx: usize, second: i64) {
        let score = self.maps[idx].score;
        if self.maps[idx].ended
            || score.left.max(score.right) < 13
            || (score.left - score.right).abs() < 2
        {
            return;
        }
        self.maps[idx].ended = true;
        let mut leader = self.maps[idx].left_code.clone();
        if score.right > score.left {
            leader = self.maps[idx].right_code.clone();
        }
        let match_id = self.maps[idx].match_id.clone();
        let (left, right) = (
            self.maps[idx].left_code.clone(),
            self.maps[idx].right_code.clone(),
        );
        let wins = self.series_wins.entry(match_id).or_insert_with(|| {
            let mut wins = ScoreMap::new();
            wins.insert(left, 0);
            wins.insert(right, 0);
            wins
        });
        *wins.entry(leader.clone()).or_insert(0) += 1;
        tracing::info!(
            second,
            map = %self.maps[idx].name,
            score = ?self.maps[idx].score_by_code(),
            series = ?wins,
            "map end"
        );
        self.maps[idx].winner = leader.clone();
        let event = Event {
            winner: leader,
            source: source::NORMAL.to_string(),
            ..self.run_event(idx, kind::MAP_END, second)
        };
        self.emit(event);
    }

    /// Settles the active map's team labels against Riot: a provisional run adopts
    /// Riot's real codes at once, while a reversal applies only after Riot has
    /// reported it for `REVERSAL_STABLE_SECONDS`.
    pub fn correct_sides(&mut self, second: i64) {
        let Some(idx) = self.active else {
            return;
        };
        if self.maps[idx].ended {
            return;
        }
        let game = self.riot_game(idx);
        if game != self.maps[idx].riot {
            self.maps[idx].riot = game.clone();
            tracing::info!(
                second,
                map = %self.maps[idx].name,
                number = self.maps[idx].map_number,
                riot = ?game,
                "riot game"
            );
        }
        let match_id = self.maps[idx].match_id.clone();
        let map_number = self.maps[idx].map_number;
        let sides = self.riot.sides(&match_id, map_number);
        let sides = if self.maps[idx].provisional {
            match sides {
                Some(sides) => sides,
                // Riot's details have arrived but list no explicit sides yet; its
                // listed teams are the first usable identity.
                None => self.riot.listed_order(&match_id),
            }
        } else {
            let reversed = match &sides {
                Some(sides) => {
                    sides.blue == self.maps[idx].right_code && sides.red == self.maps[idx].left_code
                }
                None => false,
            };
            if !reversed {
                if self.maps[idx].reversal_since.is_some() {
                    tracing::info!(
                        second,
                        map = %self.maps[idx].name,
                        number = map_number,
                        pending_since = self.maps[idx].reversal_since,
                        riot = ?self.maps[idx].riot,
                        "side reversal abandoned"
                    );
                    self.maps[idx].reversal_since = None;
                }
                return;
            }
            if self.maps[idx].reversal_since.is_none() {
                self.maps[idx].reversal_since = Some(second);
                tracing::info!(
                    second,
                    map = %self.maps[idx].name,
                    number = map_number,
                    riot = ?self.maps[idx].riot,
                    "side reversal pending"
                );
            }
            if second - self.maps[idx].reversal_since.unwrap_or(second) < REVERSAL_STABLE_SECONDS {
                return;
            }
            sides.unwrap_or_default()
        };
        let pending_since = self.maps[idx].reversal_since;
        if !relabel_run(&mut self.maps[idx], &sides) {
            return;
        }
        self.maps[idx].provisional = false;
        self.maps[idx].reversal_since = None;
        tracing::info!(
            second,
            map = %self.maps[idx].name,
            score = ?self.maps[idx].score_by_code(),
            pending_since = ?pending_since,
            "sides corrected"
        );
        let held = self.maps[idx].held.is_some();
        let riot = self.maps[idx].riot.clone();
        let mut event = Event {
            held,
            riot_state: riot.state.clone(),
            ..self.run_event(idx, kind::SIDES_CORRECTED, second)
        };
        if riot.listed {
            event.riot_blue = riot.sides.blue.clone();
            event.riot_red = riot.sides.red.clone();
        }
        self.emit(event);
        // A swap never changes the score values, so the only map it could end here is
        // one held at 13+ with a two-round lead, exactly the suspect score the hold
        // distrusts.
        if self.maps[idx].held.is_none() {
            self.end_map_if_complete(idx, second);
        }
    }

    /// Whether a team already holds the wins a series needs.
    pub fn series_decided(&self, match_id: &str) -> bool {
        let best_of = self.riot.best_of();
        if best_of <= 0 || match_id.is_empty() || match_id == UNKNOWN_MATCH {
            return false;
        }
        let needed = best_of / 2 + 1;
        self.riot
            .game_wins()
            .values()
            .any(|wins| i64::from(*wins) >= needed)
    }

    /// Riot's in-progress game number only when Riot actually lists one.
    pub fn in_progress_game(&self, match_id: &str) -> (i64, bool) {
        if match_id.is_empty() || match_id == UNKNOWN_MATCH {
            return (0, false);
        }
        let number = self.riot.map_number(0);
        match self.riot.game_state(number) {
            Some(state) if state == "inProgress" => (number, true),
            _ => (0, false),
        }
    }

    /// An event of `kind` for run `idx`, carrying the run's identity and current score.
    fn run_event(&self, idx: usize, kind: &str, second: i64) -> Event {
        Event {
            kind: kind.to_string(),
            second,
            map: self.maps[idx].name.clone(),
            match_id: self.maps[idx].match_id.clone(),
            map_number: self.maps[idx].map_number,
            score: Some(self.maps[idx].score_by_code()),
            ..Default::default()
        }
    }

    /// Riot's current state and explicit sides for a run's game.
    pub fn riot_game(&self, idx: usize) -> RiotGame {
        let state = self
            .riot
            .game_state(self.maps[idx].map_number)
            .unwrap_or_default();
        let (sides, listed) = match self
            .riot
            .sides(&self.maps[idx].match_id, self.maps[idx].map_number)
        {
            Some(sides) => (sides, true),
            None => (Sides::default(), false),
        };
        RiotGame {
            state,
            sides,
            listed,
        }
    }

    /// The team Riot's series tally shows as having won the run's map, or `""`.
    pub fn riot_winner(&self, idx: usize) -> String {
        let riot_wins = self.riot.game_wins();
        let mut gained = Vec::new();
        for code in [
            self.maps[idx].left_code.clone(),
            self.maps[idx].right_code.clone(),
        ] {
            let wins = riot_wins.get(&code).copied().unwrap_or(0);
            let baseline = self.maps[idx]
                .wins_baseline
                .get(&code)
                .copied()
                .unwrap_or(0);
            if wins > baseline {
                gained.push(code);
            }
        }
        if gained.len() == 1 {
            return gained.remove(0);
        }
        String::new()
    }

    /// Riot credits a game minutes after the next map can start, so a run's series
    /// baseline can miss the previous game's win, and the run's own winner then reads
    /// as two gainers. Once Riot completes the game before the active run's while the
    /// run's own is not, every credited win is an earlier game's: the baseline takes
    /// them, and a single new win fills in the previous map's winner when it closed
    /// without one or corrects a guessed one. A closed run still active settles
    /// against its own baseline instead.
    fn settle_baseline(&mut self, second: i64) {
        let Some(idx) = self.active else {
            return;
        };
        let (number, match_id) = (self.maps[idx].map_number, self.maps[idx].match_id.clone());
        // A run closed with no winner or a guessed one stays active when no next map
        // opened: it settles against its own baseline once Riot completes its game.
        if self.maps[idx].ended {
            if (self.maps[idx].guessed || self.maps[idx].winner.is_empty())
                && match_id == self.riot.match_id()
                && self.riot.game_state(number).as_deref() == Some("completed")
            {
                let winner = self.riot_winner(idx);
                if !winner.is_empty() && winner != self.maps[idx].winner {
                    self.maps[idx].guessed = false;
                    self.finish_hidden_map(idx, &winner, source::RIOT, second);
                }
            }
            return;
        }
        if self.maps[idx].baseline_settled
            || match_id != self.riot.match_id()
            || (number > 1 && self.riot.game_state(number - 1).as_deref() != Some("completed"))
        {
            return;
        }
        self.maps[idx].baseline_settled = true;
        if number <= 1 || self.riot.game_state(number).as_deref() == Some("completed") {
            return;
        }
        let wins = self.riot.game_wins();
        let baseline = std::mem::replace(&mut self.maps[idx].wins_baseline, wins.clone());
        let gained: Vec<(&String, i32)> = wins
            .iter()
            .map(|(code, won)| (code, won - baseline.get(code).copied().unwrap_or(0)))
            .filter(|(_, gain)| *gain != 0)
            .collect();
        let [(winner, 1)] = gained[..] else {
            return;
        };
        // A guessed winner Riot contradicts is corrected; one it confirms stays as is.
        let Some(prev) = self.maps.iter().rposition(|r| {
            r.match_id == match_id
                && r.map_number == number - 1
                && r.ended
                && (r.winner.is_empty() || (r.guessed && r.winner != *winner))
                && (r.left_code == *winner || r.right_code == *winner)
        }) else {
            return;
        };
        self.maps[prev].guessed = false;
        self.finish_hidden_map(prev, winner, source::RIOT, second);
    }

    /// Closes a run at the next map's start: for the team Riot credited when Riot lists
    /// its game completed, else for the team at match point as a guess, else with no
    /// winner. Riot's later credit settles a guess or a missing winner.
    fn close_at_next_map(&mut self, idx: usize, second: i64) {
        let mut winner = String::new();
        if self.riot.game_state(self.maps[idx].map_number).as_deref() == Some("completed") {
            winner = self.riot_winner(idx);
        }
        if winner.is_empty() {
            winner = self.maps[idx].match_point_leader();
            self.maps[idx].guessed = !winner.is_empty();
        }
        self.finish_hidden_map(idx, &winner, source::NEXT_MAP, second);
    }

    /// Closes a map whose final score the broadcast never showed. An empty `winner`
    /// closes it unexact with no winner and no series credit. On a map already closed
    /// for a guessed winner, the new winner replaces it in the series tally.
    pub fn finish_hidden_map(&mut self, idx: usize, winner: &str, source_name: &str, second: i64) {
        if let Some(held) = self.maps[idx].held_pair() {
            self.maps[idx].score = held;
            self.maps[idx].held = None;
        }
        let exact = !self.maps[idx].guessed
            && !winner.is_empty()
            && winner == self.maps[idx].match_point_leader();
        if exact && winner == self.maps[idx].left_code {
            self.maps[idx].score.left += 1;
        } else if exact {
            self.maps[idx].score.right += 1;
        }
        self.maps[idx].ended = true;
        let replaced = std::mem::replace(&mut self.maps[idx].winner, winner.to_string());
        let match_id = self.maps[idx].match_id.clone();
        let wins = self.series_wins.entry(match_id).or_default();
        if let Some(won) = wins.get_mut(&replaced) {
            *won -= 1;
        }
        if !winner.is_empty() {
            *wins.entry(winner.to_string()).or_insert(0) += 1;
        }
        tracing::info!(
            second,
            map = %self.maps[idx].name,
            confirmed_by = source_name,
            winner = %winner,
            score = ?self.maps[idx].score_by_code(),
            exact,
            series = ?wins,
            "hidden map end"
        );
        let event = Event {
            winner: winner.to_string(),
            source: source_name.to_string(),
            exact: Some(exact),
            ..self.run_event(idx, kind::MAP_END, second)
        };
        self.emit(event);
    }

    /// Arms, clears or confirms a map end when the score bar vanishes at match point.
    /// `hud_visible` says whether this frame still shows score-bar digits; the
    /// remembered single winning read is adopted only after the HUD is really gone.
    /// `read` is this frame's score pair, when its read produced one; without one a
    /// `gameplay` scene is not the HUD returning.
    pub fn check_pending_end(
        &mut self,
        second: i64,
        frame_scene: scene::Scene,
        hud_visible: bool,
        read: Option<Score>,
    ) {
        let Some(idx) = self.active else {
            self.pending_since = None;
            return;
        };
        if self.maps[idx].ended {
            self.pending_since = None;
            return;
        }
        // A read below the run's score on either side is not this map's HUD returning
        // but the next map's opening, a replay or another match: the contender owns it,
        // and the end stays armed with its Riot check running.
        let score = self.maps[idx].score;
        let lower = read.is_some_and(|pair| pair.left < score.left || pair.right < score.right);
        if frame_scene == scene::Scene::Gameplay && !lower {
            // Intermission footage — the crowd, the desk, the arena screens, a
            // highlight clip — trips the timer-box colour check and the plate
            // brightness proxy without a real HUD, so the scene label alone cannot
            // prove gameplay: only a read pair can. An uncorroborated gameplay frame
            // is neither the HUD returning nor a frame the end may act on: it must
            // not clear an armed end or defer arming one for its whole cluster, and
            // the gaps it creates could swallow Riot's completion.
            if read.is_none() {
                return;
            }
            self.last_gameplay_full = Some(second);
            if self.pending_since.is_some() {
                tracing::info!(second, map = %self.maps[idx].name, "pending map end cleared");
                let event = Event {
                    held: self.maps[idx].held.is_some(),
                    ..self.run_event(idx, kind::PENDING_CLEARED, second)
                };
                self.emit(event);
                self.pending_since = None;
            }
            return;
        }
        let leader = self.maps[idx].match_point_leader();
        if leader.is_empty() || self.last_gameplay_full.is_none() {
            return;
        }
        // The frame after a map-winning read had no score bar, so no second read can
        // confirm it. Adopt it once, through the normal end path, but only after the
        // HUD is actually gone: a replay still showing the score bar could yet
        // contradict it, and a suspect read must keep its hold rather than be
        // cleaned up by the disappearance.
        if !hud_visible
            && self.pending_since.is_none()
            && let Some((at, pair, suspect)) = self.maps[idx].final_read.take()
            && second - at <= PENDING_END_SECONDS
        {
            self.record_score(idx, pair, second, suspect);
            return;
        }
        if self.pending_since.is_none()
            && second - self.last_gameplay_full.unwrap_or(second) >= PENDING_END_SECONDS
        {
            self.pending_since = Some(second);
            tracing::info!(second, map = %self.maps[idx].name, leader = %leader, "map end pending");
            let event = Event {
                winner: leader,
                held: self.maps[idx].held.is_some(),
                ..self.run_event(idx, kind::PENDING_END, second)
            };
            self.emit(event);
        }
        if self.pending_since.is_none() || self.maps[idx].match_id != self.riot.match_id() {
            return;
        }
        let state = self.riot.game_state(self.maps[idx].map_number);
        if state.as_deref() != Some("completed") {
            return;
        }
        let winner = self.riot_winner(idx);
        if !winner.is_empty() {
            self.finish_hidden_map(idx, &winner, source::RIOT, second);
            self.pending_since = None;
        }
    }

    /// Writes the map score when it changes, and every heartbeat otherwise.
    pub fn publish_run(&mut self, idx: usize, second: i64) {
        // A pause keeps the map healthy: the overlay says play has stopped, not that
        // the tracker lost the broadcast. A contender in doubt is the exception: the
        // run's own score is the thing that might be wrong.
        let doubted = self.maps[idx].contender.is_some_and(|c| {
            second - c.since >= CONTENDER_DOUBT_SECONDS && c.reads >= CONTENDER_DOUBT_READS
        });
        let reading = !self.maps[idx].ended
            && self.maps[idx].held.is_none()
            && !doubted
            && (second - self.maps[idx].last_read_second <= UNREAD_SECONDS || self.pause.is_some());
        let mut current = Published {
            map_number: self.maps[idx].map_number,
            score: self.maps[idx].score_by_code(),
            status: if reading { Status::Ok } else { Status::Error },
            blue: self.maps[idx].left_code.clone(),
            pause: String::new(),
            pause_reason: String::new(),
        };
        if let Some(pause) = &self.pause {
            current.pause = pause.kind.clone();
            current.pause_reason = pause.reason.clone();
        }
        if let Some((_, held)) = &self.maps[idx].held {
            current.score = held.clone();
        }
        if self.maps[idx].reversal_since.is_some() {
            // Riot disagrees with the run's sides but has not held the reversal long
            // enough to swap, so either attribution may be wrong.
            let Some(last) = &self.published else {
                return;
            };
            if last.map_number != self.maps[idx].map_number {
                return;
            }
            current.score = last.score.clone();
            current.blue = last.blue.clone();
            current.status = Status::Error;
        }
        let last = self.published.clone();
        if last.as_ref() == Some(&current)
            && (self.now)().duration_since(self.published_at)
                < Duration::from_secs(HEARTBEAT_SECONDS)
        {
            return;
        }
        let teams = self.riot.teams();
        let pause = self.pause.clone();
        let result = match self.publish.as_ref() {
            Some(publish) => publish.publish(
                current.map_number,
                &current.score,
                &teams,
                &current.blue,
                current.status,
                pause.as_ref(),
            ),
            None => Ok(()),
        };
        match result {
            Err(err) => {
                tracing::warn!(error = %err, "push publish failed; retrying with the next heartbeat")
            }
            Ok(()) => {
                if last.as_ref() != Some(&current) {
                    tracing::info!(
                        score = ?current.score,
                        status = current.status.as_str(),
                        pause = %current.pause,
                        "published"
                    );
                }
            }
        }
        self.published = Some(current);
        self.published_at = (self.now)();
    }

    /// Delivers one event to the observer, applying the shadow rules.
    pub fn emit(&mut self, mut event: Event) {
        if self.on_event.is_none() {
            return;
        }
        if self.shadow {
            event = self.with_evidence(event);
        }
        if let Some(callback) = self.on_event.as_mut() {
            callback(&event);
            if self.shadow
                && !event.held
                && matches!(
                    event.kind.as_str(),
                    kind::MAP_START
                        | kind::MAP_JOIN
                        | kind::ROUND
                        | kind::SYNC
                        | kind::SCORE_CORROBORATED
                        | kind::SCORE_ROLLBACK
                        | kind::RESYNC
                )
            {
                let mut accepted = event.clone();
                accepted.source = event.kind.clone();
                accepted.kind = kind::VIDEO_ACCEPTED.to_string();
                callback(&accepted);
            }
        }
    }

    /// Delivers one event without the shadow wrapper.
    pub fn deliver(&mut self, event: Event) {
        if let Some(callback) = self.on_event.as_mut() {
            callback(&event);
        }
    }
}

/// Moves a run's codes onto the given sides, carrying any held score with them, and
/// reports whether either label changed.
pub fn relabel_run(run: &mut MapRun, sides: &Sides) -> bool {
    if sides.blue == run.left_code && sides.red == run.right_code {
        return false;
    }
    if let Some((_, held)) = &mut run.held {
        *held = ScoreMap::from([
            (
                sides.blue.clone(),
                held.get(&run.left_code).copied().unwrap_or(0),
            ),
            (
                sides.red.clone(),
                held.get(&run.right_code).copied().unwrap_or(0),
            ),
        ]);
    }
    run.left_code = sides.blue.clone();
    run.right_code = sides.red.clone();
    true
}

/// The most rounds a confirmed change may add after `play_seconds` of unpaused hidden play.
pub fn max_gain(play_seconds: i64) -> i32 {
    1 + (play_seconds.max(0) / MIN_ROUND_SECONDS) as i32
}

/// Whether a score can occur on a Valorant map (regulation to 13, overtime win by 2).
pub fn valid_map_score(s: Score) -> bool {
    let (hi, lo) = (s.left.max(s.right), s.left.min(s.right));
    lo >= 0 && (hi <= 13 || (lo >= 12 && hi - lo <= 2))
}

/// Rounds gained from `from` to `to`, or `None` when either side went down.
fn advance_of(from: Score, to: Score) -> Option<i32> {
    (to.left >= from.left && to.right >= from.right)
        .then(|| to.left - from.left + to.right - from.right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_point_leader_reads_held_score() {
        let mut run = MapRun {
            left_code: "A".into(),
            right_code: "B".into(),
            score: Score {
                left: 13,
                right: 11,
            },
            ..Default::default()
        };
        let mut held = ScoreMap::new();
        held.insert("A".into(), 12);
        held.insert("B".into(), 11);
        run.held = Some((0, held));
        assert_eq!(run.match_point_leader(), "A");
    }

    #[test]
    fn match_point_leader_needs_a_one_round_edge() {
        let leader = |left, right| {
            MapRun {
                left_code: "A".into(),
                right_code: "B".into(),
                score: Score { left, right },
                ..Default::default()
            }
            .match_point_leader()
        };
        assert_eq!(leader(12, 11), "A");
        assert_eq!(leader(11, 12), "B");
        assert_eq!(leader(12, 5), "A");
        assert_eq!(leader(10, 8), "");
    }
}
