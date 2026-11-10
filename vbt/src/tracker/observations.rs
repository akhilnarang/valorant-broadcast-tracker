//! Frame observations and the in-process scoreboard hook.

use vbt_vision::frame::Frame;
use vbt_vision::learned;
use vbt_vision::scene::Scene;

use super::{
    Event, RECOGNIZER_EVERY_SECOND, SCOREBOARD_EVERY_SECONDS, Score, Tracker, advance_of, kind,
};

impl Tracker {
    /// The two expected competitor codes: the replay override, then the map-order
    /// source, then Riot's teams.
    pub fn expected_codes(&self, match_id: &str) -> Vec<String> {
        if self.expected_teams.len() == 2 {
            return self.expected_teams.clone();
        }
        if let Some(order) = &self.map_order
            && let Some(codes) = order.team_codes(match_id)
            && codes.len() == 2
        {
            return codes;
        }
        self.riot
            .teams()
            .into_iter()
            .map(|team| team.code)
            .collect()
    }

    /// Reads the tag plates, settles the visual mapping, and emits the shadow
    /// frame observation.
    pub(super) fn observe_frame(
        &mut self,
        second: i64,
        frame: &Frame,
        frame_scene: Scene,
        score: Score,
        have_score: bool,
    ) {
        let plate = if frame_scene == Scene::Gameplay {
            self.read_plates(second, frame, score, have_score)
        } else {
            learned::Read::default()
        };
        self.plate = plate.clone();
        if plate.ok {
            let candidate = super::Sides {
                blue: plate.left.clone(),
                red: plate.right.clone(),
            };
            if candidate == self.plate_candidate {
                self.plate_count += 1;
            } else {
                self.plate_candidate = candidate.clone();
                self.plate_count = 1;
            }
            if self.plate_count >= 2 {
                if candidate != self.visual {
                    let mut number = self.riot.map_number(0);
                    if let Some(run) = self.active_run()
                        && !run.ended
                        && run.match_id == self.last_match_id
                    {
                        number = run.map_number;
                    }
                    let riot_sides = self.riot.sides(&self.last_match_id, number);
                    tracing::info!(
                        second,
                        left = %plate.left,
                        right = %plate.right,
                        riot = ?riot_sides,
                        "visual sides"
                    );
                }
                self.visual = candidate.clone();
                if let Some(idx) = self.active
                    && !self.maps[idx].ended
                    && self.maps[idx].match_id == self.last_match_id
                {
                    self.maps[idx].visual = candidate;
                }
            }
        } else {
            self.plate_count = 0;
        }
        if !self.shadow {
            return;
        }
        let mut event = Event {
            kind: kind::FRAME_OBSERVATION.to_string(),
            second,
            match_id: self.match_id_or_unknown(),
            scene: frame_scene.as_str().to_string(),
            visual_left: plate.left.clone(),
            visual_right: plate.right.clone(),
            plate_ok: plate.ok,
            left_confidence: plate.left_confidence,
            right_confidence: plate.right_confidence,
            ..Default::default()
        };
        let active_matches = self
            .active_run()
            .is_some_and(|run| run.match_id == event.match_id && !run.ended);
        if active_matches {
            let idx = self.active.unwrap();
            event.map = self.maps[idx].name.clone();
            event.map_number = self.maps[idx].map_number;
        } else {
            event.map_number = self.next_map_number("", &event.match_id.clone());
            event.map = self.veto_name(&event.match_id.clone(), event.map_number);
        }
        if let Some(sides) = self.riot.sides(&event.match_id, event.map_number) {
            event.riot_blue = sides.blue;
            event.riot_red = sides.red;
        }
        event.riot_state = self.riot.game_state(event.map_number).unwrap_or_default();
        if have_score {
            event.left_score = Some(score.left);
            event.right_score = Some(score.right);
            if plate.ok {
                let mut video_score = super::ScoreMap::new();
                video_score.insert(plate.left, score.left);
                video_score.insert(plate.right, score.right);
                event.video_score = Some(video_score);
            }
        }
        // A frame's missing evidence must stay missing, not inherit the run's plates.
        self.deliver(event);
    }

    /// Reads both plates from the learned crops, and asks the OCR when they leave a plate
    /// unread and the frame needs tags: while an expected code has no crop at this height,
    /// while the visual sides are unsettled, on the scheduled reads a join counts, and on a
    /// read that jumps the run by more than one round, which `plates_agree` gates. When the
    /// OCR reads both tags it teaches the crops each plate they missed, so a code whose
    /// crops stopped reading (a changed HUD) relearns.
    fn read_plates(
        &mut self,
        second: i64,
        frame: &Frame,
        score: Score,
        have_score: bool,
    ) -> learned::Read {
        let Some(plates) = &self.plates else {
            return learned::Read::default();
        };
        let codes: Vec<String> = self
            .expected_codes(&self.last_match_id)
            .iter()
            .map(|code| code.trim().to_uppercase())
            .collect();
        let mut plate = plates.read(frame, &codes);
        let (Some(tags), [a, b]) = (self.tag_reader.as_ref(), codes.as_slice()) else {
            return plate;
        };
        let unlearned = codes.iter().any(|code| plates.count(code, frame.h) == 0);
        let jump = self
            .active_run()
            .filter(|run| !run.ended && run.match_id == self.last_match_id)
            .is_some_and(|run| {
                have_score && score != run.score && advance_of(run.score, score) != Some(1)
            });
        let needed = unlearned
            || self.visual.blue.is_empty()
            || second % RECOGNIZER_EVERY_SECOND == 0
            || jump;
        if plate.ok || !needed {
            return plate;
        }
        // Both plates are read so the frame can teach the crops.
        let ocr = tags.read(frame, [a, b]);
        let missed = [plate.left.is_empty(), plate.right.is_empty()];
        for (read, ocr) in [(&mut plate.left, &ocr[0]), (&mut plate.right, &ocr[1])] {
            if read.is_empty() {
                read.clone_from(ocr);
            } else if !ocr.is_empty() && read != ocr {
                // The crops and the OCR disagree; neither is trusted.
                read.clear();
            }
        }
        plate.ok = !plate.left.is_empty() && !plate.right.is_empty() && plate.left != plate.right;
        if plate.ok
            && ocr.iter().all(|code| !code.is_empty())
            && let Some(plates) = self.plates.as_mut()
        {
            let teach = [0, 1].map(|side| if missed[side] { ocr[side].as_str() } else { "" });
            match plates.learn(frame, teach) {
                Ok(0) => {}
                Ok(n) => tracing::info!(second, codes = ?teach, n, "learned team tag crops"),
                Err(error) => tracing::warn!(
                    second,
                    error = %error,
                    "saving team tag crops failed; keeping learned crops in memory only"
                ),
            }
        }
        plate
    }

    /// Enriches an event with the run's visual evidence, for shadow mode.
    pub fn with_evidence(&self, mut event: Event) -> Event {
        for (idx, run) in self.maps.iter().enumerate() {
            if run.match_id != event.match_id || run.map_number != event.map_number {
                continue;
            }
            event.visual_left = run.visual.blue.clone();
            event.visual_right = run.visual.red.clone();
            let left = run.score.left;
            let right = run.score.right;
            event.left_score = Some(left);
            event.right_score = Some(right);
            if !run.visual.blue.is_empty() && !run.visual.red.is_empty() {
                let mut video_score = super::ScoreMap::new();
                video_score.insert(run.visual.blue.clone(), left);
                video_score.insert(run.visual.red.clone(), right);
                event.video_score = Some(video_score);
            }
            let game = self.riot_game(idx);
            event.riot_state = game.state;
            if game.listed {
                event.riot_blue = game.sides.blue;
                event.riot_red = game.sides.red;
            }
            break;
        }
        event
    }

    /// Periodically hands a gameplay frame to the experimental table reader. The
    /// read is logged and emitted for audit only; it never changes a run's score.
    pub(super) fn observe_scoreboard(&mut self, second: i64, frame: &Frame, frame_scene: Scene) {
        if self.scoreboard.is_none() || frame_scene != Scene::Gameplay {
            return;
        }
        if second - self.last_scoreboard_second < SCOREBOARD_EVERY_SECONDS {
            return;
        }
        self.last_scoreboard_second = second;
        let table = match self.scoreboard.as_mut() {
            Some(reader) => match reader.analyze(frame) {
                Ok(table) => table,
                Err(error) => {
                    tracing::warn!(second, error = %error, "scoreboard analysis failed");
                    return;
                }
            },
            None => return,
        };
        let Some(table) = table else {
            return;
        };
        if table.layout != "broadcast_table" && table.players.is_empty() {
            return;
        }
        let round = table.round_displayed.unwrap_or(0);
        tracing::info!(
            second,
            round,
            players_count = table.players.len(),
            teams = ?table.teams.iter().map(|t| format!("{}:{:?}", t.name, t.score)).collect::<Vec<_>>(),
            "scoreboard detected"
        );
        let mut event = Event {
            kind: kind::SCOREBOARD_OBSERVATION.to_string(),
            second,
            match_id: self.last_match_id.clone(),
            scoreboard: Some(table),
            ..Default::default()
        };
        if let Some(idx) = self.active
            && !self.maps[idx].ended
            && self.maps[idx].match_id == self.last_match_id
        {
            event.map = self.maps[idx].name.clone();
            event.map_number = self.maps[idx].map_number;
        }
        self.emit(event);
    }
}
