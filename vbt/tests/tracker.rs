//! Tracker state machine tests. They are the executable spec of the state machine.

mod common;

use std::collections::BTreeMap;

use common::*;
use vbt::tracker::{MapRun, Pause, Score, ScoreMap, Sides, Status, TagReader, Team};
use vbt_vision::frame::Frame;
use vbt_vision::learned::{Crops, KEEP};
use vbt_vision::recognizer::PauseRead;
use vbt_vision::scene::Scene;

const REVERSAL_STABLE_SECONDS: i64 = 120;

fn scores(run: &MapRun) -> (i32, i32) {
    (run.score.left, run.score.right)
}

// Templates-only reads must move the score one round on two identical reads, and a
// frame with no score bar must not change it.
#[test]
fn test_two_reads_move_the_score() {
    let cases = [
        (
            "summit-12-4.jpg",
            Score { left: 11, right: 4 },
            Score { left: 12, right: 4 },
        ),
        (
            "lotus-half-time.jpg",
            Score { left: 8, right: 3 },
            Score { left: 8, right: 3 },
        ),
    ];
    for (name, before, after) in cases {
        let (mut tracker, _) = test_tracker();
        set_active(
            &mut tracker,
            MapRun {
                name: "Summit".into(),
                match_id: "match".into(),
                map_number: 2,
                left_code: "LOUD".into(),
                right_code: "EDG".into(),
                score: before,
                ..Default::default()
            },
        );
        let frame = load_frame(name);
        // Seconds 1 and 2 are off the forced recognizer read, so the template path decides.
        tracker.step(1, &frame);
        tracker.step(2, &frame);
        assert_eq!(
            scores(active(&tracker)),
            (after.left, after.right),
            "{name}"
        );
    }
}

// The deciding 13-5 is on screen for about a second, so the panel reads it once and
// the next frame is the FRAG OUT break card with no HUD. The unconfirmed but
// map-winning read must still end the map through the normal path, not stall at
// 12-5 until Riot's late confirmation.
#[test]
fn test_single_map_winning_read_ends_the_map() {
    let (mut tracker, riot) = test_tracker();
    {
        let mut state = riot.get();
        state.teams = vec![
            Team {
                id: "1".into(),
                code: "VIT".into(),
                name: "Team Vitality".into(),
            },
            Team {
                id: "2".into(),
                code: "LOUD".into(),
                name: "LOUD".into(),
            },
        ];
        let sides = Sides {
            blue: "VIT".into(),
            red: "LOUD".into(),
        };
        state.sides.insert(2, sides.clone());
        state.order = sides;
    }
    set_active(
        &mut tracker,
        MapRun {
            name: "Summit".into(),
            match_id: "match".into(),
            map_number: 2,
            left_code: "VIT".into(),
            right_code: "LOUD".into(),
            score: Score { left: 5, right: 12 },
            ..Default::default()
        },
    );
    let events = capture(&mut tracker);
    // Second 1 is the only 13-5 frame; the broadcast cuts away before the next read.
    tracker.step(1, &load_frame("summit-13-5.jpg"));
    let break_card = load_frame("frag-out-084033.png");
    tracker.step(2, &break_card);
    tracker.step(22, &break_card); // a regression would arm a pending end here

    let events = events.lock().unwrap();
    let won = events.iter().find(|event| event.kind == "round");
    assert_eq!(
        won.and_then(|event| event.score.clone()),
        Some(ScoreMap::from([
            ("VIT".to_string(), 5),
            ("LOUD".to_string(), 13),
        ])),
        "want the deciding round recorded: {events:?}"
    );
    let ends: Vec<(i64, String, Option<ScoreMap>)> = events
        .iter()
        .filter(|event| event.kind == "map_end")
        .map(|event| (event.second, event.source.clone(), event.score.clone()))
        .collect();
    assert_eq!(
        ends,
        vec![(
            2,
            "normal".to_string(),
            Some(ScoreMap::from([
                ("VIT".to_string(), 5),
                ("LOUD".to_string(), 13),
            ])),
        )],
        "want exactly one normal map_end at the first HUD-less frame: {events:?}"
    );
    assert!(
        !events.iter().any(|event| event.kind == "pending_end"),
        "want no pending_end churn: {events:?}"
    );
    drop(events);

    // A readable read that contradicts the remembered final — here a return to the
    // run's own 5-12 — must invalidate it, so the later cutaway cannot end the map
    // on the contradicted 5-13.
    let (mut tracker, _) = test_tracker();
    set_active(
        &mut tracker,
        MapRun {
            name: "Summit".into(),
            match_id: "match".into(),
            map_number: 2,
            left_code: "VIT".into(),
            right_code: "LOUD".into(),
            score: Score { left: 5, right: 12 },
            ..Default::default()
        },
    );
    let recognizer = FixedPairRecognizer::new(Score { left: 5, right: 13 });
    tracker.recognizer = Some(Box::new(recognizer.clone()));
    let events = capture(&mut tracker);
    let winning = load_frame("summit-13-5.jpg");
    let break_card = load_frame("frag-out-084033.png");
    // Second 1 reads the deciding 5-13 and remembers it.
    tracker.step(1, &winning);
    // Second 2 reads the run's own 5-12 again, contradicting the remembered final.
    recognizer.get().pair = Score { left: 5, right: 12 };
    tracker.step(2, &winning);
    // The broadcast cuts away before any further read; the contradicted read must
    // not be adopted.
    tracker.step(3, &break_card);
    tracker.step(22, &break_card);

    let events = events.lock().unwrap();
    assert!(
        !events.iter().any(|event| event.kind == "round"),
        "contradicted final read recorded a round: {events:?}"
    );
    assert!(
        !events.iter().any(|event| event.kind == "map_end"),
        "contradicted final read ended the map: {events:?}"
    );
}

// After the deciding read the broadcast cuts away: the break card, the
// celebrations, the desk, the arena screens and a highlight clip follow. The
// timer-box check calls four of those frames gameplay, yet none of them reads a
// score pair. On VIT vs LOUD map 2 five such clusters cleared and re-armed the
// pending end, and could swallow Riot's completion inside a cleared gap, so a
// gameplay frame without a readable pair must neither defer the end's arming nor
// clear one that is armed.
#[test]
fn test_intermission_footage_does_not_defer_a_pending_end() {
    let (mut tracker, _) = test_tracker();
    tracker.recognizer = Some(Box::new(vision_recognizer()));
    let index = set_active(
        &mut tracker,
        MapRun {
            name: "Summit".into(),
            match_id: "match".into(),
            map_number: 2,
            left_code: "VIT".into(),
            right_code: "LOUD".into(),
            score: Score { left: 5, right: 12 },
            ..Default::default()
        },
    );
    let events = capture(&mut tracker);

    // The last readable 5-12 frame: a real HUD read corroborates gameplay.
    tracker.step(1, &load_frame("summit-5-12.png"));

    let intermission = [
        "falsegame-084008.png",
        "falsegame-084018.png",
        "falsegame-084305.png",
        "falsegame-084425.png",
        "falsegame-084605.png",
    ];
    for (i, name) in intermission.iter().enumerate() {
        tracker.step(2 + i as i64, &load_frame(name));
    }
    tracker.step(7, &load_frame("frag-out-084033.png"));
    assert!(
        tracker.pending_since.is_none(),
        "an end armed inside the break"
    );

    // 20 s after the last real HUD frame the end arms on schedule.
    tracker.step(21, &load_frame("frag-out-084033.png"));
    assert_eq!(
        tracker.pending_since,
        Some(21),
        "the break deferred the pending end"
    );

    // More intermission footage must not clear the armed end.
    for (i, name) in intermission.iter().enumerate() {
        tracker.step(22 + i as i64, &load_frame(name));
    }
    assert_eq!(
        tracker.pending_since,
        Some(21),
        "intermission footage cleared the pending end"
    );
    assert!(!tracker.maps[index].ended, "the break ended the map");

    let events = events.lock().unwrap();
    let pending: Vec<(i64, String)> = events
        .iter()
        .filter(|event| event.kind == "pending_end")
        .map(|event| (event.second, event.winner.clone()))
        .collect();
    assert_eq!(
        pending,
        vec![(21, "LOUD".to_string())],
        "want exactly one pending_end for LOUD: {events:?}"
    );
    assert!(
        !events.iter().any(|event| event.kind == "pending_cleared"),
        "want no pending_cleared: {events:?}"
    );
}

// The next map's opening, a replay or another match's HUD can show a pair below the
// run's score while a hidden end is armed (KRU-TLV 156: Haven's 0-0 cleared Summit's
// armed end at 12-9). Such a read is not this map's HUD returning: it must leave the end
// armed, while a read that continues the run's score clears it.
#[test]
fn test_lower_read_keeps_a_pending_end_armed_and_a_higher_read_clears_it() {
    let (mut tracker, _) = test_tracker();
    tracker.recognizer = Some(Box::new(vision_recognizer()));
    set_active(
        &mut tracker,
        MapRun {
            name: "Summit".into(),
            match_id: "match".into(),
            map_number: 2,
            left_code: "VIT".into(),
            right_code: "LOUD".into(),
            score: Score { left: 4, right: 12 },
            ..Default::default()
        },
    );
    tracker.last_gameplay_full = Some(1);
    let events = capture(&mut tracker);
    tracker.step(21, &load_frame("frag-out-084033.png"));
    assert_eq!(tracker.pending_since, Some(21), "the end never armed");

    tracker.recognizer = Some(Box::new(StubRecognizer::new("")));
    for second in [25, 30] {
        tracker.step(second, &zero_zero_frame());
    }
    assert_eq!(
        tracker.pending_since,
        Some(21),
        "a 0-0 read cleared the armed end"
    );

    tracker.recognizer = Some(Box::new(vision_recognizer()));
    tracker.step(31, &load_frame("summit-5-12.png"));
    assert!(
        tracker.pending_since.is_none(),
        "a 5-12 continuation kept the end armed"
    );
    let cleared: Vec<i64> = events
        .lock()
        .unwrap()
        .iter()
        .filter(|event| event.kind == "pending_cleared")
        .map(|event| event.second)
        .collect();
    assert_eq!(cleared, vec![31]);
}

// A replay has no live Riot match, so no reversal can ever arrive; a round must
// still be emitted the moment it is recorded.
#[test]
fn test_replay_emits_rounds_without_a_live_riot() {
    let (mut tracker, riot) = test_tracker();
    {
        let mut state = riot.get();
        state.live = false;
        state.event = false;
    }
    tracker.map_order = Some(Box::new(FakeMapOrder::new(&[("match", &["Split"])])));
    tracker.recognizer = Some(Box::new(StubRecognizer::new("")));
    let events = capture(&mut tracker);

    tracker.step(5, &zero_zero_frame());
    tracker.step(10, &zero_zero_frame()); // the veto names game one and the run starts
    let run = tracker.active_run().expect("active run");
    assert_eq!(run.name, "Split");

    let recognizer = FixedPairRecognizer::new(Score { left: 1, right: 0 });
    tracker.recognizer = Some(Box::new(recognizer.clone()));
    let frame = occluded_plate_frame();
    tracker.step(11, &frame);
    tracker.step(12, &frame); // the round is confirmed on the replay's second frame

    let found = events
        .lock()
        .unwrap()
        .iter()
        .any(|event| event.kind == "round" && event.second == 12 && event.winner == "LOUD");
    assert!(found, "want the round emitted at once");
}

/// A tag OCR that reads TL left and G2 right on its first call and nothing after.
struct OnceTags(std::sync::Mutex<bool>);

impl TagReader for OnceTags {
    fn read(&self, _: &Frame, _: [&str; 2]) -> [String; 2] {
        if std::mem::replace(&mut *self.0.lock().unwrap(), true) {
            return Default::default();
        }
        ["TL".to_string(), "G2".to_string()]
    }
}

// Shadow must retain video identity when Riot starts reversed, emit accepted/final
// scores under that identity, and never call even an enabled publishing boundary. The
// store starts with a full set of stale TL and G2 crops (cut from other plates) that read
// nothing, so the identity comes from one OCR read of the map start: it must replace a
// stale crop of each code and carry the later frames.
#[test]
fn test_shadow_audits_abyss_without_publishing() {
    let (mut engine, riot) = test_tracker();
    {
        let mut state = riot.get();
        state.live = true;
        state.event = true;
        state.game = 2;
        state.teams = vec![
            Team {
                id: String::new(),
                code: "G2".into(),
                name: String::new(),
            },
            Team {
                id: String::new(),
                code: "TL".into(),
                name: String::new(),
            },
        ];
        state.sides = BTreeMap::from([(
            2,
            Sides {
                blue: "G2".into(),
                red: "TL".into(),
            },
        )]);
        state.states = BTreeMap::from([(2, "inProgress".to_string())]);
    }
    engine.map_order = Some(Box::new(FakeMapOrder::new(&[(
        "match",
        &["Split", "Abyss"],
    )])));
    engine.shadow = true;
    let mut stale = Crops::open(None).unwrap();
    let other = load_frame("ge-vit-abyss-4322.jpg");
    let other = paste_plates(
        &load_teamplate_frame("abyss-start.png"),
        Some(&other.crop(PLATE_LEFT)),
        Some(&other.crop(PLATE_RIGHT)),
    );
    for _ in 0..KEEP {
        stale.learn(&other, ["TL", "G2"]).unwrap();
    }
    engine.plates = Some(stale);
    engine.tag_reader = Some(Box::new(OnceTags(Default::default())));
    let publisher = FakePublisher::new(true);
    engine.publish = Some(Box::new(publisher.clone()));
    let events = capture(&mut engine);

    let start = load_teamplate_frame("abyss-start.png");
    let round5 = load_teamplate_frame("abyss-round5.png");
    engine.step(0, &start);
    engine.step(5, &start);
    // The 0-0 -> 3-1 jump adds 4 rounds, so it needs 4 rounds' worth of play (135 s).
    engine.step(140, &round5);
    engine.step(141, &round5);
    engine.recognizer = Some(Box::new(FixedPairRecognizer::new(Score {
        left: 13,
        right: 9,
    })));
    // The 3-1 -> 13-9 jump adds 18 rounds, so it needs 765 s of play since 141.
    engine.step(905, &round5);
    engine.step(910, &round5);
    assert!(publisher.get().scores.is_empty(), "shadow published");

    let events = events.lock().unwrap();
    let mut observed = false;
    let mut accepted = false;
    let mut ended = false;
    for event in events.iter() {
        if event.kind == "frame_observation" && event.second == 140 {
            observed = event.plate_ok
                && event.visual_left == "TL"
                && event.riot_blue == "G2"
                && event.left_score == Some(3)
                && event.right_score == Some(1);
        }
        if event.kind == "video_accepted" && event.second == 141 {
            accepted = event.video_score.as_ref()
                == Some(&ScoreMap::from([
                    ("TL".to_string(), 3),
                    ("G2".to_string(), 1),
                ]));
        }
        if event.kind == "map_end" {
            ended = event.visual_left == "TL"
                && event.video_score.as_ref()
                    == Some(&ScoreMap::from([
                        ("TL".to_string(), 13),
                        ("G2".to_string(), 9),
                    ]));
        }
    }
    assert!(observed, "missing shadow observation: {events:?}");
    let plates = engine.plates.as_ref().unwrap();
    assert_eq!(
        (plates.count("TL", 1080), plates.count("G2", 1080)),
        (KEEP, KEEP),
        "the stale crops must be replaced, not added to"
    );
    assert!(accepted, "missing accepted score: {events:?}");
    assert!(ended, "missing final evidence: {events:?}");
}

// A change emerging from a visible but unreadable plate must not be published until
// a periodic recognizer re-check after the hold window reads it.
#[test]
fn test_suspect_change_waits_for_corroboration() {
    let (mut tracker, riot) = test_tracker();
    {
        let mut state = riot.get();
        state.live = true;
        state.event = true;
    }
    let publisher = FakePublisher::default();
    tracker.publish = Some(Box::new(publisher.clone()));
    set_active(
        &mut tracker,
        MapRun {
            name: "Summit".into(),
            match_id: "match".into(),
            map_number: 2,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            score: Score { left: 11, right: 4 },
            ..Default::default()
        },
    );
    let frame = load_frame("summit-12-4.jpg");
    tracker.step(1, &occluded_plate_frame()); // the plate is visible but unreadable
    tracker.step(2, &frame); // the change emerges from the withhold -> suspect
    publisher.set_enabled(true); // start publishing only after the suspect change appears
    tracker.step(3, &frame); // second identical read -> recorded, publication withheld
    assert_eq!(scores(active(&tracker)), (12, 4), "recorded");
    {
        let state = publisher.get();
        assert!(
            !state.scores.is_empty() && state.scores[0].score["LOUD"] == 11,
            "first published score = {:?}, want the previous 11-4",
            state.scores
        );
        for published in &state.scores {
            assert!(
                published.score["LOUD"] != 12,
                "suspect change published before corroboration: {:?}",
                published.score
            );
        }
    }
    for second in 4..=9 {
        tracker.step(second, &frame); // no periodic check past the hold window yet
    }
    tracker.step(10, &frame); // a periodic check re-reads the score -> released
    let released = publisher
        .get()
        .scores
        .iter()
        .any(|published| published.score["LOUD"] == 12);
    assert!(released, "corroborated score was never published");
}

#[test]
fn test_new_suspect_change_refreshes_hold_and_reports_error() {
    let (mut tracker, riot) = test_tracker();
    tracker.recognizer = Some(Box::new(FixedPairRecognizer::new(Score {
        left: 13,
        right: 11,
    })));
    {
        let mut state = riot.get();
        state.live = true;
        state.event = true;
    }
    let publisher = FakePublisher::new(true);
    tracker.publish = Some(Box::new(publisher.clone()));
    let index = set_active(
        &mut tracker,
        MapRun {
            name: "Ascent".into(),
            match_id: "match".into(),
            map_number: 1,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            score: Score {
                left: 12,
                right: 11,
            },
            held: Some((
                3,
                ScoreMap::from([("LOUD".to_string(), 12), ("EDG".to_string(), 10)]),
            )),
            ..Default::default()
        },
    );
    tracker.prev_plate_withheld = true;
    let frame = zero_zero_frame();

    tracker.step(9, &frame);
    tracker.step(10, &frame);

    let run = &tracker.maps[index];
    assert!(!run.ended, "must not have ended");
    assert_eq!(
        run.held,
        Some((
            10,
            ScoreMap::from([("LOUD".to_string(), 12), ("EDG".to_string(), 10)])
        )),
        "new suspect hold keeps the held score"
    );
    let state = publisher.get();
    let last = state.scores.last().expect("a publication");
    assert_eq!(last.status, Status::Error, "status while withheld");
    assert_eq!(last.score["LOUD"], 12);
    assert_eq!(last.score["EDG"], 10);
}

// Both language implementations replay these same captured Lotus frames.
#[test]
fn test_lotus_banner_occlusion_matches_captured_score_sequence() {
    let (mut tracker, riot) = test_tracker();
    tracker.recognizer = Some(Box::new(vision_recognizer()));
    {
        let mut state = riot.get();
        state.live = true;
        state.event = true;
    }
    let publisher = FakePublisher::new(true);
    tracker.publish = Some(Box::new(publisher.clone()));
    set_active(
        &mut tracker,
        MapRun {
            name: "Lotus".into(),
            match_id: "match".into(),
            map_number: 3,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            score: Score { left: 8, right: 3 },
            ..Default::default()
        },
    );

    for second in [
        15978, 15979, 15980, 15981, 15985, 15986, 16130, 16131, 16132, 16133, 16134, 16135,
    ] {
        publisher.get().current_second = second;
        tracker.step(
            second,
            &load_band_frame(&format!("lotus-occlusion-{second}.jpg")),
        );
    }

    let state = publisher.get();
    let mut score_changes: Vec<Score> = Vec::new();
    let mut published_nine_at: Vec<i64> = Vec::new();
    for (i, published) in state.scores.iter().enumerate() {
        let score = Score {
            left: published.score["LOUD"],
            right: published.score["EDG"],
        };
        if score_changes.last() != Some(&score) {
            score_changes.push(score);
        }
        if score == (Score { left: 9, right: 3 }) {
            published_nine_at.push(state.seconds[i]);
        }
    }
    assert_eq!(
        score_changes,
        vec![Score { left: 8, right: 3 }, Score { left: 9, right: 3 }],
        "published score changes"
    );
    assert!(
        published_nine_at
            .first()
            .is_some_and(|second| *second >= 16131),
        "9-3 publication seconds = {published_nine_at:?}"
    );
}

// The real GE-VIT right-plate 8 reads at 0.965-0.971. Keep that valid score live.
#[test]
fn test_legitimate_low_confidence_eight_advances_score() {
    let (mut tracker, riot) = test_tracker();
    tracker.recognizer = Some(Box::new(vision_recognizer()));
    {
        let mut state = riot.get();
        state.live = true;
        state.event = true;
    }
    let publisher = FakePublisher::new(true);
    tracker.publish = Some(Box::new(publisher.clone()));
    set_active(
        &mut tracker,
        MapRun {
            name: "Abyss".into(),
            match_id: "match".into(),
            map_number: 1,
            left_code: "GE".into(),
            right_code: "VIT".into(),
            score: Score { left: 4, right: 7 },
            ..Default::default()
        },
    );

    for second in [4322, 4324] {
        publisher.get().current_second = second;
        tracker.step(
            second,
            &load_band_frame(&format!("ge-vit-abyss-{second}.jpg")),
        );
    }
    assert_eq!(
        scores(active(&tracker)),
        (4, 8),
        "valid low-confidence read"
    );
    let state = publisher.get();
    for (i, published) in state.scores.iter().enumerate() {
        if published.score["GE"] == 4 && published.score["VIT"] == 8 && state.seconds[i] >= 4324 {
            return;
        }
    }
    panic!("4-8 was not published after the second valid read");
}

// A suspect match-point read stays live and out of the series tally until the hold
// is corroborated.
#[test]
fn test_suspect_match_point_waits_to_end_map() {
    let (mut tracker, _) = test_tracker();
    tracker.recognizer = Some(Box::new(TerminalScoreRecognizer::default()));
    let index = set_active(
        &mut tracker,
        MapRun {
            name: "Ascent".into(),
            match_id: "match".into(),
            map_number: 1,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            score: Score {
                left: 12,
                right: 11,
            },
            ..Default::default()
        },
    );
    tracker.series_wins.insert(
        "match".into(),
        ScoreMap::from([("LOUD".to_string(), 0), ("EDG".to_string(), 0)]),
    );
    let events = capture(&mut tracker);
    let frame = occluded_plate_frame();

    tracker.step(1, &frame);
    tracker.step(2, &frame);
    tracker.step(3, &frame);
    assert_eq!(
        scores(&tracker.maps[index]),
        (13, 11),
        "suspect score recorded"
    );
    assert!(!tracker.maps[index].ended, "must not end early");
    assert_eq!(tracker.series_wins["match"]["LOUD"], 0);

    tracker.step(5, &frame);
    assert!(
        !tracker.maps[index].ended,
        "map ended before the hold window elapsed"
    );
    tracker.step(10, &frame);
    assert!(
        tracker.maps[index].ended,
        "corroborated match point did not end"
    );
    assert_eq!(tracker.series_wins["match"]["LOUD"], 1);

    let events = events.lock().unwrap();
    for event in events.iter() {
        if event.kind == "map_end" {
            assert!(event.second >= 10, "map_end emitted before corroboration");
        }
    }
    let mut held = false;
    let mut held_round = false;
    let mut corroborated = false;
    for event in events.iter() {
        match event.kind.as_str() {
            "score_held" => {
                held = event.score.as_ref().is_some_and(|s| s["LOUD"] == 12)
                    && event.read_score.as_ref().is_some_and(|s| s["LOUD"] == 13);
            }
            "round" => held_round = event.held,
            "score_corroborated" => {
                corroborated =
                    event.second == 10 && event.score.as_ref().is_some_and(|s| s["LOUD"] == 13);
            }
            _ => {}
        }
    }
    assert!(
        held && held_round && corroborated,
        "hold evidence events = held:{held} round:{held_round} corroborated:{corroborated}: {events:?}"
    );
    drop(events);

    // The HUD can vanish after a suspect winning read before a second read confirms
    // it. Adoption must carry the suspect flag into the hold instead of publishing
    // the round clean just because the plate is no longer readable.
    let (mut tracker, _) = test_tracker();
    tracker.recognizer = Some(Box::new(TerminalScoreRecognizer::default()));
    let index = set_active(
        &mut tracker,
        MapRun {
            name: "Ascent".into(),
            match_id: "match".into(),
            map_number: 1,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            score: Score {
                left: 12,
                right: 11,
            },
            ..Default::default()
        },
    );
    tracker.series_wins.insert(
        "match".into(),
        ScoreMap::from([("LOUD".to_string(), 0), ("EDG".to_string(), 0)]),
    );
    let events = capture(&mut tracker);
    let frame = occluded_plate_frame();
    tracker.step(1, &frame); // visible but unreadable: the next read is suspect
    tracker.step(2, &frame); // the suspect 13-11 winning read is remembered
    tracker.step(3, &load_frame("frag-out-084033.png")); // the HUD disappears
    assert!(
        !tracker.maps[index].ended,
        "a suspect read ended the map through the HUD disappearance"
    );
    let events = events.lock().unwrap();
    assert!(
        !events.iter().any(|event| event.kind == "map_end"),
        "adopted a suspect read as clean: {events:?}"
    );
    let held = events.iter().any(|event| event.kind == "score_held");
    let held_round = events
        .iter()
        .any(|event| event.kind == "round" && event.held && event.winner == "LOUD");
    assert!(
        held && held_round,
        "want the adopted suspect read held: held:{held} round:{held_round}: {events:?}"
    );
}

#[test]
fn test_rejected_read_includes_observed_score() {
    let (mut tracker, _) = test_tracker();
    let index = set_active(
        &mut tracker,
        MapRun {
            name: "Ascent".into(),
            match_id: "match".into(),
            map_number: 1,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            score: Score { left: 8, right: 3 },
            ..Default::default()
        },
    );
    let events = capture(&mut tracker);
    tracker.record_score(index, Score { left: 7, right: 3 }, 5, false);

    let events = events.lock().unwrap();
    assert_eq!(events.len(), 1, "reject evidence");
    assert_eq!(events[0].kind, "reject");
    assert_eq!(
        events[0].read_score,
        Some(ScoreMap::from([
            ("LOUD".to_string(), 7),
            ("EDG".to_string(), 3)
        ])),
        "want observed score LOUD 7, EDG 3"
    );
}

// Riot confirms a hidden final from the last corroborated match-point score after a
// suspect final read.
#[test]
fn test_hidden_map_end_from_riot() {
    let (mut tracker, riot) = test_tracker();
    let index = set_active(
        &mut tracker,
        MapRun {
            name: "Summit".into(),
            match_id: "match".into(),
            map_number: 2,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            score: Score { left: 13, right: 4 },
            held: Some((
                50,
                ScoreMap::from([("LOUD".to_string(), 12), ("EDG".to_string(), 4)]),
            )),
            ..Default::default()
        },
    );
    tracker.series_wins.insert(
        "match".into(),
        ScoreMap::from([("LOUD".to_string(), 0), ("EDG".to_string(), 0)]),
    );
    {
        let mut state = riot.get();
        state.states = BTreeMap::from([(2, "completed".to_string())]);
        state.wins = ScoreMap::from([("LOUD".to_string(), 1), ("EDG".to_string(), 0)]);
    }
    tracker.last_gameplay_full = Some(3);
    let events = capture(&mut tracker);
    tracker.step(24, &load_frame("lotus-half-time.jpg"));

    let run = &tracker.maps[index];
    assert!(run.ended, "hidden end");
    assert_eq!(scores(run), (13, 4));
    assert_eq!(tracker.series_wins["match"]["LOUD"], 1);

    let events = events.lock().unwrap();
    let found = events.iter().any(|event| {
        // The leader comes from the corroborated 12-4, the score shown is the withheld 13-4.
        event.kind == "pending_end"
            && event.held
            && event.winner == "LOUD"
            && event.score.as_ref().is_some_and(|s| s["LOUD"] == 13)
    });
    assert!(found, "pending end evidence = {events:?}");
}

// A side correction that arrives before Riot confirms a hidden final must settle the
// map under the corrected leader.
#[test]
fn test_side_correction_before_riot_hidden_end_uses_corrected_leader() {
    let (mut tracker, riot) = test_tracker();
    riot.get().states = BTreeMap::from([(1, "inProgress".to_string())]);
    let index = set_active(
        &mut tracker,
        MapRun {
            name: "Ascent".into(),
            match_id: "match".into(),
            map_number: 1,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            score: Score { left: 12, right: 4 },
            wins_baseline: ScoreMap::from([("LOUD".to_string(), 0), ("EDG".to_string(), 0)]),
            ..Default::default()
        },
    );
    tracker.series_wins.insert(
        "match".into(),
        ScoreMap::from([("LOUD".to_string(), 0), ("EDG".to_string(), 0)]),
    );
    let events = capture(&mut tracker);

    tracker.check_pending_end(
        10,
        Scene::Gameplay,
        true,
        Some(Score { left: 12, right: 4 }),
    );
    assert_eq!(tracker.last_gameplay_full, Some(10));
    // The scoreboard vanishes at match point, so an end is armed from our own labels.
    tracker.check_pending_end(35, Scene::Replay, false, None);
    assert!(tracker.pending_since.is_some(), "an end armed");
    assert!(!tracker.maps[index].ended);

    riot.get().sides.insert(
        1,
        Sides {
            blue: "EDG".into(),
            red: "LOUD".into(),
        },
    );
    tracker.correct_sides(40);
    tracker.correct_sides(40 + REVERSAL_STABLE_SECONDS);
    {
        let mut state = riot.get();
        state.states.insert(1, "completed".to_string());
        state.wins = ScoreMap::from([("LOUD".to_string(), 0), ("EDG".to_string(), 1)]);
    }
    tracker.check_pending_end(40 + REVERSAL_STABLE_SECONDS, Scene::Replay, false, None);

    assert!(
        tracker.maps[index].ended,
        "hidden end after side correction"
    );
    assert_eq!(tracker.series_wins["match"]["EDG"], 1, "want EDG to win");
    let events = events.lock().unwrap();
    let found = events.iter().any(|event| {
        event.kind == "map_end"
            && event.winner == "EDG"
            && event.score.as_ref().is_some_and(|s| s["EDG"] == 13)
    });
    assert!(
        found,
        "want hidden map end for EDG after side correction: {events:?}"
    );
}

#[test]
fn test_side_correction_relabels_held_score_before_hidden_final() {
    let (mut tracker, riot) = test_tracker();
    {
        let mut state = riot.get();
        state.live = true;
        state.event = true;
        state.sides.insert(
            1,
            Sides {
                blue: "EDG".into(),
                red: "LOUD".into(),
            },
        );
    }
    let publisher = FakePublisher::new(true);
    tracker.publish = Some(Box::new(publisher.clone()));
    let index = set_active(
        &mut tracker,
        MapRun {
            name: "Ascent".into(),
            match_id: "match".into(),
            map_number: 1,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            score: Score {
                left: 13,
                right: 11,
            },
            held: Some((
                20,
                ScoreMap::from([("LOUD".to_string(), 12), ("EDG".to_string(), 11)]),
            )),
            ..Default::default()
        },
    );
    let events = capture(&mut tracker);
    tracker.publish_run(index, 29);

    tracker.correct_sides(30);
    tracker.correct_sides(30 + REVERSAL_STABLE_SECONDS); // the swap waits until Riot holds it
    assert!(
        !tracker.maps[index].ended,
        "side correction ended a held map"
    );
    assert_eq!(tracker.series_wins.get("match").map_or(0, |w| w["EDG"]), 0);

    let events = events.lock().unwrap();
    let mut corrected = false;
    for event in events.iter() {
        match event.kind.as_str() {
            "map_end" => panic!("side correction produced a map end: {event:?}"),
            "sides_corrected" => {
                corrected = true;
                // The event reports the run's tracked (still withheld) 13-11 under
                // the new labels: publication keeps carrying the corroborated 12-11.
                assert!(
                    event.held
                        && event
                            .score
                            .as_ref()
                            .is_some_and(|s| s["EDG"] == 13 && s["LOUD"] == 11),
                    "sides_corrected under a hold = {event:?}, want the withheld 13-11 marked held"
                );
            }
            _ => {}
        }
    }
    assert!(
        corrected,
        "side correction emitted no sides_corrected event"
    );
    drop(events);

    tracker.publish_run(index, 30 + REVERSAL_STABLE_SECONDS);
    {
        let state = publisher.get();
        assert_eq!(
            state.scores.len(),
            2,
            "want an immediate corrected-score publish"
        );
        assert_eq!(
            state.scores[1].score,
            ScoreMap::from([("EDG".to_string(), 12), ("LOUD".to_string(), 11)]),
            "corrected published score"
        );
        assert_eq!(
            state.scores[0].score,
            ScoreMap::from([("LOUD".to_string(), 12), ("EDG".to_string(), 11)]),
            "prior published snapshot changed after side correction"
        );
    }

    let run = &tracker.maps[index];
    assert_eq!(run.left_code, "EDG");
    assert_eq!(run.right_code, "LOUD");
    assert_eq!(run.match_point_leader(), "EDG");
    tracker.finish_hidden_map(index, "EDG", "riot", 40);
    assert_eq!(
        scores(&tracker.maps[index]),
        (13, 11),
        "hidden final by corrected order"
    );
}

// A map start publishes immediately, and Riot's reversal must hold before the run
// relabels; while pending the tracker publishes an error repeating the last score.
#[test]
fn test_side_correction_republishes_corrected_tie() {
    let (mut tracker, riot) = test_tracker();
    {
        let mut state = riot.get();
        state.live = true;
        state.event = true;
        state.teams = vec![
            Team {
                id: "1".into(),
                code: "G2".into(),
                name: "G2".into(),
            },
            Team {
                id: "2".into(),
                code: "PRX".into(),
                name: "Paper Rex".into(),
            },
        ];
        state.order = Sides {
            blue: "G2".into(),
            red: "PRX".into(),
        };
        state.sides.insert(
            1,
            Sides {
                blue: "G2".into(),
                red: "PRX".into(),
            },
        );
    }
    let publisher = FakePublisher::new(true);
    tracker.publish = Some(Box::new(publisher.clone()));
    let recognizer = FixedPairRecognizer::new(Score::default());
    tracker.recognizer = Some(Box::new(recognizer.clone()));
    tracker.map_order = Some(Box::new(FakeMapOrder::new(&[("match", &["Split"])])));
    let events = capture(&mut tracker);
    let step =
        |tracker: &mut vbt::tracker::Tracker, second: i64, frame: &vbt_vision::frame::Frame| {
            publisher.get().current_second = second;
            tracker.step(second, frame);
        };

    step(&mut tracker, 5, &zero_zero_frame());
    step(&mut tracker, 10, &zero_zero_frame()); // map starts with G2 left, PRX right
    {
        let run = tracker.active_run().expect("active run");
        assert_eq!(run.left_code, "G2");
        assert_eq!(run.right_code, "PRX");
    }
    {
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 1, "map start events = {events:?}");
        assert_eq!(events[0].kind, "map_start");
    }
    {
        let state = publisher.get();
        assert_eq!(state.scores.len(), 1, "map start publications");
        assert_eq!(state.scores[0].status, Status::Ok);
        assert_eq!(state.scores[0].score["G2"], 0);
        assert_eq!(state.scores[0].score["PRX"], 0);
        assert_eq!(state.blues[0], "G2", "map start sides");
    }

    recognizer.get().pair = Score { left: 1, right: 0 };
    let frame = occluded_plate_frame();
    step(&mut tracker, 11, &frame);
    step(&mut tracker, 12, &frame); // G2's left plate takes round 1
    recognizer.get().pair = Score { left: 1, right: 1 };
    step(&mut tracker, 13, &frame);
    step(&mut tracker, 14, &frame); // PRX's right plate ties it at 1-1
    {
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 3, "round events = {events:?}");
        assert_eq!(events[1].kind, "round");
        assert_eq!(events[1].winner, "G2");
        assert_eq!(events[2].kind, "round");
        assert_eq!(events[2].winner, "PRX");
    }
    {
        let state = publisher.get();
        assert_eq!(
            state.scores.len(),
            3,
            "published scores = {:?}",
            state.scores
        );
        assert_eq!(state.scores[2].score["G2"], 1);
        assert_eq!(state.scores[2].score["PRX"], 1);
        assert_eq!(state.scores[2].status, Status::Ok);
    }

    riot.get().sides.insert(
        1,
        Sides {
            blue: "PRX".into(),
            red: "G2".into(),
        },
    ); // Riot reverses at a tie
    step(&mut tracker, 15, &frame);
    assert_eq!(events.lock().unwrap().len(), 3, "no sides_corrected yet");
    assert_eq!(
        tracker.active_run().unwrap().left_code,
        "G2",
        "swapped too early"
    );
    {
        let state = publisher.get();
        assert_eq!(state.scores.len(), 4, "pending reversal publications");
        assert_eq!(state.scores[3].status, Status::Error);
        assert_eq!(state.scores[3].score["G2"], 1);
        assert_eq!(state.scores[3].score["PRX"], 1);
        assert_eq!(state.blues[3], "G2");
    }

    step(&mut tracker, 15 + REVERSAL_STABLE_SECONDS - 1, &frame);
    assert_eq!(
        events.lock().unwrap().len(),
        3,
        "reversal applied before stable"
    );
    assert_eq!(publisher.get().scores.len(), 4);

    step(&mut tracker, 15 + REVERSAL_STABLE_SECONDS, &frame);
    {
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 4, "want sides_corrected");
        assert_eq!(events[3].kind, "sides_corrected");
    }
    {
        let run = tracker.active_run().unwrap();
        assert_eq!(run.left_code, "PRX");
        assert_eq!(run.right_code, "G2");
    }
    // The per-team aggregate is tied, so only the changed sides distinguish this
    // publication; it is the retraction the tracker must publish.
    let state = publisher.get();
    assert_eq!(
        state.scores.len(),
        5,
        "want an immediate corrected publish at the tie"
    );
    let last = &state.scores[4];
    assert_eq!(last.score["PRX"], 1);
    assert_eq!(last.score["G2"], 1);
    assert_eq!(last.status, Status::Ok);
    assert_eq!(state.blues[4], "PRX", "corrected publish sides");
    assert_eq!(
        state.seconds[4],
        15 + REVERSAL_STABLE_SECONDS,
        "corrected publication second"
    );
}

// The 2 Oct 2026 Abyss incident: Riot's cacheable responses reported one game's
// sides flipped three times 90 s apart.
#[test]
fn test_oscillating_side_reversal_waits_for_stable_riot_sides() {
    let (mut tracker, riot) = test_tracker();
    {
        let mut state = riot.get();
        state.live = true;
        state.event = true;
        state.teams = vec![
            Team {
                id: "1".into(),
                code: "G2".into(),
                name: "G2".into(),
            },
            Team {
                id: "2".into(),
                code: "PRX".into(),
                name: "Paper Rex".into(),
            },
        ];
        state.order = Sides {
            blue: "G2".into(),
            red: "PRX".into(),
        };
        state.sides.insert(
            1,
            Sides {
                blue: "G2".into(),
                red: "PRX".into(),
            },
        );
        state.states = BTreeMap::from([(1, "unstarted".to_string())]);
    }
    let publisher = FakePublisher::new(true);
    tracker.publish = Some(Box::new(publisher.clone()));
    let recognizer = FixedPairRecognizer::new(Score::default());
    tracker.recognizer = Some(Box::new(recognizer.clone()));
    tracker.map_order = Some(Box::new(FakeMapOrder::new(&[("match", &["Split"])])));
    let events = capture(&mut tracker);
    let step =
        |tracker: &mut vbt::tracker::Tracker, second: i64, frame: &vbt_vision::frame::Frame| {
            publisher.get().current_second = second;
            tracker.step(second, frame);
        };

    step(&mut tracker, 5, &zero_zero_frame());
    step(&mut tracker, 10, &zero_zero_frame()); // G2-left, PRX-right published at once
    recognizer.get().pair = Score { left: 1, right: 0 };
    let frame = occluded_plate_frame();
    step(&mut tracker, 11, &frame);
    step(&mut tracker, 12, &frame);
    recognizer.get().pair = Score { left: 1, right: 1 };
    step(&mut tracker, 13, &frame);
    step(&mut tracker, 14, &frame);
    recognizer.get().pair = Score { left: 2, right: 1 };
    step(&mut tracker, 15, &frame);
    step(&mut tracker, 16, &frame);
    let n = publisher.get().scores.len();
    {
        let state = publisher.get();
        let second_last = &state.scores[n - 1];
        assert_eq!(second_last.score["G2"], 2, "pre-correction G2 2-1 PRX");
        assert_eq!(second_last.score["PRX"], 1);
    }
    {
        let events = events.lock().unwrap();
        assert_eq!(events[0].riot_state, "unstarted");
        assert_eq!(events[0].riot_blue, "G2");
        assert_eq!(events[0].riot_red, "PRX");
    }

    {
        let mut state = riot.get();
        state.sides.insert(
            1,
            Sides {
                blue: "PRX".into(),
                red: "G2".into(),
            },
        );
        state.states.insert(1, "inProgress".to_string());
    }
    step(&mut tracker, 17, &frame);
    {
        let state = publisher.get();
        assert_eq!(state.scores.len(), n + 1, "pending reversal publications");
        assert_eq!(state.scores[n].status, Status::Error);
    }

    riot.get().sides.insert(
        1,
        Sides {
            blue: "G2".into(),
            red: "PRX".into(),
        },
    ); // ...and back 90 s later
    step(&mut tracker, 107, &frame);
    {
        let state = publisher.get();
        assert_eq!(state.scores.len(), n + 2, "abandoned reversal publications");
        assert_eq!(state.scores[n + 1].status, Status::Ok);
    }

    riot.get().sides.insert(
        1,
        Sides {
            blue: "PRX".into(),
            red: "G2".into(),
        },
    ); // it flips again
    step(&mut tracker, 197, &frame);
    {
        let state = publisher.get();
        assert_eq!(
            state.scores.len(),
            n + 3,
            "second pending reversal publications"
        );
        assert_eq!(state.scores[n + 2].status, Status::Error);
    }
    step(&mut tracker, 197 + REVERSAL_STABLE_SECONDS - 1, &frame);
    {
        let state = publisher.get();
        assert_eq!(state.scores.len(), n + 3, "reversal applied before stable");
        assert_eq!(events.lock().unwrap().len(), 4);
        // While the reversal is pending, every publication repeats the last score
        // and sides, so the original mapping stays on the wire.
        for (i, blue) in state.blues[n..].iter().enumerate() {
            assert_eq!(blue, "G2", "publication {} sides", n + i);
        }
    }
    step(&mut tracker, 197 + REVERSAL_STABLE_SECONDS, &frame);
    {
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 5, "want the side correction: {events:?}");
        assert_eq!(events[4].kind, "sides_corrected");
        assert_eq!(events[4].riot_state, "inProgress");
        assert_eq!(events[4].riot_blue, "PRX");
        assert_eq!(events[4].riot_red, "G2");
    }
    let state = publisher.get();
    let last = state.scores.last().unwrap();
    assert_eq!(
        last.score["PRX"], 2,
        "want the round wins moved to PRX 2-1 G2"
    );
    assert_eq!(last.score["G2"], 1);
    assert_eq!(last.status, Status::Ok);
    assert_eq!(state.blues.last().unwrap(), "PRX");
}

// A map that starts before Riot's getEventDetails succeeds binds the LEFT/RIGHT
// placeholders; when the details arrive, the run must adopt Riot's real codes.
#[test]
fn test_provisional_identity_rebinds_when_riot_details_arrive() {
    let (mut tracker, riot) = test_tracker();
    {
        let mut state = riot.get();
        state.live = true; // the match is live, but its details have not arrived
        state.order = Sides {
            blue: "LEFT".into(),
            red: "RIGHT".into(),
        };
    }
    tracker.recognizer = Some(Box::new(StubRecognizer::new("")));
    tracker.map_order = Some(Box::new(FakeMapOrder::new(&[("match", &["Lotus"])])));
    let publisher = FakePublisher::new(true);
    tracker.publish = Some(Box::new(publisher.clone()));
    let events = capture(&mut tracker);

    let frame = zero_zero_frame();
    tracker.step(5, &frame);
    tracker.step(10, &frame);
    {
        let run = tracker.active_run().expect("active run");
        assert_eq!(run.left_code, "LEFT");
        assert_eq!(run.right_code, "RIGHT");
        assert!(run.provisional, "want a provisional LEFT/RIGHT run");
    }
    assert_eq!(
        publisher.get().scores.len(),
        0,
        "no publications before Riot's details"
    );

    // Riot recovers and lists the match's teams and sides for the game.
    {
        let mut state = riot.get();
        state.event = true;
        state.sides.insert(
            1,
            Sides {
                blue: "LOUD".into(),
                red: "EDG".into(),
            },
        );
    }
    tracker.step(11, &frame);

    {
        let run = tracker.active_run().unwrap();
        assert_eq!(run.left_code, "LOUD");
        assert_eq!(run.right_code, "EDG");
        assert!(
            !run.provisional,
            "want Riot's real codes after its details arrived"
        );
    }
    {
        let events = events.lock().unwrap();
        let corrected = events.iter().any(|event| {
            event.kind == "sides_corrected"
                && event
                    .score
                    .as_ref()
                    .is_some_and(|s| s["LOUD"] == 0 && s["EDG"] == 0)
        });
        assert!(corrected, "want the identity rebind reported: {events:?}");
    }
    {
        let state = publisher.get();
        assert!(
            !state.scores.is_empty(),
            "no publication after the identity rebind"
        );
        let last = state.scores.last().unwrap();
        assert_eq!(last.score["LOUD"], 0);
        assert_eq!(last.score["EDG"], 0);
        assert_eq!(state.blues.last().unwrap(), "LOUD");
    }

    // The rebind clears the provisional flag, so a later reversal still corrects.
    riot.get().sides.insert(
        1,
        Sides {
            blue: "EDG".into(),
            red: "LOUD".into(),
        },
    );
    tracker.step(12, &frame);
    tracker.step(12 + REVERSAL_STABLE_SECONDS, &frame);
    {
        let run = tracker.active_run().unwrap();
        assert_eq!(run.left_code, "EDG");
        assert_eq!(run.right_code, "LOUD");
    }
}

// A pause is confirmed on two consecutive overlay frames, keeps the map's status ok
// through the HUD gap, suppresses the hidden map end while it is on, and only
// clears on two consecutive frames without the overlay once the HUD is back.
#[test]
fn test_pause_keeps_the_map_healthy_and_suppresses_pending_end() {
    let (mut tracker, riot) = test_tracker();
    {
        let mut state = riot.get();
        state.live = true;
        state.event = true;
    }
    let publisher = FakePublisher::new(true);
    tracker.publish = Some(Box::new(publisher.clone()));
    let recognizer = FixedPairRecognizer::new(Score {
        left: 12,
        right: 11,
    });
    tracker.recognizer = Some(Box::new(recognizer.clone()));
    let events = capture(&mut tracker);

    set_active(
        &mut tracker,
        MapRun {
            name: "Ascent".into(),
            match_id: "match".into(),
            map_number: 1,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            score: Score {
                left: 12,
                right: 11,
            },
            last_hud_second: Some(39),
            last_read_second: 39,
            ..Default::default()
        },
    );
    tracker.last_gameplay_full = Some(39);
    let pause_frame = load_frame("tech-pause-2610.jpg");
    let hud_frame = load_frame("summit-12-4.jpg");

    {
        let mut state = recognizer.get();
        state.on = true;
        state.pause = PauseRead {
            kind: "tech_pause".into(),
            reason: String::new(),
            score: 0.79,
        };
    }
    tracker.step(40, &pause_frame);
    assert!(
        tracker.pause.is_none(),
        "one overlay frame must not confirm it"
    );
    tracker.step(41, &pause_frame);
    {
        let pause = tracker.pause.as_ref().expect("pause confirmed");
        assert_eq!(pause.kind, "tech_pause");
        assert_eq!(pause.since, 40);
    }

    // The HUD stays hidden for minutes. No score is read through the gap, but the
    // pause keeps the map healthy, and the match-point HUD gap must not arm an end.
    tracker.step(200, &pause_frame);
    assert!(
        tracker.pending_since.is_none(),
        "pending end armed while paused"
    );
    {
        let run = active(&tracker);
        assert_eq!(run.last_read_second, 39);
        assert_eq!(run.last_hud_second, Some(39));
    }
    {
        let state = publisher.get();
        let last = state.scores.last().expect("a publication");
        assert_eq!(last.status, Status::Ok);
        assert_eq!(last.pause, "tech_pause");
        let pause = state
            .pauses
            .last()
            .unwrap()
            .as_ref()
            .expect("published pause");
        assert_eq!(pause.since, 40);
    }

    // The overlay can vanish before play resumes: without the gameplay HUD the pause holds.
    recognizer.get().on = false;
    tracker.step(201, &pause_frame);
    tracker.step(202, &pause_frame);
    assert!(
        tracker.pause.is_some(),
        "pause cleared while the HUD was absent"
    );
    tracker.step(203, &hud_frame);
    assert!(
        tracker.pause.is_some(),
        "pause cleared on the first HUD frame"
    );
    tracker.step(204, &hud_frame);
    assert!(
        tracker.pause.is_none(),
        "want it cleared after two HUD frames"
    );
    {
        let events = events.lock().unwrap();
        let found = events.iter().any(|event| {
            event.kind == "pause_end" && event.pause == "tech_pause" && event.reason.is_empty()
        });
        assert!(found, "want pause start and end reported: {events:?}");
    }

    // A later overlay confirms a fresh pause after the previous one cleared.
    recognizer.get().on = true;
    tracker.step(300, &pause_frame);
    tracker.step(301, &pause_frame);
    let pause = tracker.pause.as_ref().expect("second pause confirmed");
    assert_eq!(pause.since, 300);
}

#[test]
fn test_held_false_match_point_rolls_back_and_allows_genuine_score() {
    let (mut tracker, riot) = test_tracker();
    {
        let mut state = riot.get();
        state.live = true;
        state.event = true;
    }
    let publisher = FakePublisher::new(true);
    tracker.publish = Some(Box::new(publisher.clone()));
    let recognizer = FixedPairRecognizer::new(Score {
        left: 13,
        right: 10,
    });
    tracker.recognizer = Some(Box::new(recognizer.clone()));
    let index = set_active(
        &mut tracker,
        MapRun {
            name: "Ascent".into(),
            match_id: "match".into(),
            map_number: 1,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            score: Score {
                left: 12,
                right: 10,
            },
            ..Default::default()
        },
    );
    tracker.prev_plate_withheld = true;
    let frame = zero_zero_frame();
    let events = capture(&mut tracker);

    tracker.step(1, &frame);
    tracker.step(2, &frame);
    assert_eq!(
        scores(&tracker.maps[index]),
        (13, 10),
        "suspect match point"
    );
    assert!(!tracker.maps[index].ended);
    assert!(tracker.maps[index].held.is_some());

    recognizer.get().pair = Score {
        left: 12,
        right: 10,
    };
    tracker.step(3, &frame);
    tracker.step(4, &frame);
    {
        let run = &tracker.maps[index];
        assert_eq!(scores(run), (12, 10), "rollback state");
        assert!(!run.ended);
        assert!(run.held.is_none());
    }
    {
        let state = publisher.get();
        let last = state.scores.last().unwrap();
        assert_eq!(last.status, Status::Ok, "rollback publication");
        assert_eq!(last.score["LOUD"], 12);
    }

    recognizer.get().pair = Score {
        left: 13,
        right: 10,
    };
    tracker.step(5, &frame);
    tracker.step(6, &frame);
    {
        let run = &tracker.maps[index];
        assert_eq!(scores(run), (13, 10), "genuine final score");
        assert!(run.ended);
        assert!(run.held.is_none());
    }
    {
        let state = publisher.get();
        let last = state.scores.last().unwrap();
        assert_eq!(last.status, Status::Error, "genuine final publication");
        assert_eq!(last.score["LOUD"], 13);
    }
    let events = events.lock().unwrap();
    assert!(events.len() >= 5, "score evidence ordering = {events:?}");
    assert_eq!(events[2].kind, "score_rollback");
    assert_eq!(events[2].score.as_ref().unwrap()["LOUD"], 12);
    assert_eq!(events[events.len() - 2].kind, "round");
    assert_eq!(events[events.len() - 1].kind, "map_end");
}

#[test]
fn test_held_evidence_marks_clean_changes_while_hold_remains() {
    for (name, pair, second, want) in [
        ("round", Score { left: 5, right: 4 }, 10, "round"),
        ("sync", Score { left: 6, right: 4 }, 60, "sync"),
    ] {
        let (mut tracker, _) = test_tracker();
        let index = set_active(
            &mut tracker,
            MapRun {
                name: "Ascent".into(),
                match_id: "match".into(),
                map_number: 1,
                left_code: "LOUD".into(),
                right_code: "EDG".into(),
                score: Score { left: 4, right: 4 },
                held: Some((
                    1,
                    ScoreMap::from([("LOUD".to_string(), 4), ("EDG".to_string(), 4)]),
                )),
                last_hud_second: Some(0),
                ..Default::default()
            },
        );
        let events = capture(&mut tracker);
        tracker.record_score(index, pair, second, false);
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 1, "{name}: want one held event: {events:?}");
        assert_eq!(events[0].kind, want, "{name}");
        assert!(events[0].held, "{name}: held marker");
    }
}

#[test]
fn test_suspect_sync_evidence_is_marked_held() {
    let (mut tracker, _) = test_tracker();
    let index = set_active(
        &mut tracker,
        MapRun {
            name: "Ascent".into(),
            match_id: "match".into(),
            map_number: 1,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            score: Score { left: 4, right: 4 },
            last_hud_second: Some(0),
            ..Default::default()
        },
    );
    let events = capture(&mut tracker);
    tracker.record_score(index, Score { left: 6, right: 4 }, 60, true);
    let events = events.lock().unwrap();
    assert_eq!(events.len(), 2, "suspect sync evidence = {events:?}");
    assert_eq!(events[0].kind, "score_held");
    assert_eq!(events[1].kind, "sync");
    assert!(events[1].held, "want held marker on sync");
}

// A score that jumps behind an unreadable plate recovers through the contender, whose
// jump allowance counts play since the run's score was last read back: neither the
// unreadable frames nor the rejected jump reads may restart that clock, or the
// recovery stalls. A steady visible jump must never pass as a two-read sync either,
// so a misread on screen cannot skip the contender's gates.
#[test]
fn test_jump_behind_an_unreadable_plate_recovers_through_the_contender() {
    let (mut tracker, _) = test_tracker();
    let index = set_active(
        &mut tracker,
        MapRun {
            name: "Summit".into(),
            match_id: "match".into(),
            map_number: 2,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            score: Score::default(),
            last_hud_second: Some(10),
            last_known_second: Some(10),
            last_read_second: 10,
            ..Default::default()
        },
    );
    let events = capture(&mut tracker);

    // The plate stays on screen but unreadable for two and a half minutes.
    let occluded = occluded_plate_frame();
    for second in (12..=158).step_by(2) {
        tracker.step(second, &occluded);
    }

    // It then reads 6-0. Six rounds need 225 s of play since the last known 0-0,
    // which second 235 reaches; the contender adopts the steady read once it has
    // stood 120 s, at about second 286. Timed from the last unreadable frame instead,
    // the allowance would hold it back until second 383.
    tracker.recognizer = Some(Box::new(FixedPairRecognizer::new(Score {
        left: 6,
        right: 0,
    })));
    let frame = load_frame("summit-12-4.jpg");
    for second in (160..=330).step_by(2) {
        tracker.step(second, &frame);
    }

    assert_eq!(
        scores(&tracker.maps[index]),
        (6, 0),
        "the jump behind the unreadable plate never recovered"
    );
    let events = events.lock().unwrap();
    let changes: Vec<_> = events
        .iter()
        .filter(|event| matches!(event.kind.as_str(), "round" | "sync" | "resync"))
        .map(|event| (event.kind.as_str(), event.source.as_str()))
        .collect();
    assert_eq!(
        changes,
        [("resync", "contender_stable")],
        "the visible jump must be adopted only by the contender"
    );
}

// The incident: a rogue 1-0 -> 10-10 jump after a hidden pause must be rejected,
// and then the real lower series (2-0, 3-0, ...) must resync the run instead of
// being rejected forever.
#[test]
fn test_rogue_jump_during_pause_is_rejected_and_a_lower_series_resyncs() {
    let (mut tracker, _) = test_tracker();
    let index = set_active(
        &mut tracker,
        MapRun {
            name: "Summit".into(),
            match_id: "match".into(),
            map_number: 1,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            score: Score { left: 1, right: 0 },
            last_hud_second: Some(0),
            last_read_second: 0,
            ..Default::default()
        },
    );
    let events = capture(&mut tracker);

    tracker.pause = Some(Pause {
        kind: "tech".into(),
        reason: String::new(),
        since: 2,
    });
    tracker.record_score(
        index,
        Score {
            left: 10,
            right: 10,
        },
        32,
        false,
    );
    assert_eq!(scores(&tracker.maps[index]), (1, 0), "rogue jump recorded");
    {
        let events = events.lock().unwrap();
        assert_eq!(
            events.last().map(|event| event.kind.as_str()),
            Some("reject"),
            "rogue jump evidence = {events:?}"
        );
    }
    tracker.clear_pause(40, "overlay_gone");

    // Stand-in for a bad sync: the run's score is the rogue 10-10.
    tracker.maps[index].score = Score {
        left: 10,
        right: 10,
    };
    tracker.maps[index].contender = None;

    for second in (50..=71).step_by(3) {
        tracker.record_score(index, Score { left: 2, right: 0 }, second, false);
        assert_eq!(
            scores(&tracker.maps[index]),
            (10, 10),
            "series adopted early at {second}"
        );
    }
    tracker.record_score(index, Score { left: 3, right: 0 }, 115, false);

    assert_eq!(scores(&tracker.maps[index]), (3, 0), "resync");
    let events = events.lock().unwrap();
    let last = events.last().expect("a resync event");
    assert_eq!(last.kind, "resync");
    assert_eq!(last.source, "contender_series");
    assert_eq!(
        last.previous_score,
        Some(ScoreMap::from([
            ("LOUD".to_string(), 10),
            ("EDG".to_string(), 10),
        ])),
        "resync must report the score it replaced"
    );
}

// Production, KRU-TLV 2026-10-28: the run sat at 12-9, the map-ending round was never
// read, and Riot still listed game 1 in progress. The next map's 0-0 and 1-0 reads
// formed a contender series that resynced map 1 down to 1-0, so map 1 never ended.
// That series is the next map starting: map 1 must close for the team at match point
// and map 2 start under its veto name, with no resync inside map 1. Map 1 had no strip
// baseline, so its 0-0 strip read could not start map 2; the new run keeps that read as
// map 2's baseline. A run cut at 11-11 has no leader, but its 0-0 opening is the same
// boundary: map 1 closes with no winner instead of resyncing to 1-0.
#[test]
fn test_match_point_contender_at_the_next_maps_opening_starts_the_next_map() {
    for (score, winner) in [
        (Score { left: 12, right: 9 }, "LOUD"),
        (
            Score {
                left: 11,
                right: 11,
            },
            "",
        ),
    ] {
        let (mut tracker, riot) = test_tracker();
        {
            let mut state = riot.get();
            state.game = 1;
            state.states = BTreeMap::from([(1, "inProgress".to_string())]);
        }
        tracker.map_order = Some(Box::new(FakeMapOrder::new(&[(
            "match",
            &["Summit", "Lotus", "Ascent"],
        )])));
        let index = set_active(
            &mut tracker,
            MapRun {
                name: "Summit".into(),
                match_id: "match".into(),
                map_number: 1,
                left_code: "LOUD".into(),
                right_code: "EDG".into(),
                score,
                last_hud_second: Some(0),
                last_known_second: Some(0),
                last_read_second: 0,
                ..Default::default()
            },
        );
        let events = capture(&mut tracker);
        tracker.recognizer = Some(Box::new(StubRecognizer::new("Lotus")));
        tracker.step(5, &zero_zero_frame());

        for second in (10..=70).step_by(6) {
            tracker.record_score(index, Score::default(), second, false);
        }
        tracker.record_score(index, Score { left: 1, right: 0 }, 100, false);

        let events = events.lock().unwrap();
        assert!(
            !events.iter().any(|event| event.kind == "resync"),
            "the next map's opening resynced map 1"
        );
        let end = events
            .iter()
            .find(|event| event.kind == "map_end")
            .expect("map 1 never ended");
        // Riot still lists game 1 in progress, so a leader is only a guess: never exact.
        assert_eq!(
            (
                end.map_number,
                end.winner.as_str(),
                end.source.as_str(),
                end.exact
            ),
            (1, winner, "next_map", Some(false)),
            "{score:?}"
        );
        let active = tracker.active.expect("an active run");
        assert_ne!(active, index, "map 2 never started");
        assert_eq!(
            (
                tracker.maps[active].name.as_str(),
                tracker.maps[active].map_number,
                tracker.maps[active].strip_at_start.as_str()
            ),
            ("Lotus", 2, "Lotus")
        );
        assert_eq!(
            scores(&tracker.maps[active]),
            (1, 0),
            "map 2 must open at the corroborated series, not 0-0"
        );

        // Without a veto name map 2 never opens and the closed map 1 stays active.
        // Riot's credit for game 1 must still settle it before any map 2 baseline
        // could absorb that win: EDG, not the guessed leader, gets map 1.
        let (mut tracker, riot) = test_tracker();
        {
            let mut state = riot.get();
            state.game = 1;
            state.states = BTreeMap::from([(1, "inProgress".to_string())]);
        }
        let index = set_active(
            &mut tracker,
            MapRun {
                name: "Summit".into(),
                match_id: "match".into(),
                map_number: 1,
                left_code: "LOUD".into(),
                right_code: "EDG".into(),
                score,
                last_hud_second: Some(0),
                last_known_second: Some(0),
                last_read_second: 0,
                ..Default::default()
            },
        );
        let captured = capture(&mut tracker);
        for second in (10..=70).step_by(6) {
            tracker.record_score(index, Score::default(), second, false);
        }
        tracker.record_score(index, Score { left: 1, right: 0 }, 100, false);
        assert_eq!(
            (tracker.active, tracker.maps[index].ended),
            (Some(index), true)
        );
        {
            let mut state = riot.get();
            state.states = BTreeMap::from([(1, "completed".to_string())]);
            state.wins = ScoreMap::from([("EDG".to_string(), 1)]);
        }
        tracker.step(120, &zero_zero_frame());
        let events = captured.lock().unwrap();
        let settled = events.last().expect("no events");
        assert_eq!(
            (
                settled.kind.as_str(),
                settled.winner.as_str(),
                settled.source.as_str(),
                settled.exact
            ),
            ("map_end", "EDG", "riot", Some(false)),
            "{score:?}: Riot never settled the closed map: {events:?}"
        );
        assert_eq!(
            (
                tracker.series_wins["match"]
                    .get("LOUD")
                    .copied()
                    .unwrap_or(0),
                tracker.series_wins["match"]
                    .get("EDG")
                    .copied()
                    .unwrap_or(0)
            ),
            (0, 1),
            "{score:?}: the series credit did not settle on EDG"
        );
    }
}

// A false match-point score (12-2 from a bad read, really 2-2) or late score (11-5,
// really 4-4) is not a map boundary: a steady series that opened mid-map corrects the
// run instead of ending it.
#[test]
fn test_match_point_contender_from_mid_map_corrects_a_false_high_score() {
    for (score, real) in [
        (Score { left: 12, right: 2 }, Score { left: 2, right: 2 }),
        (Score { left: 11, right: 5 }, Score { left: 4, right: 4 }),
    ] {
        let (mut tracker, _) = test_tracker();
        tracker.map_order = Some(Box::new(FakeMapOrder::new(&[(
            "match",
            &["Summit", "Lotus", "Ascent"],
        )])));
        let index = set_active(
            &mut tracker,
            MapRun {
                name: "Summit".into(),
                match_id: "match".into(),
                map_number: 1,
                left_code: "LOUD".into(),
                right_code: "EDG".into(),
                score,
                last_hud_second: Some(0),
                last_known_second: Some(0),
                last_read_second: 0,
                ..Default::default()
            },
        );
        let events = capture(&mut tracker);
        for second in (10..=136).step_by(6) {
            tracker.record_score(index, real, second, false);
        }

        assert_eq!(
            tracker.active,
            Some(index),
            "a false {score:?} started the next map"
        );
        assert_eq!(
            scores(&tracker.maps[index]),
            (real.left, real.right),
            "never resynced"
        );
        let events = events.lock().unwrap();
        assert!(
            !events.iter().any(|event| event.kind == "map_end"),
            "a false {score:?} ended the map: {events:?}"
        );
    }
}

// A rogue jump that holds still during a tech pause must never replace the run's
// score, no matter how many reads or how much wall time pass: pause time is not
// play time, so neither the stable path nor a rate-limited advance may adopt it.
#[test]
fn test_rogue_stable_contender_is_not_adopted_during_pause() {
    let (mut tracker, _) = test_tracker();
    let index = set_active(
        &mut tracker,
        MapRun {
            name: "Summit".into(),
            match_id: "match".into(),
            map_number: 1,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            score: Score { left: 1, right: 0 },
            last_hud_second: Some(0),
            last_read_second: 0,
            ..Default::default()
        },
    );
    let events = capture(&mut tracker);
    tracker.pause = Some(Pause {
        kind: "tech".into(),
        reason: String::new(),
        since: 10,
    });

    // The rogue 10-10 is rejected and becomes a contender, then every read for
    // more than two minutes matches it while the pause keeps running.
    tracker.record_score(
        index,
        Score {
            left: 10,
            right: 10,
        },
        30,
        false,
    );
    for second in (32..=160).step_by(2) {
        tracker.record_score(
            index,
            Score {
                left: 10,
                right: 10,
            },
            second,
            false,
        );
    }
    assert_eq!(
        scores(&tracker.maps[index]),
        (1, 0),
        "rogue jump adopted as a stable contender during the pause"
    );

    // A paused series cannot ratchet either: the 1-round read is refused by the
    // cumulative play-time bound and restarts the series instead of advancing it.
    tracker.record_score(
        index,
        Score {
            left: 11,
            right: 10,
        },
        162,
        false,
    );
    {
        let contender = tracker.maps[index].contender.expect("a contender");
        assert_eq!(contender.advances, 0, "paused increment was kept");
        assert_eq!(contender.first, contender.latest, "series restarted");
    }
    for second in (164..=284).step_by(2) {
        tracker.record_score(
            index,
            Score {
                left: 11,
                right: 10,
            },
            second,
            false,
        );
    }
    // Clearing the pause does not retroactively earn the jump play time.
    tracker.clear_pause(286, "overlay_gone");
    tracker.record_score(
        index,
        Score {
            left: 11,
            right: 10,
        },
        288,
        false,
    );
    assert_eq!(
        scores(&tracker.maps[index]),
        (1, 0),
        "rogue jump adopted after the pause cleared"
    );
    let events = events.lock().unwrap();
    assert!(
        events.iter().all(|event| event.kind != "resync"),
        "resync evidence = {events:?}"
    );
}

// A rejected series cannot gain rounds faster than one per MIN_ROUND_SECONDS of
// unpaused play: each faster 1-round read restarts it, and an advance that really
// has the play time behind it is kept.
#[test]
fn test_contender_advances_are_rate_limited_by_play_time() {
    let (mut tracker, _) = test_tracker();
    let index = set_active(
        &mut tracker,
        MapRun {
            name: "Summit".into(),
            match_id: "match".into(),
            map_number: 1,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            score: Score { left: 1, right: 0 },
            last_hud_second: Some(0),
            last_read_second: 0,
            ..Default::default()
        },
    );
    let events = capture(&mut tracker);

    // 5-0 is rejected as a jump; each following read adds a round a second later,
    // faster than 45 s of play per round, so the series restarts every time.
    tracker.record_score(index, Score { left: 5, right: 0 }, 10, false);
    for (step, left) in (6..=12).enumerate() {
        tracker.record_score(index, Score { left, right: 0 }, 11 + step as i64, false);
    }
    {
        let contender = tracker.maps[index].contender.expect("a contender");
        assert_eq!(contender.advances, 0, "rapid increments were kept");
        assert_eq!(
            (contender.first.left, contender.latest.left),
            (12, 12),
            "series state after the rapid reads"
        );
    }
    assert_eq!(
        scores(&tracker.maps[index]),
        (1, 0),
        "rapid series was adopted"
    );

    // Holding 12-0 for the next 45 s earns the one-round advance to 13-0.
    for second in 18..=61 {
        tracker.record_score(index, Score { left: 12, right: 0 }, second, false);
    }
    tracker.record_score(index, Score { left: 13, right: 0 }, 62, false);
    {
        let contender = tracker.maps[index].contender.expect("a contender");
        assert_eq!(
            (contender.latest.left, contender.advances),
            (13, 1),
            "an advance backed by 45 s of play must be kept"
        );
    }
    assert_eq!(
        scores(&tracker.maps[index]),
        (1, 0),
        "the young series resynced before it earned its age"
    );
    let events = events.lock().unwrap();
    assert!(
        events.iter().all(|event| event.kind != "resync"),
        "resync evidence = {events:?}"
    );
}

// Clearing a pause must not retroactively credit its wall time as contender play:
// the pause that ran after the series started is subtracted from the series'
// unpaused time, so a one-round advance right after the pause ends has not earned
// its MIN_ROUND_SECONDS and restarts the series instead of being kept.
#[test]
fn test_cleared_pause_does_not_inflate_contender_advances() {
    let (mut tracker, _) = test_tracker();
    let index = set_active(
        &mut tracker,
        MapRun {
            name: "Summit".into(),
            match_id: "match".into(),
            map_number: 1,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            score: Score { left: 1, right: 0 },
            last_hud_second: Some(0),
            last_read_second: 0,
            ..Default::default()
        },
    );
    let events = capture(&mut tracker);

    // 5-0 is rejected as a jump and starts a contender at second 10.
    tracker.record_score(index, Score { left: 5, right: 0 }, 10, false);
    tracker.record_score(index, Score { left: 5, right: 0 }, 20, false);
    // A long pause runs and clears; its wall time is not unpaused play.
    tracker.pause = Some(Pause {
        kind: "tech".into(),
        reason: String::new(),
        since: 30,
    });
    tracker.clear_pause(110, "overlay_gone");
    // Only 21 unpaused seconds have passed since the series started at second 10,
    // so the 5-0 -> 6-0 advance has not earned its 45 s and must restart it.
    tracker.record_score(index, Score { left: 6, right: 0 }, 111, false);
    let contender = tracker.maps[index].contender.expect("a contender");
    assert_eq!(
        contender.advances, 0,
        "cleared pause wall time was credited"
    );
    assert_eq!(contender.first, contender.latest, "series restarted");
    assert_eq!(
        scores(&tracker.maps[index]),
        (1, 0),
        "unearned series was adopted"
    );
    let events = events.lock().unwrap();
    assert!(
        events.iter().all(|event| event.kind != "resync"),
        "resync evidence = {events:?}"
    );
}

// A mixed series can raise one side while dropping the other (1-2 -> 10-1); a run
// that only bounded both-sides-up gains adopted it regardless of the net rounds.
// The jump allowance must count net rounds, so a stable mixed series does not
// resync until its play time earns the difference.
#[test]
fn test_mixed_score_contender_jump_is_bounded() {
    let (mut tracker, _) = test_tracker();
    let index = set_active(
        &mut tracker,
        MapRun {
            name: "Summit".into(),
            match_id: "match".into(),
            map_number: 1,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            score: Score { left: 1, right: 2 },
            last_hud_second: Some(0),
            last_read_second: 0,
            ..Default::default()
        },
    );
    let events = capture(&mut tracker);

    // 10-1 nets 8 rounds over the run's 1-2. The series stands for 168 s with 57
    // reads, well past CONTENDER_STABLE_SECONDS, but 198 s of play earns at most
    // 5, so the mixed jump must never be adopted.
    for second in (30..=198).step_by(3) {
        tracker.record_score(index, Score { left: 10, right: 1 }, second, false);
    }
    assert_eq!(
        scores(&tracker.maps[index]),
        (1, 2),
        "mixed jump resynced without earning its net rounds"
    );
    let events = events.lock().unwrap();
    assert!(
        events.iter().all(|event| event.kind != "resync"),
        "resync evidence = {events:?}"
    );
}

// The veto's name for the game number wins over the strip, which can misread an
// unknown map as a known one, and starts the map on the CURRENT anchor alone when
// the strip name is unreadable.
#[test]
fn test_veto_names_map_at_start() {
    let cases: [(&str, &str, &[&str], &str); 3] = [
        (
            "veto overrides a wrong strip name",
            "Summit",
            &["Lotus", "Ascent"],
            "Lotus",
        ),
        (
            "veto names the map when the strip name is unreadable",
            "",
            &["Lotus", "Ascent"],
            "Lotus",
        ),
        (
            "unnamed veto entry falls back to the strip",
            "Ascent",
            &["", ""],
            "Ascent",
        ),
    ];
    for (name, strip, order, want_map) in cases {
        let (mut tracker, _) = test_tracker();
        tracker.recognizer = Some(Box::new(StubRecognizer::new(strip)));
        tracker.map_order = Some(Box::new(FakeMapOrder::new(&[("match", order)])));
        let frame = zero_zero_frame();
        tracker.step(5, &frame);
        tracker.step(10, &frame);
        let run = tracker
            .active_run()
            .unwrap_or_else(|| panic!("{name}: no map"));
        assert_eq!(run.name, want_map, "{name}");
        assert_eq!(run.map_number, 1, "{name}");
        assert_eq!(
            tracker.maps.len(),
            1,
            "{name}: the veto start must not repeat"
        );
    }
}

#[test]
fn test_strip_fallback_needs_repeated_name_for_veto_candidate() {
    let (mut tracker, _) = test_tracker();
    tracker.map_order = Some(Box::new(FakeMapOrder::new(&[("match", &["", ""])])));
    let frame = zero_zero_frame();
    tracker.recognizer = Some(Box::new(StubRecognizer::new("Lotus")));
    tracker.step(5, &frame);
    tracker.recognizer = Some(Box::new(StubRecognizer::new("Ascent")));
    tracker.step(10, &frame);
    assert!(
        tracker.active_run().is_none(),
        "map started after mismatched fallback names"
    );
    tracker.step(15, &frame);
    assert_eq!(
        tracker.active_run().map(|run| run.name.as_str()),
        Some("Ascent"),
        "want Ascent after its second strip read"
    );
}

// A scheduled 0-0 observation must be followed by another agreeing one.
#[test]
fn test_interrupted_start_observation_restarts_confirmation() {
    let (mut tracker, _) = test_tracker();
    let recognizer = FixedPairRecognizer::new(Score::default());
    tracker.recognizer = Some(Box::new(recognizer.clone()));
    tracker.map_order = Some(Box::new(FakeMapOrder::new(&[("match", &["Lotus"])])));
    let frame = zero_zero_frame();

    tracker.step(5, &frame);
    recognizer.get().pair = Score { left: 1, right: 0 };
    tracker.step(10, &frame);
    recognizer.get().pair = Score::default();
    tracker.step(15, &frame);
    assert!(
        tracker.active_run().is_none(),
        "map started from an interrupted 0-0, 1-0, 0-0 sequence"
    );
    tracker.step(20, &frame);
    assert_eq!(
        tracker.active_run().map(|run| run.name.as_str()),
        Some("Lotus"),
        "want Lotus after two agreeing observations"
    );
}

#[test]
fn test_new_match_can_start_same_map_number_without_closing_previous() {
    let (mut tracker, riot) = test_tracker();
    {
        let mut state = riot.get();
        state.live = true;
        state.event = true;
    }
    let previous = MapRun {
        name: "Lotus".into(),
        match_id: "match".into(),
        map_number: 1,
        left_code: "LOUD".into(),
        right_code: "EDG".into(),
        score: Score {
            left: 12,
            right: 11,
        },
        ..Default::default()
    };
    set_active(&mut tracker, previous);
    tracker.map_order = Some(Box::new(FakeMapOrder::new(&[("next-match", &["Ascent"])])));
    tracker.recognizer = Some(Box::new(StubRecognizer::new("")));
    tracker.veto_candidate_num = 1;
    tracker.veto_candidate_n = 1;
    tracker.pending_since = Some(2);
    tracker.last_gameplay_full = Some(1);
    tracker.pause = Some(Pause {
        kind: "tech_pause".into(),
        reason: String::new(),
        since: 2,
    });
    let publisher = FakePublisher::new(true);
    tracker.publish = Some(Box::new(publisher.clone()));
    let events = capture(&mut tracker);
    tracker.last_match_id = "match".into();
    riot.get().match_id = "next-match".into();
    let frame = zero_zero_frame();

    tracker.step(5, &frame);
    {
        let run = tracker.active_run().unwrap();
        assert_eq!(run.name, "Lotus", "the previous run stays active");
        assert_eq!(run.match_id, "match");
        assert!(tracker.pending_since.is_none());
        // The new match's 0-0 is below the previous run's score, so it is not that
        // run's HUD returning.
        assert_eq!(tracker.last_gameplay_full, None);
    }
    assert_eq!(
        publisher.get().scores.len(),
        0,
        "no publications for the previous match's run"
    );
    assert!(tracker.pause.is_none(), "the previous match's pause reset");
    {
        let events = events.lock().unwrap();
        let found = events
            .iter()
            .any(|event| event.kind == "pause_end" && event.source == "match_change");
        assert!(found, "want the previous match's pause ended: {events:?}");
    }
    tracker.step(10, &frame);
    {
        let run = tracker.active_run().unwrap();
        assert_eq!(tracker.maps.len(), 2);
        assert_eq!(run.match_id, "next-match");
        assert_eq!(run.map_number, 1);
    }
    {
        let state = publisher.get();
        assert_eq!(
            state.scores.len(),
            1,
            "want the new run's 0-0 published once"
        );
        assert_eq!(state.scores[0].score["LOUD"], 0);
        assert_eq!(state.scores[0].score["EDG"], 0);
    }
    assert!(!tracker.maps[0].ended, "previous map must stay open");
}

#[test]
fn test_strip_only_starts_game_one_for_new_match() {
    let (mut tracker, riot) = test_tracker();
    let previous = MapRun {
        name: "Lotus".into(),
        match_id: "match".into(),
        map_number: 1,
        left_code: "LOUD".into(),
        right_code: "EDG".into(),
        ended: true,
        ..Default::default()
    };
    set_active(&mut tracker, previous);
    tracker.last_match_id = "match".into();
    riot.get().match_id = "next-match".into();
    tracker.recognizer = Some(Box::new(StubRecognizer::new("Ascent")));
    let frame = zero_zero_frame();
    tracker.step(5, &frame);
    assert_eq!(
        tracker.maps.len(),
        1,
        "strip-only start still needs a repeated read"
    );
    tracker.step(10, &frame);
    assert_eq!(tracker.maps.len(), 2);
    let run = tracker.active_run().unwrap();
    assert_eq!(run.match_id, "next-match");
    assert_eq!(run.map_number, 1);
}

#[test]
fn test_next_map_number_advances_past_highest_tracked_game() {
    let cases: [(&str, Vec<MapRun>, i64); 2] = [
        (
            "contiguous unnamed games",
            vec![
                MapRun {
                    name: String::new(),
                    match_id: "match".into(),
                    map_number: 1,
                    ..Default::default()
                },
                MapRun {
                    name: String::new(),
                    match_id: "match".into(),
                    map_number: 2,
                    ..Default::default()
                },
                MapRun {
                    name: "Lotus".into(),
                    match_id: "other-match".into(),
                    map_number: 1,
                    ..Default::default()
                },
            ],
            3,
        ),
        (
            // Joining mid-series must not reuse the current game's number.
            "joined at game two",
            vec![MapRun {
                name: "Ascent".into(),
                match_id: "match".into(),
                map_number: 2,
                ..Default::default()
            }],
            3,
        ),
    ];
    for (name, maps, want) in cases {
        let (mut tracker, _) = test_tracker();
        tracker.maps = maps;
        assert_eq!(tracker.next_map_number("", "match"), want, "{name}");
    }
}

#[test]
fn test_riot_game_number_beats_a_repeated_strip_name() {
    let (mut tracker, riot) = test_tracker();
    {
        let mut state = riot.get();
        state.game = 2;
        state.states = BTreeMap::from([(2, "inProgress".to_string())]);
    }
    tracker.recognizer = Some(Box::new(StubRecognizer::new("Lotus")));
    tracker.map_order = Some(Box::new(FakeMapOrder::new(&[(
        "match",
        &["Lotus", "Ascent"],
    )])));
    set_active(
        &mut tracker,
        MapRun {
            name: "Lotus".into(),
            match_id: "match".into(),
            map_number: 1,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            score: Score { left: 5, right: 4 },
            strip_at_start: "Lotus".into(),
            ..Default::default()
        },
    );
    let frame = zero_zero_frame();
    tracker.step(5, &frame);
    tracker.step(10, &frame);

    assert_eq!(tracker.maps.len(), 2, "want Riot-confirmed game two");
    let run = tracker.active_run().unwrap();
    assert_eq!(run.map_number, 2);
    assert_eq!(run.name, "Ascent");
    assert_eq!(run.strip_at_start, "Lotus");
}

// Game Changers 2026: Riot listed game N in progress for 2-5 minutes after map N+1's
// 0-0 was on screen. The old gate saw no new game number, so map N+1 was never
// started: a match-point map was resynced to the next map's 1-0 (G2-MIBR, 120), and
// after an ended map the next one went untracked and every later map was numbered one
// low (SRG-KRU 168, M8-KRU 174). A changed strip must start game N+1 on its own sides,
// closing a match-point map as a next_map end first.
#[test]
fn test_changed_strip_starts_the_next_game_while_riot_lags() {
    let cases = [
        (
            "match point",
            Score {
                left: 11,
                right: 12,
            },
            false,
        ),
        ("ended", Score { left: 13, right: 5 }, true),
    ];
    for (name, score, ended) in cases {
        let (mut tracker, riot) = test_tracker();
        {
            let mut state = riot.get();
            state.game = 1;
            state.states = BTreeMap::from([(1, "inProgress".to_string())]);
            state.sides.insert(
                2,
                Sides {
                    blue: "EDG".into(),
                    red: "LOUD".into(),
                },
            );
        }
        tracker.recognizer = Some(Box::new(StubRecognizer::new("Lotus")));
        set_active(
            &mut tracker,
            MapRun {
                name: "Ascent".into(),
                match_id: "match".into(),
                map_number: 1,
                left_code: "LOUD".into(),
                right_code: "EDG".into(),
                score,
                ended,
                strip_at_start: "Ascent".into(),
                ..Default::default()
            },
        );
        let captured = capture(&mut tracker);
        let frame = zero_zero_frame();
        tracker.step(5, &frame);
        tracker.step(10, &frame);

        let run = tracker.active_run().unwrap();
        assert_eq!(
            (run.name.as_str(), run.map_number),
            ("Lotus", 2),
            "{name}: the next map never started"
        );
        assert_eq!(
            (run.left_code.as_str(), run.right_code.as_str()),
            ("EDG", "LOUD"),
            "{name}: the next map took the previous game's sides"
        );
        let events = captured.lock().unwrap();
        assert!(
            !events.iter().any(|event| event.kind == "resync"),
            "{name}: {events:?}"
        );
        if !ended {
            let end = events
                .iter()
                .find(|event| event.kind == "map_end")
                .expect("the match-point map never ended");
            // Riot has not completed game 1, so the leader is a guess: no final round.
            assert_eq!(
                (
                    end.map_number,
                    end.winner.as_str(),
                    end.source.as_str(),
                    end.exact
                ),
                (1, "EDG", "next_map", Some(false))
            );
            assert_eq!(
                end.score,
                Some(ScoreMap::from([
                    ("LOUD".to_string(), 11),
                    ("EDG".to_string(), 12),
                ]))
            );
            let end_score = end.score.clone();
            drop(events);
            // M8-KRU 174 Sunset shape: the leader on 12 lost the map. Riot's late credit
            // for game 1 corrects the guess and moves the series credit.
            {
                let mut state = riot.get();
                state.states =
                    BTreeMap::from([(1, "completed".to_string()), (2, "inProgress".to_string())]);
                state.wins = ScoreMap::from([("LOUD".to_string(), 1)]);
            }
            tracker.step(15, &frame);
            let events = captured.lock().unwrap();
            let corrected = events.last().expect("no correction");
            assert_eq!(
                (
                    corrected.kind.as_str(),
                    corrected.map_number,
                    corrected.winner.as_str(),
                    corrected.source.as_str(),
                    corrected.exact
                ),
                ("map_end", 1, "LOUD", "riot", Some(false)),
                "Riot's winner did not correct the guess: {events:?}"
            );
            assert_eq!(
                corrected.score, end_score,
                "the correction changed the score"
            );
            assert_eq!(
                (
                    tracker.series_wins["match"].get("LOUD"),
                    tracker.series_wins["match"].get("EDG")
                ),
                (Some(&1), Some(&0)),
                "the series credit did not move"
            );
        }
    }
}

// KRU-GEN 2026-10-23 (116): with map 3 Abyss just started, Riot flipped back to
// listing game 2 in progress, and the same 0-0 restarted Abyss as map 2 on game 2's
// sides. A start must never go back to a game the match has already tracked.
#[test]
fn test_backwards_riot_game_cannot_reopen_an_earlier_map() {
    let (mut tracker, riot) = test_tracker();
    {
        let mut state = riot.get();
        state.game = 2;
        state.states = BTreeMap::from([(2, "inProgress".to_string())]);
    }
    tracker.recognizer = Some(Box::new(StubRecognizer::new("Abyss")));
    set_active(
        &mut tracker,
        MapRun {
            name: "Abyss".into(),
            match_id: "match".into(),
            map_number: 3,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            strip_at_start: "Abyss".into(),
            ..Default::default()
        },
    );
    let frame = zero_zero_frame();
    for second in (5..=15).step_by(5) {
        tracker.step(second, &frame);
    }
    assert_eq!(
        tracker.maps.len(),
        1,
        "Riot's flap back to game 2 reopened it"
    );
    assert_eq!(tracker.active_run().unwrap().map_number, 3);

    // Nor does an earlier map's strip on screen (a replay or recap of map 1) end a
    // match-point map 2 and start map 3, not even once the recap's 0-0 and 1-0 pass the
    // contender's gates, and it does not resync map 2 down to the recap's score.
    let (mut tracker, riot) = test_tracker();
    {
        let mut state = riot.get();
        state.game = 2;
        state.states = BTreeMap::from([(2, "inProgress".to_string())]);
    }
    tracker.map_order = Some(Box::new(FakeMapOrder::new(&[(
        "match",
        &["Ascent", "Lotus", "Split"],
    )])));
    tracker.recognizer = Some(Box::new(StubRecognizer::new("Ascent")));
    let index = set_active(
        &mut tracker,
        MapRun {
            name: "Lotus".into(),
            match_id: "match".into(),
            map_number: 2,
            left_code: "LOUD".into(),
            right_code: "EDG".into(),
            score: Score { left: 12, right: 9 },
            strip_at_start: "Lotus".into(),
            last_hud_second: Some(0),
            last_known_second: Some(0),
            ..Default::default()
        },
    );
    // The recap's 0-0 shows at 5-9 (one strip read, on the candidate's first read),
    // the HUD hides at 10-50, and 1-0 shows at 51-66, past the contender's gates.
    for second in 5..=9 {
        tracker.step(second, &frame);
    }
    let hidden = Frame::from_rgb(1280, 720, vec![0; 1280 * 720 * 3]).unwrap();
    for second in 10..=50 {
        tracker.step(second, &hidden);
    }
    for second in 51..=66 {
        tracker.record_score(index, Score { left: 1, right: 0 }, second, false);
    }
    let run = tracker.active_run().unwrap();
    assert_eq!(
        (tracker.maps.len(), run.map_number, run.ended, scores(run)),
        (1, 2, false, (12, 9)),
        "map 1's recap closed, resynced or replaced Lotus"
    );
}

// A cut can hide two rounds, not just the map-ending one: a run at 11-11 whose map
// really ended 13-11 is not at match point (G2-MIBR 120 shape: Ascent, then Lotus while
// Riot still lists game 1). A 0-0 whose strip changed to the veto's next map closes the
// map with no winner, since none is known, and starts the next one instead of letting
// its opening resync map 1. Mid-map, the same reads start nothing.
#[test]
fn test_late_cut_closes_the_map_without_a_winner_and_starts_the_next_veto_map() {
    // `credited`: Riot completes game 1 for G2 before map 2's 0-0, so map 2's baseline
    // already holds that win and only map 1's own baseline can name it.
    for (score, late, credited) in [
        (
            Score {
                left: 11,
                right: 11,
            },
            true,
            false,
        ),
        (
            Score {
                left: 11,
                right: 11,
            },
            true,
            true,
        ),
        // The leader on 12 lost the map, and Riot already completed it: Riot's winner,
        // not the leader, is credited.
        (
            Score {
                left: 11,
                right: 12,
            },
            true,
            true,
        ),
        (Score { left: 6, right: 5 }, false, false),
    ] {
        let (mut tracker, riot) = test_tracker();
        {
            let mut state = riot.get();
            state.game = 1;
            state.states = BTreeMap::from([(1, "inProgress".to_string())]);
            if credited {
                state.states.insert(1, "completed".to_string());
                state.wins = ScoreMap::from([("G2".to_string(), 1), ("MIBR".to_string(), 0)]);
            }
            state.sides.insert(
                2,
                Sides {
                    blue: "MIBR".into(),
                    red: "G2".into(),
                },
            );
        }
        tracker.recognizer = Some(Box::new(StubRecognizer::new("Lotus")));
        tracker.map_order = Some(Box::new(FakeMapOrder::new(&[(
            "match",
            &["Ascent", "Lotus", "Split"],
        )])));
        set_active(
            &mut tracker,
            MapRun {
                name: "Ascent".into(),
                match_id: "match".into(),
                map_number: 1,
                left_code: "G2".into(),
                right_code: "MIBR".into(),
                score,
                strip_at_start: "Ascent".into(),
                ..Default::default()
            },
        );
        let events = capture(&mut tracker);
        for second in [5, 10] {
            tracker.step(second, &zero_zero_frame());
        }

        let events = events.lock().unwrap();
        let ends: Vec<_> = events
            .iter()
            .filter(|event| event.kind == "map_end")
            .collect();
        if !late {
            assert_eq!(tracker.maps.len(), 1, "a mid-map strip started a map");
            assert!(ends.is_empty(), "{events:?}");
            continue;
        }
        let run = tracker.active_run().unwrap();
        assert_eq!(
            (run.name.as_str(), run.map_number, run.left_code.as_str()),
            ("Lotus", 2, "MIBR")
        );
        assert_eq!(ends.len(), 1, "{events:?}");
        assert_eq!(
            (
                ends[0].map_number,
                ends[0].winner.as_str(),
                ends[0].source.as_str(),
                ends[0].exact
            ),
            (1, if credited { "G2" } else { "" }, "next_map", Some(false))
        );
        assert_eq!(
            ends[0].score,
            Some(ScoreMap::from([
                ("G2".to_string(), score.left),
                ("MIBR".to_string(), score.right)
            ]))
        );
        assert!(!events.iter().any(|event| event.kind == "resync"));
        if credited {
            drop(events);
            tracker.step(15, &zero_zero_frame());
            assert_eq!(
                tracker.series_wins["match"].get("G2"),
                Some(&1),
                "map 1's winner was not credited exactly once"
            );
        }
    }
}

// MIBR-M8 136: Ascent was never tracked, and Haven's 0-0 opened as map 1 on game 1's
// sides because Riot still listed game 1. The strip's place in the veto numbers the
// game without Riot.
#[test]
fn test_veto_order_numbers_an_untracked_series_while_riot_lags() {
    let (mut tracker, riot) = test_tracker();
    {
        let mut state = riot.get();
        state.game = 1;
        state.states = BTreeMap::from([(1, "inProgress".to_string())]);
        state.sides.insert(
            1,
            Sides {
                blue: "M8".into(),
                red: "MIBR".into(),
            },
        );
        state.sides.insert(
            2,
            Sides {
                blue: "MIBR".into(),
                red: "M8".into(),
            },
        );
    }
    tracker.recognizer = Some(Box::new(StubRecognizer::new("Haven")));
    tracker.map_order = Some(Box::new(FakeMapOrder::new(&[(
        "match",
        &["Ascent", "Haven", "Abyss"],
    )])));
    for second in [5, 10] {
        tracker.step(second, &zero_zero_frame());
    }
    let run = tracker.active_run().expect("Haven never started");
    assert_eq!(
        (run.name.as_str(), run.map_number, run.left_code.as_str()),
        ("Haven", 2, "MIBR")
    );
}

// KRU-GEN 116: GEN won Sunset, KRU won Ascent. Riot credits game 1 minutes after
// Ascent's 0-0, so Ascent's series baseline missed GEN's win and its own end read two
// gainers and no winner. Riot's later credit must name map 1's missing winner and
// leave map 2's win to KRU.
#[test]
fn test_late_riot_credit_names_the_previous_map_and_rebaselines_the_next() {
    let (mut tracker, riot) = test_tracker();
    {
        let mut state = riot.get();
        state.game = 1;
        state.states = BTreeMap::from([(1, "inProgress".to_string())]);
        state.wins = ScoreMap::from([("KRU".to_string(), 0), ("GEN".to_string(), 0)]);
        state.sides.insert(
            2,
            Sides {
                blue: "KRU".into(),
                red: "GEN".into(),
            },
        );
    }
    tracker.recognizer = Some(Box::new(StubRecognizer::new("Ascent")));
    tracker.map_order = Some(Box::new(FakeMapOrder::new(&[(
        "match",
        &["Sunset", "Ascent", "Abyss"],
    )])));
    set_active(
        &mut tracker,
        MapRun {
            name: "Sunset".into(),
            match_id: "match".into(),
            map_number: 1,
            left_code: "KRU".into(),
            right_code: "GEN".into(),
            score: Score {
                left: 10,
                right: 11,
            },
            strip_at_start: "Sunset".into(),
            ..Default::default()
        },
    );
    let events = capture(&mut tracker);
    for second in [5, 10] {
        tracker.step(second, &zero_zero_frame());
    }
    {
        let mut state = riot.get();
        state.game = 2;
        state.states =
            BTreeMap::from([(1, "completed".to_string()), (2, "inProgress".to_string())]);
        state.wins = ScoreMap::from([("KRU".to_string(), 0), ("GEN".to_string(), 1)]);
    }
    tracker.step(15, &zero_zero_frame());

    update_active(&mut tracker, |run| run.score = Score { left: 12, right: 7 });
    tracker.last_gameplay_full = Some(20);
    tracker.check_pending_end(40, Scene::Replay, false, None);
    {
        let mut state = riot.get();
        state.states.insert(2, "completed".to_string());
        state.wins = ScoreMap::from([("KRU".to_string(), 1), ("GEN".to_string(), 1)]);
    }
    tracker.check_pending_end(41, Scene::Replay, false, None);

    let events = events.lock().unwrap();
    let ends: Vec<(i64, &str, &str)> = events
        .iter()
        .filter(|event| event.kind == "map_end")
        .map(|event| {
            (
                event.map_number,
                event.winner.as_str(),
                event.source.as_str(),
            )
        })
        .collect();
    assert_eq!(
        ends,
        vec![(1, "", "next_map"), (1, "GEN", "riot"), (2, "KRU", "riot")],
        "{events:?}"
    );
    assert_eq!(
        tracker.series_wins["match"],
        ScoreMap::from([("KRU".to_string(), 1), ("GEN".to_string(), 1)])
    );
}

#[test]
fn test_strip_names_map_without_veto() {
    let (mut tracker, _) = test_tracker();
    tracker.recognizer = Some(Box::new(StubRecognizer::new("Lotus")));
    let frame = zero_zero_frame();
    tracker.step(5, &frame);
    assert!(
        tracker.active_run().is_none(),
        "a single strip read must not start the map"
    );
    tracker.step(10, &frame);
    assert_eq!(
        tracker.active_run().map(|run| run.name.as_str()),
        Some("Lotus")
    );
    assert_eq!(tracker.maps.len(), 1);
}

// The 2 Oct 2026 JDG-T1 incident: Riot alone reported a game in progress while no
// source named the map.
#[test]
fn test_riot_in_progress_alone_does_not_start_an_unnamed_map() {
    let (mut tracker, riot) = test_tracker();
    {
        let mut state = riot.get();
        state.game = 3;
        state.states = BTreeMap::from([(3, "inProgress".to_string())]);
    }
    tracker.recognizer = Some(Box::new(StubRecognizer::new("")));
    let frame = zero_zero_frame();
    for second in (5..=15).step_by(5) {
        tracker.step(second, &frame);
    }
    assert!(
        tracker.active_run().is_none(),
        "Riot in progress alone started a run"
    );
    assert_eq!(tracker.maps.len(), 0);
    tracker.recognizer = Some(Box::new(StubRecognizer::new("Lotus")));
    tracker.step(20, &frame);
    tracker.step(25, &frame);
    let run = tracker.active_run().expect("active map");
    assert_eq!(run.name, "Lotus");
    assert_eq!(run.map_number, 3);
}

// A deploy, a reconnect or a late join leaves the tracker mid-map. Once 60 s of
// confirmed reads corroborate the on-screen score while Riot lists the game and this
// match's plates, the run opens at that score, and a pause found afterwards must
// be published: publishing is run-gated, so before the join it was never sent.
#[test]
fn test_mid_map_join_adopts_a_corroborated_score_and_publishes_its_pause() {
    let (mut tracker, riot) = test_tracker();
    {
        let mut state = riot.get();
        state.live = true;
        state.event = true;
        state.game = 2;
        state.teams = vec![
            Team {
                id: String::new(),
                code: "TL".into(),
                name: String::new(),
            },
            Team {
                id: String::new(),
                code: "G2".into(),
                name: String::new(),
            },
        ];
        // Riot's sides match the frame's plates: TL on the left, G2 on the right.
        state.sides = BTreeMap::from([(
            2,
            Sides {
                blue: "TL".into(),
                red: "G2".into(),
            },
        )]);
        state.states = BTreeMap::from([(2, "inProgress".to_string())]);
    }
    tracker.map_order = Some(Box::new(FakeMapOrder::new(&[(
        "match",
        &["Split", "Abyss"],
    )])));
    tracker.plates = Some(learned_plates());
    let recognizer = FixedPairRecognizer::new(Score { left: 3, right: 1 });
    // The strip must name the game and agree with the veto; the veto's spelling
    // wins for the run's name.
    recognizer.get().map = "ABYSS".to_string();
    tracker.recognizer = Some(Box::new(recognizer.clone()));
    let publisher = FakePublisher::new(true);
    tracker.publish = Some(Box::new(publisher.clone()));
    let events = capture(&mut tracker);
    let frame = load_teamplate_frame("abyss-round5.png");

    // Twelve corroborating reads must not adopt: the series also needs 60 s of play.
    for second in 0..=59 {
        tracker.step(second, &frame);
    }
    assert!(
        tracker.active_run().is_none(),
        "joined before 60 s of confirmed reads"
    );
    assert!(publisher.get().scores.is_empty(), "published without a run");

    tracker.step(60, &frame);
    let run = active(&tracker);
    assert_eq!(run.map_number, 2);
    assert_eq!(run.name, "Abyss");
    assert_eq!(scores(run), (3, 1));
    {
        let events = events.lock().unwrap();
        assert!(
            events
                .iter()
                .any(|event| event.kind == "map_join" && event.second == 60),
            "want a map_join event at 60: {events:?}"
        );
    }
    {
        let state = publisher.get();
        let last = state.scores.last().expect("the joined score");
        assert_eq!(last.status, Status::Ok);
        assert_eq!(last.blue, "TL");
        assert_eq!(last.map_number, 2);
        assert_eq!(
            last.score,
            ScoreMap::from([("TL".to_string(), 3), ("G2".to_string(), 1)])
        );
    }

    // The regression: a pause detected after the join must be published with it.
    {
        let mut state = recognizer.get();
        state.on = true;
        state.pause = PauseRead {
            kind: "halftime".into(),
            reason: String::new(),
            score: 0.9,
        };
    }
    tracker.step(61, &frame);
    tracker.step(62, &frame);
    let state = publisher.get();
    let last = state.scores.last().expect("the paused publication");
    assert_eq!(last.pause, "halftime", "pause did not reach the publisher");
}

// A mid-map clip is not this match's evidence: plates from another match, or a paused
// broadcast, must never open a run at the score they show.
#[test]
fn test_mid_map_join_never_adopts_another_matchs_score_or_a_paused_read() {
    // Part (a): the frame's plates (TL/G2) are not the match Riot lists (LOUD/EDG),
    // so no read ever corroborates.
    {
        let (mut tracker, riot) = test_tracker();
        {
            let mut state = riot.get();
            state.live = true;
            state.event = true;
            state.game = 2;
            state.states = BTreeMap::from([(2, "inProgress".to_string())]);
            state.sides = BTreeMap::from([(
                2,
                Sides {
                    blue: "LOUD".into(),
                    red: "EDG".into(),
                },
            )]);
        }
        tracker.map_order = Some(Box::new(FakeMapOrder::new(&[(
            "match",
            &["Split", "Abyss"],
        )])));
        tracker.plates = Some(learned_plates());
        let recognizer = FixedPairRecognizer::new(Score { left: 3, right: 1 });
        recognizer.get().map = "Abyss".to_string();
        tracker.recognizer = Some(Box::new(recognizer));
        let publisher = FakePublisher::new(true);
        tracker.publish = Some(Box::new(publisher.clone()));
        let frame = load_teamplate_frame("abyss-round5.png");
        for second in 0..=300 {
            tracker.step(second, &frame);
        }
        assert!(
            tracker.active_run().is_none(),
            "joined another match's clip"
        );
        assert!(
            publisher.get().scores.is_empty(),
            "published another match's clip"
        );
    }

    // Part (b): a confirmed pause overlaps the whole window, so no series survives
    // it. Adoption can only start 60 s of unpaused, confirmed reads after the clear.
    let (mut tracker, riot) = test_tracker();
    {
        let mut state = riot.get();
        state.live = true;
        state.event = true;
        state.game = 2;
        state.teams = vec![
            Team {
                id: String::new(),
                code: "TL".into(),
                name: String::new(),
            },
            Team {
                id: String::new(),
                code: "G2".into(),
                name: String::new(),
            },
        ];
        state.sides = BTreeMap::from([(
            2,
            Sides {
                blue: "TL".into(),
                red: "G2".into(),
            },
        )]);
        state.states = BTreeMap::from([(2, "inProgress".to_string())]);
    }
    tracker.map_order = Some(Box::new(FakeMapOrder::new(&[(
        "match",
        &["Split", "Abyss"],
    )])));
    tracker.plates = Some(learned_plates());
    let recognizer = FixedPairRecognizer::new(Score { left: 3, right: 1 });
    recognizer.get().map = "Abyss".to_string();
    {
        let mut state = recognizer.get();
        state.on = true;
        state.pause = PauseRead {
            kind: "tech_pause".into(),
            reason: String::new(),
            score: 0.9,
        };
    }
    tracker.recognizer = Some(Box::new(recognizer.clone()));
    let publisher = FakePublisher::new(true);
    tracker.publish = Some(Box::new(publisher.clone()));
    let events = capture(&mut tracker);
    let frame = load_teamplate_frame("abyss-round5.png");

    // The pause covers every scheduled read.
    for second in 0..=300 {
        tracker.step(second, &frame);
    }
    assert!(
        tracker.active_run().is_none(),
        "joined from a paused broadcast"
    );
    assert!(publisher.get().scores.is_empty());

    // Two unmatched frames with the HUD visible end the pause at 302.
    recognizer.get().on = false;
    tracker.step(301, &frame);
    tracker.step(302, &frame);
    assert!(tracker.pause.is_none(), "pause did not clear");

    // The 12 reads up to 360 corroborate but stop short of 60 s from the first one.
    for second in 303..=361 {
        tracker.step(second, &frame);
    }
    assert!(
        tracker.active_run().is_none(),
        "joined before clear + 60 s of unpaused reads"
    );
    tracker.step(365, &frame);
    assert_eq!(
        tracker
            .active_run()
            .map(|run| (run.name.as_str(), scores(run))),
        Some(("Abyss", (3, 1)))
    );
    let events = events.lock().unwrap();
    let adopted: Vec<i64> = events
        .iter()
        .filter(|event| event.kind == "map_join")
        .map(|event| event.second)
        .collect();
    assert_eq!(
        adopted,
        vec![365],
        "want exactly one join, after the pause's clear + 60 s: {events:?}"
    );
    drop(events);

    // A fresh tracker whose frame plates sit in Riot's order for game 2 and whose
    // strip names `strip`, so only the case under test can block a join.
    let setup = |strip: &str| {
        let (mut tracker, riot) = test_tracker();
        {
            let mut state = riot.get();
            state.live = true;
            state.event = true;
            state.game = 2;
            state.teams = vec![
                Team {
                    id: String::new(),
                    code: "TL".into(),
                    name: String::new(),
                },
                Team {
                    id: String::new(),
                    code: "G2".into(),
                    name: String::new(),
                },
            ];
            state.sides = BTreeMap::from([(
                2,
                Sides {
                    blue: "TL".into(),
                    red: "G2".into(),
                },
            )]);
            state.states = BTreeMap::from([(2, "inProgress".to_string())]);
        }
        tracker.map_order = Some(Box::new(FakeMapOrder::new(&[(
            "match",
            &["Split", "Abyss"],
        )])));
        tracker.plates = Some(learned_plates());
        let recognizer = FixedPairRecognizer::new(Score { left: 3, right: 1 });
        recognizer.get().map = strip.to_string();
        tracker.recognizer = Some(Box::new(recognizer.clone()));
        let publisher = FakePublisher::new(true);
        tracker.publish = Some(Box::new(publisher.clone()));
        (tracker, recognizer, publisher)
    };

    // Part (c): the strip is the only independent map-identity evidence, so a strip
    // that disagrees with the veto resets the series even when Riot's game, sides,
    // score and this match's plates all agree.
    {
        let (mut tracker, _, publisher) = setup("ASCENT");
        let events = capture(&mut tracker);
        let frame = load_teamplate_frame("abyss-round5.png");
        for second in 0..=300 {
            tracker.step(second, &frame);
        }
        assert!(
            tracker.active_run().is_none(),
            "joined on a strip that contradicts the veto"
        );
        assert!(publisher.get().scores.is_empty());
        let events = events.lock().unwrap();
        assert!(
            !events.iter().any(|event| event.kind == "map_join"),
            "a strip/veto disagreement must not join: {events:?}"
        );
    }

    // Part (d): a short pause between two scheduled reads must reset the series. The
    // pre-pause series would otherwise have adopted on the 60 s read just after the
    // clear; with the reset, adoption waits for a fresh 60 s window.
    {
        let (mut tracker, recognizer, _) = setup("Abyss");
        let events = capture(&mut tracker);
        let frame = load_teamplate_frame("abyss-round5.png");
        for second in 0..=55 {
            tracker.step(second, &frame);
        }
        {
            let mut state = recognizer.get();
            state.on = true;
            state.pause = PauseRead {
                kind: "tech_pause".into(),
                reason: String::new(),
                score: 0.9,
            };
        }
        tracker.step(56, &frame);
        tracker.step(57, &frame);
        assert!(tracker.pause.is_some(), "pause confirmed at 57");
        recognizer.get().on = false;
        tracker.step(58, &frame);
        tracker.step(59, &frame);
        assert!(tracker.pause.is_none(), "pause cleared at 59");
        tracker.step(60, &frame);
        assert!(
            tracker.active_run().is_none(),
            "a pre-pause series joined on the first read after the pause cleared"
        );
        for second in 61..=119 {
            tracker.step(second, &frame);
        }
        assert!(
            tracker.active_run().is_none(),
            "joined before a fresh 60 s of unpaused reads"
        );
        tracker.step(120, &frame);
        assert_eq!(
            tracker
                .active_run()
                .map(|run| (run.name.as_str(), scores(run))),
            Some(("Abyss", (3, 1)))
        );
        let events = events.lock().unwrap();
        let adopted: Vec<i64> = events
            .iter()
            .filter(|event| event.kind == "map_join")
            .map(|event| event.second)
            .collect();
        assert_eq!(
            adopted,
            vec![120],
            "want exactly one join, from a fresh window: {events:?}"
        );
    }

    // Part (f): a plate the crops cannot read is no evidence either way, so it keeps the
    // series; before an OCR has taught a team's crop most plates are unread, and a reset
    // on each would leave the join waiting for 8 readable reads in a row. A plate read on
    // the wrong side is another match's evidence and still resets the series.
    {
        let frame = load_teamplate_frame("abyss-round5.png");
        let unread = paste_plates(&frame, None, None);
        let swapped = paste_plates(
            &frame,
            Some(&frame.crop(PLATE_RIGHT)),
            Some(&frame.crop(PLATE_LEFT)),
        );
        for (odd, joins_at) in [(&unread, 60), (&swapped, 95)] {
            let (mut tracker, _, _) = setup("Abyss");
            for second in 0..=joins_at {
                let shown = if second == 30 { odd } else { &frame };
                tracker.step(second, shown);
                if second < joins_at {
                    assert!(tracker.active_run().is_none(), "joined at {second}");
                }
            }
            assert_eq!(
                tracker.active_run().map(scores),
                Some((3, 1)),
                "want a join at {joins_at}"
            );
        }
    }

    // Part (e): the first overlay frame is itself pause evidence. On the boundary
    // second 60 it must block adoption even though the pause is only confirmed on
    // the next frame.
    {
        let (mut tracker, recognizer, _) = setup("Abyss");
        let events = capture(&mut tracker);
        let frame = load_teamplate_frame("abyss-round5.png");
        for second in 0..=55 {
            tracker.step(second, &frame);
        }
        {
            let mut state = recognizer.get();
            state.on = true;
            state.pause = PauseRead {
                kind: "tech_pause".into(),
                reason: String::new(),
                score: 0.9,
            };
        }
        tracker.step(60, &frame);
        assert!(
            tracker.pause.is_none(),
            "one overlay frame must not confirm the pause yet"
        );
        assert!(
            tracker.active_run().is_none(),
            "adopted on the first, unconfirmed pause-overlay frame"
        );
        tracker.step(61, &frame);
        assert!(
            tracker.pause.is_some(),
            "the second overlay frame confirms the pause"
        );
        let events = events.lock().unwrap();
        assert!(
            !events.iter().any(|event| event.kind == "map_join"),
            "a boundary pause still allowed a join: {events:?}"
        );
    }
}

// A strip that keeps reading the same wrong name must not advance the series.
#[test]
fn test_veto_does_not_advance_on_persistent_wrong_strip() {
    let (mut tracker, _) = test_tracker();
    tracker.recognizer = Some(Box::new(StubRecognizer::new("Summit")));
    tracker.map_order = Some(Box::new(FakeMapOrder::new(&[(
        "match",
        &["Lotus", "Split"],
    )])));
    let frame = zero_zero_frame();
    for second in (5..=20).step_by(5) {
        tracker.step(second, &frame);
    }
    assert_eq!(
        tracker.maps.len(),
        1,
        "a persistent wrong strip must not advance"
    );
    assert_eq!(
        tracker.active_run().map(|run| run.name.as_str()),
        Some("Lotus")
    );
}

// A numbering fallback is not a game in progress, so it must not corroborate the
// next game.
#[test]
fn test_veto_needs_real_in_progress_for_next_game() {
    let (mut tracker, _) = test_tracker();
    tracker.recognizer = Some(Box::new(StubRecognizer::new("")));
    tracker.map_order = Some(Box::new(FakeMapOrder::new(&[(
        "match",
        &["Lotus", "Split", "Ascent"],
    )])));
    let active_run = MapRun {
        name: "Split".into(),
        match_id: "match".into(),
        map_number: 2,
        left_code: "LOUD".into(),
        right_code: "EDG".into(),
        score: Score { left: 4, right: 4 },
        ..Default::default()
    };
    set_active(&mut tracker, active_run);
    let frame = zero_zero_frame();
    for second in (5..=15).step_by(5) {
        tracker.step(second, &frame);
    }
    assert_eq!(
        tracker.maps.len(),
        1,
        "a fallback game number must not start the next map"
    );
}

// A decided series must not start a phantom decider from post-match 0-0 frames.
#[test]
fn test_veto_refuses_game_after_series_decided() {
    let cases: [(&str, Option<MapRun>, usize); 2] = [
        ("without an active run", None, 0),
        (
            "after a changed strip on an active run",
            Some(MapRun {
                name: "Split".into(),
                match_id: "match".into(),
                map_number: 2,
                left_code: "LOUD".into(),
                right_code: "EDG".into(),
                score: Score { left: 13, right: 5 },
                strip_at_start: "Lotus".into(),
                ..Default::default()
            }),
            1,
        ),
    ];
    for (name, active_run, want_maps) in cases {
        let (mut tracker, riot) = test_tracker();
        {
            let mut state = riot.get();
            state.best_of = 3;
            state.wins = ScoreMap::from([("LOUD".to_string(), 2), ("EDG".to_string(), 0)]);
            state.game = 3;
            state.states = BTreeMap::from([(3, "inProgress".to_string())]);
        }
        tracker.recognizer = Some(Box::new(StubRecognizer::new("Ascent")));
        tracker.map_order = Some(Box::new(FakeMapOrder::new(&[(
            "match",
            &["Lotus", "Split", "Ascent"],
        )])));
        if let Some(active_run) = active_run {
            set_active(&mut tracker, active_run);
        }
        let frame = zero_zero_frame();
        for second in (5..=15).step_by(5) {
            tracker.step(second, &frame);
        }
        assert_eq!(
            tracker.maps.len(),
            want_maps,
            "{name}: a decided series must not start a decider"
        );
    }
}

#[test]
fn test_strip_only_refuses_game_after_series_decided() {
    let (mut tracker, riot) = test_tracker();
    {
        let mut state = riot.get();
        state.best_of = 3;
        state.wins = ScoreMap::from([("LOUD".to_string(), 2), ("EDG".to_string(), 0)]);
    }
    tracker.recognizer = Some(Box::new(StubRecognizer::new("Ascent")));
    let frame = zero_zero_frame();
    for second in (5..=15).step_by(5) {
        tracker.step(second, &frame);
    }
    assert_eq!(
        tracker.maps.len(),
        0,
        "no strip-only start after the series was decided"
    );
}

// An unreadable strip when a map started leaves nothing to compare against.
#[test]
fn test_veto_does_not_advance_from_unreadable_start() {
    let (mut tracker, _) = test_tracker();
    tracker.map_order = Some(Box::new(FakeMapOrder::new(&[(
        "match",
        &["Lotus", "Split"],
    )])));
    tracker.recognizer = Some(Box::new(StubRecognizer::new("")));
    let frame = zero_zero_frame();
    tracker.step(5, &frame);
    tracker.step(10, &frame);
    let run = tracker.active_run().expect("active map");
    assert_eq!(run.name, "Lotus");
    assert_eq!(run.strip_at_start, "", "want an empty baseline");
    tracker.recognizer = Some(Box::new(StubRecognizer::new("Summit")));
    for second in (15..=30).step_by(5) {
        tracker.step(second, &frame);
    }
    assert_eq!(
        tracker.maps.len(),
        1,
        "a name read after an unreadable start is not a transition"
    );
}
