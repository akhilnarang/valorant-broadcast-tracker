//! Plate, map-strip and pause reads on the shared broadcast frames.

mod common;

use common::{frames_dir, load_named};
use vbt_vision::frame::Frame;
use vbt_vision::recognizer::Recognizer;
use vbt_vision::regions::{
    MAP_STRIP, MAP_STRIP_WIDTH, PAUSE_CARD, PAUSE_HEADER, SCORE_LEFT, SCORE_RIGHT,
};

/// The classifier must read the fixture plates at or above the trust floor, and
/// the strip reader must name the map on a gameplay frame and stay silent when the
/// strip is blank (half-time). The tracker's strip read goes through the CTC
/// lexicon reader in `vbt-scoreboard` first, and these reads are its fallback.
#[test]
fn test_plates_and_map_strip() {
    let recognizer = Recognizer::new().expect("load recognizer");
    let summit = load_named(&frames_dir(), "summit-12-4.jpg");
    let left = recognizer.score(&summit.crop(SCORE_LEFT));
    let right = recognizer.score(&summit.crop(SCORE_RIGHT));
    assert_eq!(left, Some(12), "summit left");
    assert_eq!(right, Some(4), "summit right");

    let strip = summit.crop_raw(MAP_STRIP);
    assert_eq!(recognizer.map(&strip), "Summit", "summit strip");

    // The Sunset template scores 0.833 on this Summit strip. The Summit template
    // (0.987) must still clear it by MAP_MARGIN.
    let summit = load_named(&frames_dir(), "summit-5-12.png");
    let strip = summit.crop_raw(MAP_STRIP);
    assert_eq!(recognizer.map(&strip), "Summit", "summit 5-12 strip");

    let ticker = load_named(&frames_dir(), "prx-g2-lotus-ticker.jpg");
    let strip = ticker.crop_raw(MAP_STRIP);
    assert_eq!(recognizer.map(&strip), "Lotus", "live ticker strip");

    // Abyss correlates 0.58-0.72 with a Haven strip, so Haven needs its own
    // template to read.
    let haven = load_named(&frames_dir(), "haven-0-0.jpg");
    let strip = haven.crop_raw(MAP_STRIP);
    assert_eq!(recognizer.map(&strip), "Haven", "haven strip");

    // Without their own templates, Split reads as Abyss (0.55), Sunset as Summit
    // (0.76), and Breeze reads empty because Abyss (0.60) sits inside the
    // runner-up margin.
    let split = load_named(&frames_dir(), "split-0-0.jpg");
    let strip = split.crop_raw(MAP_STRIP);
    assert_eq!(recognizer.map(&strip), "Split", "split strip");

    let sunset = load_named(&frames_dir(), "sunset-0-0.jpg");
    let strip = sunset.crop_raw(MAP_STRIP);
    assert_eq!(recognizer.map(&strip), "Sunset", "sunset strip");

    let breeze = load_named(&frames_dir(), "breeze-0-0.jpg");
    let strip = breeze.crop_raw(MAP_STRIP);
    assert_eq!(recognizer.map(&strip), "Breeze", "breeze strip");

    // Bind, Corrode, Fracture, Icebox and Pearl are outside the current map pool,
    // so their templates come from archived broadcasts. Without them, Bind, Pearl
    // and Icebox read empty, and Corrode and Fracture misread as Breeze (0.72 and
    // 0.70).
    let bind = load_named(&frames_dir(), "bind-0-0.jpg");
    let strip = bind.crop_raw(MAP_STRIP);
    assert_eq!(recognizer.map(&strip), "Bind", "bind strip");

    let corrode = load_named(&frames_dir(), "corrode-0-0.jpg");
    let strip = corrode.crop_raw(MAP_STRIP);
    assert_eq!(recognizer.map(&strip), "Corrode", "corrode strip");

    let fracture = load_named(&frames_dir(), "fracture-0-0.jpg");
    let strip = fracture.crop_raw(MAP_STRIP);
    assert_eq!(recognizer.map(&strip), "Fracture", "fracture strip");

    let pearl = load_named(&frames_dir(), "pearl-0-0.jpg");
    let strip = pearl.crop_raw(MAP_STRIP);
    assert_eq!(recognizer.map(&strip), "Pearl", "pearl strip");

    // The Icebox template and this fixture come from different broadcasts, so the
    // read does not just match a frame against itself.
    let icebox = load_named(&frames_dir(), "icebox-0-0.jpg");
    let strip = icebox.crop_raw(MAP_STRIP);
    assert_eq!(recognizer.map(&strip), "Icebox", "icebox strip");

    let half = load_named(&frames_dir(), "lotus-half-time.jpg");
    let strip = half.crop_raw(MAP_STRIP);
    assert_eq!(recognizer.map(&strip), "", "half-time strip");
}

/// The deciding read can sit on screen for about a second, so the tracker keeps a
/// map-winning gameplay read and adopts it once the score bar is gone. The break
/// footage the broadcast cuts to afterwards — the crowd, the walk-out, the desk, the
/// arena screens, a highlight clip and the FRAG OUT card — trips the timer-box scene
/// check, and two of those frames' plate crops carry bright pixels, so the tracker
/// cannot treat that footage as the HUD returning: no plate of any of them may read
/// a digit, while the real 5-12 frame before the switch must.
#[test]
fn test_intermission_plates_read_no_digits() {
    let recognizer = Recognizer::new().expect("load recognizer");
    let summit = load_named(&frames_dir(), "summit-5-12.png");
    assert_eq!(recognizer.score(&summit.crop(SCORE_LEFT)), Some(5));
    assert_eq!(recognizer.score(&summit.crop(SCORE_RIGHT)), Some(12));

    for name in [
        "falsegame-084008.png",
        "falsegame-084018.png",
        "falsegame-084305.png",
        "falsegame-084425.png",
        "falsegame-084605.png",
        "frag-out-084033.png",
    ] {
        let frame = load_named(&frames_dir(), name);
        assert_eq!(
            (
                recognizer.score(&frame.crop(SCORE_LEFT)),
                recognizer.score(&frame.crop(SCORE_RIGHT)),
            ),
            (None, None),
            "{name} plates"
        );
    }
}

/// The strip reader must stay silent when the two best map matches sit inside the
/// runner-up margin: after the real CURRENT anchor, the name window is an even
/// blend of the Lotus and Abyss templates, so only the margin separates them.
#[test]
fn test_map_strip_rejects_a_name_inside_the_runner_up_margin() {
    let recognizer = Recognizer::new().expect("load recognizer");
    let mut strip = Frame::from_rgb(
        MAP_STRIP_WIDTH as usize,
        18,
        vec![0; MAP_STRIP_WIDTH as usize * 18 * 3],
    )
    .unwrap();
    let paste = |strip: &mut Frame, x: i64, y: i64, tmpl: &[Vec<f64>]| {
        for (dy, row) in tmpl.iter().enumerate() {
            for (dx, v) in row.iter().enumerate() {
                let i = (((y + dy as i64) * strip.w as i64 + x + dx as i64) * 3) as usize;
                let b = v.round() as u8;
                strip.pix[i] = b;
                strip.pix[i + 1] = b;
                strip.pix[i + 2] = b;
            }
        }
    };
    let lotus = recognizer.map_rows("Lotus").expect("Lotus template");
    let abyss = recognizer.map_rows("Abyss").expect("Abyss template");
    let blend: Vec<Vec<f64>> = lotus
        .iter()
        .zip(abyss.iter())
        .map(|(lotus_row, abyss_row)| {
            lotus_row
                .iter()
                .zip(abyss_row.iter())
                .map(|(l, a)| (l + a) / 2.0)
                .collect()
        })
        .collect();
    let current = recognizer.current_rows();
    paste(&mut strip, 6, 2, current);
    paste(&mut strip, 6 + current[0].len() as i64 + 2, 2, &blend);
    assert_eq!(recognizer.map(&strip), "", "ambiguous strip");
}

/// The pause reader must read the overlay phrase at the correlation the detector
/// was measured at, from the lower-third header or the full-screen half-time card,
/// and must stay silent on every frame that is not an overlay.
#[test]
fn test_pause_reads_only_pause_overlays() {
    let recognizer = Recognizer::new().expect("load recognizer");
    let positives = [
        ("tech-pause-2610.jpg", "tech_pause"),
        ("tech-pause-2630.jpg", "tech_pause"),
        ("tech-pause-gear-5445.jpg", "tech_pause"),
        ("timeout-8385.jpg", "timeout"),
        ("halftime-6000.jpg", "halftime"),
        ("lotus-half-time.jpg", "halftime"),
        ("halftime-card-2790.jpg", "halftime"),
        ("halftime-card-5955.jpg", "halftime"),
        ("halftime-card-vit-loud-5190.jpg", "halftime"),
    ];
    for (name, kind) in positives {
        let frame = load_named(&frames_dir(), name);
        let (pause, ok) = recognizer.pause(&frame.crop(PAUSE_HEADER), &frame.crop(PAUSE_CARD));
        assert!(ok, "{name} = {pause:?} ok={ok}, want {kind}");
        assert_eq!(pause.kind, kind, "{name} kind");
        assert!(
            pause.score >= 0.7,
            "{name} score = {:.3}, want it well above the 0.6 threshold",
            pause.score
        );
        assert_eq!(pause.reason, "", "{name} reason");
    }
    let negatives = [
        "stage-3712.jpg",
        "summit-12-4.jpg",
        "prx-g2-lotus-ticker.jpg",
        "countdown-card-5835.jpg",
        "ge-vit-abyss-4322.jpg",
        "ge-vit-abyss-4324.jpg",
        "lotus-occlusion-15978.jpg",
        "lotus-occlusion-15979.jpg",
        "lotus-occlusion-15980.jpg",
        "lotus-occlusion-15981.jpg",
        "lotus-occlusion-15985.jpg",
        "lotus-occlusion-15986.jpg",
        "lotus-occlusion-16130.jpg",
        "lotus-occlusion-16131.jpg",
        "lotus-occlusion-16132.jpg",
        "lotus-occlusion-16133.jpg",
        "lotus-occlusion-16134.jpg",
        "lotus-occlusion-16135.jpg",
    ];
    for name in negatives {
        let frame = load_named(&frames_dir(), name);
        let (pause, ok) = recognizer.pause(&frame.crop(PAUSE_HEADER), &frame.crop(PAUSE_CARD));
        assert!(!ok, "{name} = {pause:?}, want no pause");
    }
}
