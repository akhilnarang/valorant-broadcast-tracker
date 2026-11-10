//! The §3.3 pipeline: presence gate, chunked inference batches, accept rules, assembly.

use std::collections::HashMap;

use ort::session::Session;
use vbt_vision::frame::Frame;
use vbt_vision::plates::{PlateReader, PlateScore};

use crate::ctc::{self, Best};
use crate::geometry;
use crate::schema::{Context, PlayerOut, Scoreboard, TeamOut};
use crate::tensor::{Batch, Crop, timesteps};
use crate::{Error, dlog};

/// Acceptance thresholds from §3.3.
const TAG_MARGIN: f64 = 6.0;
const TAG_PER_CHAR: f64 = -3.0;
const ROUND_MARGIN: f64 = 3.0;
const NAME_MARGIN: f64 = 8.0;
const NAME_PER_CHAR: f64 = -3.0;
const CELL_MARGIN: f64 = 4.0;
const CELL_SCORE: f64 = -1.5;

/// The round header's text before the number.
const ROUND_PREFIX: &str = "ROUND ";

/// Batch widths from §3.1; the actual width grows to a multiple of 8 when a crop resizes
/// wider, so the 720p right-name crops (307 px) keep their full width.
const NAMES_W: usize = 300;
const CELLS_W: usize = 64;
const HEADER_W: usize = 272;

/// Samples per run, which caps activation memory (§3.1). Samples in a batch are
/// independent, so the chunk size does not change results. Every chunk keeps the group's
/// padded width.
const HEADER_CHUNK: usize = 1;
const NAMES_CHUNK: usize = 1;
const CELLS_CHUNK: usize = 2;

/// How name and cell crops are fed to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchMode {
    /// Right-padded batches, chunked as above, every crop padded to its group's width.
    Auto,
    /// One `Run` per crop with no padding, for the batched-vs-per-crop equivalence check.
    PerCrop,
}

/// One field decision, kept for threshold reporting and the padding check.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub field: String,
    pub value: String,
    pub accepted: bool,
    pub score: f64,
    pub margin: f64,
}

/// The standalone reader: model session, class list, plate classifier, and last frame's log.
pub struct Reader {
    session: Session,
    classes: Vec<char>,
    index: HashMap<char, usize>,
    plates: PlateReader,
    mode: BatchMode,
    decisions: Vec<Decision>,
}

impl Reader {
    /// Loads the embedded model and classifier with `threads` intra-op threads.
    pub fn new(threads: usize) -> Result<Reader, Error> {
        Reader::with_mode(threads, BatchMode::Auto)
    }

    /// Loads the reader with an explicit batching mode.
    pub fn with_mode(threads: usize, mode: BatchMode) -> Result<Reader, Error> {
        let mut session = crate::model::session(threads)?;
        let (classes, index) = crate::model::classes(&mut session)?;
        Ok(Reader {
            session,
            classes,
            index,
            plates: PlateReader::new()?,
            mode,
            decisions: Vec::new(),
        })
    }

    /// Every field decision from the last [`Reader::analyze`] call.
    pub fn decisions(&self) -> &[Decision] {
        &self.decisions
    }

    /// Reads one frame, or `Ok(None)` when the table is absent.
    pub fn analyze(&mut self, frame: &Frame, ctx: &Context) -> Result<Option<Scoreboard>, Error> {
        self.decisions.clear();
        if !geometry::table_present(frame) {
            dlog!("gate: table not present");
            return Ok(None);
        }
        let Reader {
            session,
            classes,
            index,
            plates,
            mode,
            decisions,
        } = self;
        let mode = *mode;
        let class_count = classes.len();

        // Team scores come from the logistic-regression plates, not the model.
        let left_score = plates.score(&frame.crop(geometry::SCORE_LEFT));
        let right_score = plates.score(&frame.crop(geometry::SCORE_RIGHT));
        record(decisions, format_score_decision("score_left", &left_score));
        record(
            decisions,
            format_score_decision("score_right", &right_score),
        );

        // With both scores read, the round is one of two candidates.
        let round_lexicon: Vec<String> = match (left_score.value, right_score.value) {
            (Some(l), Some(r)) => {
                let s = l + r;
                vec![
                    format!("{ROUND_PREFIX}{s}"),
                    format!("{ROUND_PREFIX}{}", s + 1),
                ]
            }
            _ => (1..=50).map(|i| format!("{ROUND_PREFIX}{i}")).collect(),
        };

        // Header batch: round, then the two tags.
        let header: Vec<Crop> = [geometry::ROUND, geometry::TAG_LEFT, geometry::TAG_RIGHT]
            .iter()
            .map(|r| Crop::prepare(&frame.crop(*r)))
            .collect::<Result<_, _>>()?;
        let tag_lexicon = vec![ctx.teams[0].tag.clone(), ctx.teams[1].tag.clone()];
        let mut header_best: Vec<Option<Best>> = vec![None; 3];
        run_group(
            session,
            mode,
            &header,
            HEADER_W,
            HEADER_CHUNK,
            |i, probs, t| {
                let lexicon = if i == 0 { &round_lexicon } else { &tag_lexicon };
                header_best[i] = ctc::best(probs, class_count, t, index, lexicon, false);
            },
        )?;
        let round_best = header_best[0].take();
        let left_tag_best = header_best[1].take();
        let right_tag_best = header_best[2].take();

        let tag_accepted = |b: &Best| {
            let word = &tag_lexicon[b.index];
            b.margin >= TAG_MARGIN && b.score / word.chars().count() as f64 >= TAG_PER_CHAR
        };
        let mut left_tag = left_tag_best
            .as_ref()
            .filter(|b| tag_accepted(b))
            .map(|b| tag_lexicon[b.index].clone())
            .unwrap_or_default();
        let mut right_tag = right_tag_best
            .as_ref()
            .filter(|b| tag_accepted(b))
            .map(|b| tag_lexicon[b.index].clone())
            .unwrap_or_default();
        record_tag(
            decisions,
            "tag_left",
            &left_tag,
            left_tag_best.as_ref(),
            &tag_lexicon,
        );
        record_tag(
            decisions,
            "tag_right",
            &right_tag,
            right_tag_best.as_ref(),
            &tag_lexicon,
        );
        // Never infer one side from the other: two identical tags are both refused.
        if !left_tag.is_empty() && left_tag.eq_ignore_ascii_case(&right_tag) {
            dlog!("tags: both sides read as {left_tag:?}; refusing both");
            left_tag.clear();
            right_tag.clear();
        }

        let round_value = round_best
            .as_ref()
            .filter(|b| b.margin >= ROUND_MARGIN)
            .and_then(|b| {
                round_lexicon[b.index]
                    .strip_prefix(ROUND_PREFIX)
                    .and_then(|s| s.parse::<i64>().ok())
            });
        if let Some(b) = &round_best {
            record(
                decisions,
                Decision {
                    field: "round".into(),
                    value: round_lexicon[b.index].clone(),
                    accepted: round_value.is_some(),
                    score: b.score,
                    margin: b.margin,
                },
            );
        } else {
            record(
                decisions,
                Decision {
                    field: "round".into(),
                    value: String::new(),
                    accepted: false,
                    score: f64::NEG_INFINITY,
                    margin: 0.0,
                },
            );
        }

        // Each side uses the accepted tag's roster, or the union when the tag is unread.
        let left_roster = roster_entries(ctx, &left_tag);
        let right_roster = roster_entries(ctx, &right_tag);
        let rosters: [&Vec<(String, String)>; 2] = [&left_roster, &right_roster];

        // Each name crop is scored against its own side's roster.
        let mut name_crops = Vec::with_capacity(10);
        for i in 0..5 {
            name_crops.push(Crop::prepare(&frame.crop(geometry::name_box(false, i)))?);
            name_crops.push(Crop::prepare(&frame.crop(geometry::name_box(true, i)))?);
        }
        let name_candidates: Vec<Vec<String>> = (0..10)
            .map(|i| {
                rosters[i % 2]
                    .iter()
                    .map(|(_, name)| name.clone())
                    .collect()
            })
            .collect();
        for (i, candidates) in name_candidates.iter().enumerate() {
            for unscorable in candidates
                .iter()
                .filter(|name| ctc::extension(name, index).is_none())
            {
                dlog!("name {unscorable:?} (row {i}) has no scorable case variant");
            }
        }
        let mut name_best: Vec<Option<Best>> = vec![None; 10];
        run_group(
            session,
            mode,
            &name_crops,
            NAMES_W,
            NAMES_CHUNK,
            |i, probs, t| {
                name_best[i] = ctc::best(probs, class_count, t, index, &name_candidates[i], true);
            },
        )?;

        let mut accepted_name: Vec<Option<usize>> = vec![None; 10];
        for i in 0..10 {
            let roster = rosters[i % 2];
            let mut chosen = None;
            let mut value = String::new();
            let mut accepted = false;
            let mut score = f64::NEG_INFINITY;
            let mut margin = 0.0;
            if let Some(b) = &name_best[i] {
                let word = &roster[b.index].1;
                accepted = b.margin >= NAME_MARGIN
                    && b.score / word.chars().count() as f64 >= NAME_PER_CHAR;
                value = word.clone();
                score = b.score;
                margin = b.margin;
                if accepted {
                    chosen = Some(b.index);
                }
            }
            accepted_name[i] = chosen;
            record(
                decisions,
                Decision {
                    field: format!(
                        "name_{}_{}",
                        if i % 2 == 0 { "left" } else { "right" },
                        i / 2
                    ),
                    value,
                    accepted,
                    score,
                    margin,
                },
            );
        }

        // Two rows on a side reading the same name invalidate both.
        for side in 0..2 {
            let mut counts: HashMap<String, usize> = HashMap::new();
            for row in 0..5 {
                if let Some(idx) = accepted_name[row * 2 + side] {
                    *counts
                        .entry(rosters[side][idx].1.to_lowercase())
                        .or_default() += 1;
                }
            }
            for row in 0..5 {
                let i = row * 2 + side;
                if let Some(idx) = accepted_name[i]
                    && counts[&rosters[side][idx].1.to_lowercase()] > 1
                {
                    dlog!(
                        "name {:?} accepted twice on one side; dropping both",
                        rosters[side][idx].1
                    );
                    accepted_name[i] = None;
                }
            }
        }

        // K/D/A cells are scored against "0".."99".
        let mut cell_crops = Vec::with_capacity(30);
        for side in 0..2 {
            for i in 0..5 {
                for j in 0..3 {
                    cell_crops.push(Crop::prepare(&frame.crop(geometry::cell_box(
                        side == 1,
                        i,
                        j,
                    )))?);
                }
            }
        }
        let numbers: Vec<String> = (0..100).map(|n| n.to_string()).collect();
        let mut cell_best: Vec<Option<Best>> = vec![None; 30];
        run_group(
            session,
            mode,
            &cell_crops,
            CELLS_W,
            CELLS_CHUNK,
            |i, probs, t| {
                cell_best[i] = ctc::best(probs, class_count, t, index, &numbers, false);
            },
        )?;
        let mut cell_value: Vec<Option<i64>> = vec![None; 30];
        for i in 0..30 {
            let side = if i < 15 { "left" } else { "right" };
            let k = i % 15;
            let (value, accepted, score, margin) = match &cell_best[i] {
                Some(b) => {
                    let ok = b.margin >= CELL_MARGIN && b.score >= CELL_SCORE;
                    (numbers[b.index].clone(), ok, b.score, b.margin)
                }
                None => (String::new(), false, f64::NEG_INFINITY, 0.0),
            };
            if accepted {
                cell_value[i] = value.parse::<i64>().ok();
            }
            record(
                decisions,
                Decision {
                    field: format!("kda_{side}_{}_{}", k / 3, k % 3),
                    value,
                    accepted,
                    score,
                    margin,
                },
            );
        }

        // A player is emitted only with its name and all three cells accepted.
        let mut players = Vec::new();
        for side in 0..2 {
            for row in 0..5 {
                let Some(idx) = accepted_name[row * 2 + side] else {
                    continue;
                };
                let cells = (0..3)
                    .filter_map(|j| cell_value[side * 15 + row * 3 + j])
                    .collect::<Vec<_>>();
                if cells.len() != 3 {
                    continue;
                }
                players.push(PlayerOut {
                    team: rosters[side][idx].0.clone(),
                    name: rosters[side][idx].1.clone(),
                    kda: [cells[0], cells[1], cells[2]],
                });
            }
        }
        Ok(Some(Scoreboard {
            round: round_value,
            teams: vec![
                TeamOut {
                    name: left_tag,
                    score: left_score.value,
                },
                TeamOut {
                    name: right_tag,
                    score: right_score.value,
                },
            ],
            players,
        }))
    }
}

/// The roster for an accepted tag, or the union of both rosters when it is unread.
fn roster_entries(ctx: &Context, tag: &str) -> Vec<(String, String)> {
    if !tag.is_empty()
        && let Some(team) = ctx.teams.iter().find(|t| t.tag.eq_ignore_ascii_case(tag))
    {
        return team
            .players
            .iter()
            .map(|p| (team.tag.clone(), p.clone()))
            .collect();
    }
    ctx.teams
        .iter()
        .flat_map(|t| t.players.iter().map(move |p| (t.tag.clone(), p.clone())))
        .collect()
}

fn format_score_decision(field: &str, score: &PlateScore) -> Decision {
    Decision {
        field: field.into(),
        value: score.value.map(|v| v.to_string()).unwrap_or_default(),
        accepted: score.value.is_some(),
        score: score.confidence,
        margin: 0.0,
    }
}

fn record_tag(
    decisions: &mut Vec<Decision>,
    field: &str,
    value: &str,
    best: Option<&Best>,
    lexicon: &[String],
) {
    let decision = match best {
        Some(b) => Decision {
            field: field.into(),
            value: lexicon[b.index].clone(),
            accepted: !value.is_empty(),
            score: b.score,
            margin: b.margin,
        },
        None => Decision {
            field: field.into(),
            value: String::new(),
            accepted: false,
            score: f64::NEG_INFINITY,
            margin: 0.0,
        },
    };
    record(decisions, decision);
}

fn record(decisions: &mut Vec<Decision>, decision: Decision) {
    if !decision.accepted {
        dlog!(
            "rejected {} value={:?} score={:.2} margin={:.2}",
            decision.field,
            decision.value,
            decision.score,
            decision.margin
        );
    }
    decisions.push(decision);
}

/// Runs one group in the configured mode. In `Auto`, the crops go in right-padded
/// batches of at most `chunk` samples. The batch is widened to fit the widest crop, so
/// no crop is truncated.
fn run_group<F>(
    session: &mut Session,
    mode: BatchMode,
    crops: &[Crop],
    nominal_w: usize,
    chunk: usize,
    mut score_one: F,
) -> Result<(), Error>
where
    F: FnMut(usize, &[f32], usize),
{
    let widths: Vec<usize> = crops.iter().map(|c| c.width).collect();
    if mode == BatchMode::Auto {
        let width = Batch::width_for(crops, nominal_w);
        let mut base = 0usize;
        for group in crops.chunks(chunk) {
            let batch = Batch::new(group, width);
            run_batch(session, &batch, |i, probs, t| {
                score_one(base + i, probs, t.min(timesteps(widths[base + i])))
            })?;
            base += group.len();
        }
        return Ok(());
    }
    for (i, crop) in crops.iter().enumerate() {
        let batch = Batch::new(std::slice::from_ref(crop), crop.width);
        run_batch(session, &batch, |_, probs, t| {
            score_one(i, probs, t.min(timesteps(widths[i])))
        })?;
    }
    Ok(())
}

fn run_batch<F>(session: &mut Session, batch: &Batch, mut per_crop: F) -> Result<(), Error>
where
    F: FnMut(usize, &[f32], usize),
{
    let (data, t, c) = crate::model::run(session, batch.n, batch.w, &batch.data)?;
    for i in 0..batch.n {
        per_crop(i, &data[i * t * c..(i + 1) * t * c], t);
    }
    Ok(())
}
