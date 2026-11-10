//! The Riot esports API poller: a background task that watches the match Riot lists as live and
//! holds the details the tracker reads: teams, sides per game, game states and series wins.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use crate::cancel::{CancellationFlag, sleep_cancelled};
use crate::tracker::{Riot as RiotSource, ScoreMap, Sides, Team};

/// The persisted esports API base.
pub const API: &str = "https://esports-api.service.valorantesports.com/persisted/val";
/// The public key valorantesports.com itself sends.
pub const API_KEY: &str = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z";
/// The pause between background polls.
pub const POLL_INTERVAL: Duration = Duration::from_secs(30);
/// Bounds one API call.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);

/// Placeholder teams stand in until Riot lists a match, so the tracker can still
/// follow the score by side.
pub const PLACEHOLDER_LEFT: (&str, &str, &str) = ("left", "LEFT", "LEFT");
pub const PLACEHOLDER_RIGHT: (&str, &str, &str) = ("right", "RIGHT", "RIGHT");

fn placeholder_teams() -> Vec<Team> {
    vec![
        Team {
            id: PLACEHOLDER_LEFT.0.to_string(),
            code: PLACEHOLDER_LEFT.1.to_string(),
            name: PLACEHOLDER_LEFT.2.to_string(),
        },
        Team {
            id: PLACEHOLDER_RIGHT.0.to_string(),
            code: PLACEHOLDER_RIGHT.1.to_string(),
            name: PLACEHOLDER_RIGHT.2.to_string(),
        },
    ]
}

/// A poll or API failure. Failed polls are logged and ridden out, never fatal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// One team of a match.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TeamDetails {
    pub id: String,
    pub code: String,
    pub name: String,
    pub game_wins: i64,
}

/// One game (map) of a series.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GameDetails {
    pub number: i64,
    pub state: String,
    pub teams: Vec<GameTeam>,
}

/// One team's side in a game.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GameTeam {
    pub id: String,
    pub side: String,
}

/// The part of `getEventDetails` the tracker reads.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MatchDetails {
    pub teams: Vec<TeamDetails>,
    pub games: Vec<GameDetails>,
    pub strategy_count: i64,
}

/// The shared poller state.
#[derive(Debug, Clone, Default)]
pub struct RiotState {
    pub live_now: bool,
    pub match_id: String,
    pub event: Option<MatchDetails>,
}

/// The tracker's `Riot` boundary over the shared state.
#[derive(Clone, Default)]
pub struct LiveRiot {
    state: Arc<Mutex<RiotState>>,
}

impl LiveRiot {
    pub fn from_state(state: Arc<Mutex<RiotState>>) -> LiveRiot {
        LiveRiot { state }
    }

    fn snapshot(&self) -> (String, MatchDetails) {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        (
            state.match_id.clone(),
            state.event.clone().unwrap_or_default(),
        )
    }
}

/// The teams of a match, falling back to the placeholders when Riot lists none.
fn team_list(game_match: &MatchDetails) -> Vec<Team> {
    if game_match.teams.is_empty() {
        return placeholder_teams();
    }
    game_match
        .teams
        .iter()
        .map(|team| Team {
            id: team.id.clone(),
            code: team.code.clone(),
            name: team.name.clone(),
        })
        .collect()
}

/// One game of a match, or `None` when Riot doesn't list it.
fn find_game(game_match: &MatchDetails, number: i64) -> Option<&GameDetails> {
    game_match.games.iter().find(|game| game.number == number)
}

impl RiotSource for LiveRiot {
    fn live_now(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .live_now
    }

    fn has_event(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .event
            .is_some()
    }

    fn match_id(&self) -> String {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .match_id
            .clone()
    }

    fn teams(&self) -> Vec<Team> {
        let (_, game_match) = self.snapshot();
        team_list(&game_match)
    }

    fn sides(&self, match_id: &str, map_number: i64) -> Option<Sides> {
        let (current, game_match) = self.snapshot();
        if match_id != current {
            return None;
        }
        let mut code_by_id = std::collections::BTreeMap::new();
        for team in team_list(&game_match) {
            code_by_id.insert(team.id, team.code);
        }
        let game = find_game(&game_match, map_number)?;
        let mut by_side = std::collections::BTreeMap::new();
        for team in &game.teams {
            let code = code_by_id.get(&team.id).cloned().unwrap_or_default();
            by_side.insert(team.side.to_lowercase(), code);
        }
        let blue = by_side.get("blue").cloned().unwrap_or_default();
        let red = by_side.get("red").cloned().unwrap_or_default();
        if blue.is_empty() || red.is_empty() {
            return None;
        }
        Some(Sides { blue, red })
    }

    fn listed_order(&self, match_id: &str) -> Sides {
        let (current, game_match) = self.snapshot();
        let listed = if match_id == current {
            team_list(&game_match)
        } else {
            placeholder_teams()
        };
        Sides {
            blue: listed.first().map(|t| t.code.clone()).unwrap_or_default(),
            red: listed.get(1).map(|t| t.code.clone()).unwrap_or_default(),
        }
    }

    fn map_number(&self, maps_played: i64) -> i64 {
        let (_, game_match) = self.snapshot();
        for game in &game_match.games {
            if game.state == "inProgress" {
                return game.number;
            }
        }
        maps_played + 1
    }

    fn game_state(&self, map_number: i64) -> Option<String> {
        let (_, game_match) = self.snapshot();
        find_game(&game_match, map_number).map(|game| game.state.clone())
    }

    fn game_wins(&self) -> ScoreMap {
        let (_, game_match) = self.snapshot();
        let mut wins = ScoreMap::new();
        for team in &game_match.teams {
            wins.insert(team.code.clone(), team.game_wins as i32);
        }
        wins
    }

    fn best_of(&self) -> i64 {
        let (_, game_match) = self.snapshot();
        game_match.strategy_count
    }
}

/// The background poller for the match Riot lists as live.
pub struct Poller {
    base_url: String,
    client: reqwest::Client,
    state: Arc<Mutex<RiotState>>,
}

impl Poller {
    /// Builds a poller with an explicit base URL, for tests and the hidden override.
    pub fn with_base_url(base_url: &str, state: Arc<Mutex<RiotState>>) -> Result<Poller, Error> {
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|e| Error(format!("riot client: {e}")))?;
        Ok(Poller {
            base_url: base_url.trim_end_matches('/').to_string(),
            client,
            state,
        })
    }

    /// Refreshes the live match and its details. On failure the last known details
    /// are kept, so the next poll can recover.
    pub async fn poll(&self) -> Result<(), Error> {
        let live: LiveDetails = self.get("getLiveDetails", &[]).await?;
        let matches: Vec<&LiveEvent> = live
            .data
            .schedule
            .events
            .as_deref()
            .unwrap_or_default()
            .iter()
            .filter(|event| event.is_match())
            .collect();

        let match_id = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.live_now = !matches.is_empty();
            let ids: Vec<String> = matches
                .iter()
                .map(|event| event.match_id())
                .filter(|id| !id.is_empty())
                .collect();
            if !ids.contains(&state.match_id)
                && let Some(id) = ids.into_iter().next()
            {
                tracing::info!(id = %id, "riot live match");
                state.match_id = id;
                // Drop the previous match's details: until this one's arrive,
                // `has_event` must stay false.
                state.event = None;
            }
            state.match_id.clone()
        };
        if match_id.is_empty() {
            return Ok(());
        }
        let details: EventDetails = self.get("getEventDetails", &[("id", match_id)]).await?;
        let match_details = MatchDetails::from(details.data.event.game_match);
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.event = Some(match_details);
        Ok(())
    }

    /// Polls first, then sleeps `POLL_INTERVAL`; exits on cancellation.
    pub async fn run(self, cancel: CancellationFlag) {
        loop {
            if let Err(error) = self.poll().await {
                tracing::warn!(error = %error, "riot poll failed");
            }
            if sleep_cancelled(&cancel, POLL_INTERVAL).await {
                return;
            }
        }
    }

    async fn get<T: serde::de::DeserializeOwned>(
        &self,
        endpoint: &str,
        params: &[(&str, String)],
    ) -> Result<T, Error> {
        let url = format!("{}/{}", self.base_url, endpoint);
        let mut query: Vec<(&str, String)> =
            vec![("hl", "en-US".to_string()), ("sport", "val".to_string())];
        for (key, value) in params {
            query.push((key, value.clone()));
        }
        let response = self
            .client
            .get(&url)
            .query(&query)
            .header("x-api-key", API_KEY)
            .send()
            .await
            .map_err(|e| Error(format!("riot {endpoint}: {e}")))?;
        let status = response.status().as_u16();
        let body = response
            .text()
            .await
            .map_err(|e| Error(format!("riot {endpoint}: {e}")))?;
        if status != 200 {
            let excerpt: String = body.chars().take(200).collect();
            return Err(Error(format!(
                "riot {endpoint} answered {status}: {excerpt:?}"
            )));
        }
        serde_json::from_str(&body).map_err(|e| Error(format!("riot {endpoint}: {e}")))
    }
}

/// Renders a raw JSON string or number as text. A number keeps its literal text,
/// so a large id keeps all its digits.
pub fn id_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        _ => String::new(),
    }
}

fn de_id<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    Ok(id_string(&value))
}

fn de_flex_i64<'de, D>(deserializer: D) -> Result<i64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::Null => Ok(0),
        Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0) as i64),
        Value::String(s) => s
            .trim()
            .parse::<f64>()
            .map(|v| v as i64)
            .map_err(|_| serde::de::Error::custom(format!("invalid integer {s:?}"))),
        other => Err(serde::de::Error::custom(format!(
            "invalid integer {other:?}"
        ))),
    }
}

#[derive(Debug, Deserialize)]
struct LiveDetails {
    data: LiveData,
}

#[derive(Debug, Deserialize)]
struct LiveData {
    schedule: Schedule,
}

#[derive(Debug, Deserialize)]
struct Schedule {
    #[serde(default)]
    events: Option<Vec<LiveEvent>>,
}

/// One scheduled event; any event with a match is live.
#[derive(Debug, Deserialize)]
struct LiveEvent {
    #[serde(default)]
    id: Value,
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(rename = "match", default)]
    game_match: Value,
}

impl LiveEvent {
    fn is_match(&self) -> bool {
        self.kind == "match"
            || self
                .game_match
                .as_object()
                .is_some_and(|map| !map.is_empty())
    }

    fn match_id(&self) -> String {
        if let Some(id) = self.game_match.get("id") {
            let id = id_string(id);
            if !id.is_empty() {
                return id;
            }
        }
        id_string(&self.id)
    }
}

#[derive(Debug, Deserialize)]
struct EventDetails {
    data: EventData,
}

#[derive(Debug, Deserialize)]
struct EventData {
    event: EventBody,
}

#[derive(Debug, Deserialize)]
struct EventBody {
    #[serde(rename = "match", default)]
    game_match: MatchDetailsJson,
}

#[derive(Debug, Deserialize, Default)]
struct MatchDetailsJson {
    #[serde(default)]
    teams: Vec<TeamJson>,
    #[serde(default)]
    games: Vec<GameJson>,
    #[serde(default)]
    strategy: StrategyJson,
}

#[derive(Debug, Deserialize, Default)]
struct StrategyJson {
    #[serde(default)]
    count: i64,
}

#[derive(Debug, Deserialize, Default)]
struct TeamJson {
    #[serde(default, deserialize_with = "de_id")]
    id: String,
    #[serde(default)]
    code: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    result: ResultJson,
}

#[derive(Debug, Deserialize, Default)]
struct ResultJson {
    #[serde(default, rename = "gameWins", deserialize_with = "de_flex_i64")]
    game_wins: i64,
}

#[derive(Debug, Deserialize, Default)]
struct GameJson {
    #[serde(default, deserialize_with = "de_flex_i64")]
    number: i64,
    #[serde(default)]
    state: String,
    #[serde(default)]
    teams: Vec<GameTeamJson>,
}

#[derive(Debug, Deserialize, Default)]
struct GameTeamJson {
    #[serde(default, deserialize_with = "de_id")]
    id: String,
    #[serde(default)]
    side: String,
}

impl From<MatchDetailsJson> for MatchDetails {
    fn from(raw: MatchDetailsJson) -> MatchDetails {
        MatchDetails {
            teams: raw
                .teams
                .into_iter()
                .map(|team| TeamDetails {
                    id: team.id,
                    code: team.code,
                    name: team.name,
                    game_wins: team.result.game_wins,
                })
                .collect(),
            games: raw
                .games
                .into_iter()
                .map(|game| GameDetails {
                    number: game.number,
                    state: game.state,
                    teams: game
                        .teams
                        .into_iter()
                        .map(|team| GameTeam {
                            id: team.id,
                            side: team.side,
                        })
                        .collect(),
                })
                .collect(),
            strategy_count: raw.strategy.count,
        }
    }
}
