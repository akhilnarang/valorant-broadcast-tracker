//! Serde structs for the context file (§3.5) and the canonical output (§3.4).

use serde::{Deserialize, Serialize};

/// The context file: exactly two teams, each with a non-empty tag and its roster
/// (substitutes allowed). Parsing rejects any other shape, so a bad file fails at load
/// rather than on every frame.
#[derive(Debug, Clone, Deserialize)]
#[serde(try_from = "Teams")]
pub struct Context {
    pub teams: [ContextTeam; 2],
}

#[derive(Deserialize)]
struct Teams {
    teams: Vec<ContextTeam>,
}

impl TryFrom<Teams> for Context {
    type Error = String;

    fn try_from(raw: Teams) -> Result<Context, String> {
        let count = raw.teams.len();
        let teams: [ContextTeam; 2] = raw
            .teams
            .try_into()
            .map_err(|_| format!("context has {count} teams, need exactly 2"))?;
        if teams.iter().any(|t| t.tag.is_empty()) {
            return Err("context has a team with an empty tag".into());
        }
        Ok(Context { teams })
    }
}

/// One context team.
#[derive(Debug, Clone, Deserialize)]
pub struct ContextTeam {
    pub tag: String,
    pub players: Vec<String>,
}

/// One header side of the canonical output. An unread tag is `""`, an unread score `null`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TeamOut {
    pub name: String,
    pub score: Option<i64>,
}

/// One table row of the canonical output.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PlayerOut {
    pub team: String,
    pub name: String,
    pub kda: [i64; 3],
}

/// The canonical scoreboard: teams left then right; players left rows then right rows.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Scoreboard {
    pub round: Option<i64>,
    pub teams: Vec<TeamOut>,
    pub players: Vec<PlayerOut>,
}
