//! The backend's video context source. It holds the latest context for the live match and satisfies
//! the tracker's `MapOrder` boundary.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::cancel::{CancellationFlag, sleep_cancelled};
use crate::push::{Context, Pusher};
use crate::riot::LiveRiot;
use crate::tracker::MapOrder;
use crate::tracker::Riot as RiotSource;

/// How often the backend's video context is refreshed while a match is live.
pub const CONTEXT_REFRESH: Duration = Duration::from_secs(60);
/// How often the refresh is considered, so a new match id fetches immediately.
pub const CONTEXT_POLL: Duration = Duration::from_secs(5);

/// Holds the backend's latest video context for the live match.
#[derive(Default)]
struct Inner {
    context: Context,
    /// The Riot match id this context was requested for.
    riot_match: String,
}

/// The context holder. A zero context names no map, so the strip remains the name
/// source until the first context arrives.
pub struct ContextSource {
    pusher: Arc<Pusher>,
    inner: Mutex<Inner>,
}

impl ContextSource {
    pub fn new(pusher: Arc<Pusher>) -> ContextSource {
        ContextSource {
            pusher,
            inner: Mutex::new(Inner::default()),
        }
    }

    /// Drops the cached context, so a previous match's names cannot name the next
    /// match's maps.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        *inner = Inner::default();
    }

    /// Refreshes the context for the given Riot team codes. The codes, not a match
    /// id, bind the context to a match: the context's match id is not Riot's, so the
    /// returned teams must name the teams we asked about.
    pub async fn fetch(&self, codes: Vec<String>, riot_match: String) {
        let value = match self.pusher.fetch_context(&codes).await {
            Ok(value) => value,
            Err(error) => {
                // A 404, or a backend that does not answer, just means no veto
                // names; the strip still names maps. A response that
                // did not decode is contract drift, and it must be loud.
                if matches!(&error, crate::push::Error::Http(http) if http.is_decode()) {
                    tracing::warn!(error = %error, "video context response did not decode");
                } else {
                    tracing::debug!(error = %error, "video context fetch failed");
                }
                return;
            }
        };
        if !teams_match(&codes, &value.teams) {
            tracing::warn!(
                context = ?value.teams,
                riot = ?codes,
                "video context teams do not match Riot's"
            );
            return;
        }
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.context = value.clone();
        inner.riot_match = riot_match.clone();
        drop(inner);
        tracing::info!(
            veto_match = %value.match_id,
            riot_match = %riot_match,
            map_order = ?value.map_order,
            "video context"
        );
    }
}

impl MapOrder for ContextSource {
    /// The veto name for a game number, or `""` when the cached context does not
    /// name that game.
    fn map_name(&self, match_id: &str, map_number: i64) -> String {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if match_id.is_empty()
            || match_id != inner.riot_match
            || map_number < 1
            || map_number as usize > inner.context.map_order.len()
        {
            return String::new();
        }
        inner.context.map_order[map_number as usize - 1].clone()
    }

    /// The veto/context's competitors only for its bound Riot match.
    fn team_codes(&self, match_id: &str) -> Option<Vec<String>> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if match_id != inner.riot_match {
            return None;
        }
        Some(inner.context.teams.clone())
    }
}

/// Whether every wanted code names one of the context's teams, case-insensitively.
pub fn teams_match(wanted: &[String], got: &[String]) -> bool {
    if got.is_empty() {
        return false;
    }
    let known: std::collections::BTreeSet<String> =
        got.iter().map(|team| team.trim().to_lowercase()).collect();
    wanted
        .iter()
        .all(|code| known.contains(&code.trim().to_lowercase()))
}

/// Refreshes the video context every minute while Riot lists a live match.
pub async fn watch_context(cancel: CancellationFlag, source: Arc<ContextSource>, riot: LiveRiot) {
    let mut last_match = String::new();
    let mut last_fetch: Option<Instant> = None;
    loop {
        if sleep_cancelled(&cancel, CONTEXT_POLL).await {
            return;
        }
        let match_id = riot.match_id();
        if !riot.live_now() || match_id.is_empty() {
            continue;
        }
        let codes: Vec<String> = riot
            .teams()
            .into_iter()
            .map(|team| team.code)
            .filter(|code| !code.is_empty())
            .collect();
        if codes.len() != 2 {
            // The context endpoint requires exactly two team codes.
            continue;
        }
        if match_id != last_match {
            source.clear();
            last_match = match_id.clone();
            // A new match must fetch at once, not wait out the interval.
            last_fetch = None;
        }
        if let Some(fetched) = last_fetch
            && fetched.elapsed() < CONTEXT_REFRESH
        {
            continue;
        }
        source.fetch(codes, match_id).await;
        last_fetch = Some(Instant::now());
    }
}

/// A fixed series map order, for replaying without a backend.
#[derive(Debug, Clone, Default)]
pub struct StaticMapOrder {
    order: Vec<String>,
}

impl StaticMapOrder {
    pub fn new(order: Vec<String>) -> StaticMapOrder {
        StaticMapOrder { order }
    }
}

impl MapOrder for StaticMapOrder {
    fn map_name(&self, _match_id: &str, map_number: i64) -> String {
        if map_number < 1 || map_number as usize > self.order.len() {
            return String::new();
        }
        self.order[map_number as usize - 1].trim().to_string()
    }
}
