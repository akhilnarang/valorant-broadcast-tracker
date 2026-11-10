//! The score publisher and per-run auth token. It hands the tracker's map score to val-esports-backend
//! over a Unix socket.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use serde::{Deserialize, Serialize};

use crate::tracker::{Pause, PushError, ScoreMap, Status, Team};

/// Where the backend accepts a video score.
pub const SCORE_PATH: &str = "/api/v1/video/score";
/// Where the backend answers with the tracker's match and the series' ordered map names.
pub const CONTEXT_PATH: &str = "/api/v1/video/context";
/// Bounds one write, so a hung backend can't stall the 1 fps loop.
pub const TIMEOUT: Duration = Duration::from_secs(2);
/// The token's random bytes before encoding.
pub const TOKEN_BYTES: usize = 32;

/// The push failure type.
#[derive(Debug)]
pub enum Error {
    /// Publishing was attempted without a configured socket and token file.
    Disabled,
    /// Another live writer holds the token lock.
    Locked(String),
    /// An I/O failure.
    Io(std::io::Error),
    /// An HTTP failure.
    Http(reqwest::Error),
    /// The backend answered a status other than the expected one.
    Status(String),
    /// The token could not be generated.
    Token(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Disabled => write!(f, "push: backend socket or video token file not configured"),
            Error::Locked(path) => write!(f, "push: another live writer holds {path}"),
            Error::Io(e) => write!(f, "push: io: {e}"),
            Error::Http(e) => write!(f, "push: http: {e}"),
            Error::Status(m) => write!(f, "push: {m}"),
            Error::Token(m) => write!(f, "push: token: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Error::Io(value)
    }
}

impl From<reqwest::Error> for Error {
    fn from(value: reqwest::Error) -> Self {
        Error::Http(value)
    }
}

/// Where the backend listens and where this run's token goes, from the environment
/// or `.env`. Either empty disables publishing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Config {
    /// `BACKEND_SOCKET`.
    pub socket: String,
    /// `VIDEO_TOKEN_FILE`.
    pub token_file: String,
    /// A shadow/replay safety gate: never own the token or write scores.
    pub read_only: bool,
}

impl Config {
    /// Whether scores may be written: configured and not read-only.
    pub fn enabled(&self) -> bool {
        !self.read_only && self.context_enabled()
    }

    /// Whether read-only match context can be fetched.
    pub fn context_enabled(&self) -> bool {
        !self.socket.is_empty() && !self.token_file.is_empty()
    }

    /// Reads `BACKEND_SOCKET` and `VIDEO_TOKEN_FILE`. The process environment wins
    /// over `.env` in the working directory.
    pub fn from_env() -> Config {
        let mut values = read_dotenv(Path::new(".env"));
        for key in ["BACKEND_SOCKET", "VIDEO_TOKEN_FILE"] {
            if let Some(value) = std::env::var_os(key) {
                values.insert(key.to_string(), value.to_string_lossy().into_owned());
            }
        }
        Config {
            socket: values.get("BACKEND_SOCKET").cloned().unwrap_or_default(),
            token_file: values.get("VIDEO_TOKEN_FILE").cloned().unwrap_or_default(),
            read_only: false,
        }
    }
}

/// Returns the `KEY=VALUE` pairs in `path`, ignoring blank lines, comments and an
/// optional `export` prefix. A missing or unreadable file yields an empty map.
pub fn read_dotenv(path: &Path) -> BTreeMap<String, String> {
    let mut values = BTreeMap::new();
    let Ok(data) = std::fs::read_to_string(path) else {
        return values;
    };
    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let mut value = value.trim();
        let bytes = value.as_bytes();
        if bytes.len() >= 2
            && (bytes[0] == b'"' || bytes[0] == b'\'')
            && bytes[bytes.len() - 1] == bytes[0]
        {
            value = &value[1..value.len() - 1];
        } else if let Some(index) = value.find(" #") {
            value = value[..index].trim();
        }
        values.insert(key.to_string(), value.to_string());
    }
    values
}

/// Returns a fresh token: 32 random bytes as unpadded URL-safe base64.
pub fn new_token() -> Result<String, Error> {
    let mut raw = [0u8; TOKEN_BYTES];
    getrandom::fill(&mut raw).map_err(|e| Error::Token(e.to_string()))?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(raw))
}

/// One team's score, as val-esports-backend's `VideoTeam`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoTeam {
    pub code: String,
    pub name: String,
    pub score: i32,
    /// `"blue"` for the team on the broadcast's left score plate, `"red"` for the right.
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub side: String,
}

/// The score written, as val-esports-backend's `VideoScore`.
#[derive(Debug, Clone, Serialize)]
pub struct VideoScore {
    pub status: Status,
    pub observed_at: i64,
    pub map_number: i64,
    pub teams: Vec<VideoTeam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pause: Option<VideoPause>,
}

/// A broadcast pause the tracker saw on screen.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VideoPause {
    pub kind: String,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub reason: String,
    pub since: i64,
}

/// The backend's answer to the context request.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct Context {
    #[serde(default)]
    pub match_id: String,
    #[serde(default)]
    pub map_order: Vec<String>,
    #[serde(default)]
    pub teams: Vec<String>,
}

/// The live writer and read-only context client over the backend's Unix socket.
pub struct Pusher {
    pub config: Config,
    /// Authenticates this run; the backend compares it with the token file.
    pub token: String,
    /// Set when a socket and token file are configured.
    client: Option<reqwest::Client>,
    /// Held for the life of the process, so two live writers can't share a token file.
    lock: Option<std::fs::File>,
}

impl Pusher {
    /// Builds a pusher for the config with a fresh token.
    pub fn new(config: Config) -> Result<Pusher, Error> {
        let token = new_token()?;
        let client = if config.context_enabled() {
            Some(build_uds_client(&config.socket)?)
        } else {
            None
        };
        Ok(Pusher {
            config,
            token,
            client,
            lock: None,
        })
    }

    /// Writes this run's token to the configured file atomically, readable only by
    /// this user, taking the live-writer lock first.
    pub fn write_token(&mut self) -> Result<(), Error> {
        if self.config.read_only || self.config.token_file.is_empty() {
            return Err(Error::Disabled);
        }
        let path = PathBuf::from(&self.config.token_file);
        let lock_path = PathBuf::from(format!("{}.lock", path.display()));
        let lock = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(&lock_path)?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => {
                return Err(Error::Locked(lock_path.display().to_string()));
            }
            Err(std::fs::TryLockError::Error(e)) => return Err(Error::Io(e)),
        }
        self.lock = Some(lock);

        let tmp = PathBuf::from(format!("{}.tmp.{}", path.display(), std::process::id()));
        let result = (|| -> Result<(), Error> {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&tmp)?;
            file.write_all(self.token.as_bytes())?;
            file.sync_all()?;
            drop(file);
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
            std::fs::rename(&tmp, &path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        result
    }

    /// Writes the map score to the backend. `blue` is the code of the team on the
    /// broadcast's left score plate, or `""` for a caller without a known mapping.
    pub async fn publish(
        &self,
        map_number: i64,
        score: &ScoreMap,
        teams: &[Team],
        blue: &str,
        status: Status,
        pause: Option<&Pause>,
    ) -> Result<(), Error> {
        let (true, Some(client)) = (self.config.enabled(), &self.client) else {
            return Err(Error::Disabled);
        };
        let names: BTreeMap<&str, &str> = teams
            .iter()
            .map(|team| (team.code.as_str(), team.name.as_str()))
            .collect();
        // A score map has no order, so sort the codes to keep the body deterministic.
        let video_teams: Vec<VideoTeam> = score
            .iter()
            .map(|(code, value)| {
                let name = names.get(code.as_str()).copied().unwrap_or(code.as_str());
                let side = if blue.is_empty() {
                    ""
                } else if code == blue {
                    "blue"
                } else {
                    "red"
                };
                VideoTeam {
                    code: code.clone(),
                    name: name.to_string(),
                    score: *value,
                    side: side.to_string(),
                }
            })
            .collect();
        let body = VideoScore {
            status,
            observed_at: now_unix(),
            map_number,
            teams: video_teams,
            pause: pause.map(|pause| VideoPause {
                kind: pause.kind.clone(),
                reason: pause.reason.clone(),
                since: pause.since,
            }),
        };
        let response = client
            .put(format!("http://localhost{SCORE_PATH}"))
            .header("X-Video-Token", &self.token)
            .json(&body)
            .send()
            .await?;
        if response.status().as_u16() != 204 {
            let status = response.status().as_u16();
            let excerpt = error_excerpt(response).await;
            return Err(Error::Status(format!(
                "backend answered {status}: {excerpt:?}"
            )));
        }
        Ok(())
    }

    /// Asks the backend which match this run publishes for and the series' map order, over
    /// the same socket and token as `publish`.
    pub async fn fetch_context(&self, codes: &[String]) -> Result<Context, Error> {
        let Some(client) = &self.client else {
            return Err(Error::Disabled);
        };
        let token = if self.config.read_only {
            // Read afresh on each GET so a primary restart does not strand the shadow run.
            let data = std::fs::read_to_string(&self.config.token_file)?;
            let token = data.trim().to_string();
            if token.is_empty() {
                return Err(Error::Disabled);
            }
            token
        } else {
            self.token.clone()
        };
        let mut request = client.get(format!("http://localhost{CONTEXT_PATH}"));
        if !codes.is_empty() {
            request = request.query(&[("codes", codes.join(","))]);
        }
        let response = request.header("X-Video-Token", token).send().await?;
        if response.status().as_u16() != 200 {
            let status = response.status().as_u16();
            let excerpt = error_excerpt(response).await;
            return Err(Error::Status(format!(
                "backend answered {status}: {excerpt:?}"
            )));
        }
        Ok(response.json::<Context>().await?)
    }
}

/// Stops consuming an error response after 200 bytes, even if its body never ends.
async fn error_excerpt(mut response: reqwest::Response) -> String {
    let mut body = Vec::with_capacity(200);
    while body.len() < 200 {
        match response.chunk().await {
            Ok(Some(chunk)) => body.extend_from_slice(&chunk[..chunk.len().min(200 - body.len())]),
            _ => break,
        }
    }
    String::from_utf8_lossy(&body).into_owned()
}

fn build_uds_client(socket: &str) -> Result<reqwest::Client, Error> {
    reqwest::Client::builder()
        .unix_socket(socket)
        .timeout(TIMEOUT)
        .build()
        .map_err(Error::Http)
}

/// Unix seconds.
pub fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// The tracker's synchronous publisher: it blocks the frame thread on the async
/// write, which the client bounds to `TIMEOUT`.
pub struct SocketPublisher {
    pusher: Arc<Pusher>,
    handle: tokio::runtime::Handle,
}

impl SocketPublisher {
    pub fn new(pusher: Arc<Pusher>, handle: tokio::runtime::Handle) -> SocketPublisher {
        SocketPublisher { pusher, handle }
    }
}

impl crate::tracker::Publisher for SocketPublisher {
    fn enabled(&self) -> bool {
        self.pusher.config.enabled()
    }

    fn publish(
        &self,
        map_number: i64,
        score: &ScoreMap,
        teams: &[Team],
        blue: &str,
        status: Status,
        pause: Option<&Pause>,
    ) -> Result<(), PushError> {
        self.handle
            .block_on(
                self.pusher
                    .publish(map_number, score, teams, blue, status, pause),
            )
            .map_err(|error| PushError(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dotenv_parses_quotes_comments_and_export() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("vbt-dotenv-{}.env", std::process::id()));
        std::fs::write(
            &path,
            "# comment\nexport A=\"/run/x.sock\"\nB=plain # trailing\nC='single'\nD=\n",
        )
        .unwrap();
        let values = read_dotenv(&path);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(values.get("A").map(String::as_str), Some("/run/x.sock"));
        assert_eq!(values.get("B").map(String::as_str), Some("plain"));
        assert_eq!(values.get("C").map(String::as_str), Some("single"));
        assert_eq!(values.get("D").map(String::as_str), Some(""));
    }

    #[test]
    fn tokens_are_unpadded_url_base64() {
        let token = new_token().unwrap();
        let raw = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(&token);
        assert!(raw.is_ok());
        assert_eq!(raw.unwrap().len(), TOKEN_BYTES);
        assert!(!token.contains('='));
    }

    // The backend's codes path always answers `"map_number": null`; an `i64` field
    // made every context decode fail, so the veto names never arrived.
    #[test]
    fn context_accepts_null_map_number() {
        let got: Context = serde_json::from_str(
            r#"{"match_id":"123","map_number":null,"map_order":["Ascent",""],"teams":["VIT","LOUD"]}"#,
        )
        .unwrap();
        assert_eq!(got.map_order, vec!["Ascent".to_string(), String::new()]);
    }
}
