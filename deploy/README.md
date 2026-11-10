# Deploying with systemd

This directory is an example of running vbt as a live writer under a systemd user unit.
Everything lives in one install directory, `~/vbt`; change the paths in `vbt.service` to
use another one.

| Piece | Path |
| --- | --- |
| unit | `~/.config/systemd/user/vbt.service` |
| binary | `~/vbt/vbt` |
| launch scripts | `~/vbt/launch-primary.sh` (live writer), `~/vbt/launch.sh` (shadow) |
| streamlink | `~/vbt/tools/` (a Python venv) |
| ffmpeg | `~/vbt/ffmpeg` (a static build) |
| runtime state | `~/vbt/data/` |

`data/` holds `video-token` (written by each live start, mode 600), `video-token.lock`
(locked by the running live writer), `vbt-events.jsonl`
(the `--events-file` ledger, append-only), `vbt.log` (the writer's stdout and stderr) and
`tag-crops/`, the team tag crops vbt learns at runtime (`--tag-crops`, default
`data/tag-crops` under the working directory). The shadow script writes
`shadow-events.jsonl`, `shadow.log` and its own `shadow-tag-crops/` there too, so a shadow
run never changes the live writer's crops. Deleting either crop directory is safe: the crops
are learned again from the next frames the OCR reads.

## Pinned tools

vbt runs `streamlink` and `ffmpeg` from `PATH`. The launch scripts set `PATH` to the two
pinned binaries and exit if either one resolves anywhere else:

- `~/vbt/tools/bin/streamlink`: streamlink 8.6.2 in a venv. 8.6.2 fixes
  GHSA-j7pf-m7gx-jr33 (local file read via `file://` in DASH `Location` tags), so do not
  use 8.6.1 or older.
- `~/vbt/ffmpeg`: a static ffmpeg build (7.0.2 is tested).

Create `tools/` once. The venv's scripts have absolute shebangs, so moving or recreating
the directory breaks them. Upgrade streamlink in place and restart the unit.

## First install

```sh
mkdir -p ~/vbt ~/.config/systemd/user
cd ~/vbt && uv venv --python 3.11 tools
uv pip install --python tools/bin/python "streamlink==8.6.2"
cp /path/to/static/ffmpeg ~/vbt/ffmpeg
cp deploy/vbt.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable vbt.service
```

Set `BACKEND_SOCKET` in `vbt.service` to the backend's Unix socket. `launch-primary.sh`
exits if it is unset.

## Release

Build in a checkout of this repository, then copy into the install directory:

```sh
git pull
cargo build --release -p vbt

install -m 755 target/release/vbt ~/vbt/vbt
cp deploy/launch-primary.sh deploy/launch.sh ~/vbt/ && chmod 755 ~/vbt/launch-*.sh

systemctl --user restart vbt.service
systemctl --user show vbt.service -p NRestarts   # expect 0
```

A release does not touch `tools/` or `data/`. No previous binary is kept: to roll back,
check out the revision you want and repeat the release steps.

## Token file

Only one live writer runs per token file. Each live start locks `video-token.lock` for the
life of the process and then writes a fresh `video-token`. A second live writer on the same
file exits at startup with `another live writer holds .../video-token.lock`. The backend's `VIDEO_TOKEN_FILE` must point at the same file as the
tracker's (`~/vbt/data/video-token`).
