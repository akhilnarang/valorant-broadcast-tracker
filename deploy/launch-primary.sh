#!/bin/bash
# vbt live writer: pinned streamlink -> pinned ffmpeg -> ./vbt, which PUTs scores to the
# backend over BACKEND_SOCKET, authenticating with the token in VIDEO_TOKEN_FILE.
cd "$(dirname "$0")" || exit 1
DIR="$(pwd -P)"

# vbt resolves `streamlink` and `ffmpeg` from PATH, so PATH is pinned to the exact
# absolute binaries and the lookup is asserted: a drift must fail loudly here instead
# of surfacing as a silent outage later.
STREAMLINK_BIN="$DIR/tools/bin/streamlink"
FFMPEG_BIN="$DIR/ffmpeg"
# streamlink ships inside the install directory; abort instead of falling back to a
# host-global tool.
if [ ! -x "$STREAMLINK_BIN" ]; then
  echo "launch-primary: missing $STREAMLINK_BIN; this install has no streamlink env (see deploy/README.md)" >&2
  exit 1
fi
if [ ! -x "$FFMPEG_BIN" ]; then
  echo "launch-primary: missing $FFMPEG_BIN" >&2
  exit 1
fi
export PATH="$(dirname "$STREAMLINK_BIN"):$(dirname "$FFMPEG_BIN"):/usr/local/bin:/usr/bin:/bin"
for tool in "$STREAMLINK_BIN" "$FFMPEG_BIN"; do
  if [ "$(command -v "$(basename "$tool")")" != "$tool" ]; then
    echo "launch-primary: $(basename "$tool") resolves to $(command -v "$(basename "$tool")"), want $tool" >&2
    exit 1
  fi
done

mkdir -p "$DIR/data"
export BACKEND_SOCKET="${BACKEND_SOCKET:?set BACKEND_SOCKET to the backend Unix socket}"
export VIDEO_TOKEN_FILE="$DIR/data/video-token"

exec nice -n 10 taskset -c 0 "$DIR/vbt" \
  --source https://www.twitch.tv/valorant \
  --canonical --verbose \
  --events-file "$DIR/data/vbt-events.jsonl" \
  >> "$DIR/data/vbt.log" 2>&1
