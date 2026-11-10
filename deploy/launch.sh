#!/bin/bash
# vbt shadow reader: reads and logs everything, never writes the token or PUTs scores.
# Same pinned tools as launch-primary.sh; ledger and log live in data/.
cd "$(dirname "$0")" || exit 1
DIR="$(pwd -P)"

STREAMLINK_BIN="$DIR/tools/bin/streamlink"
FFMPEG_BIN="$DIR/ffmpeg"
# streamlink ships inside the install directory; abort instead of falling back to a
# host-global tool.
if [ ! -x "$STREAMLINK_BIN" ]; then
  echo "launch: missing $STREAMLINK_BIN; this install has no streamlink env (see deploy/README.md)" >&2
  exit 1
fi
if [ ! -x "$FFMPEG_BIN" ]; then
  echo "launch: missing $FFMPEG_BIN" >&2
  exit 1
fi
export PATH="$(dirname "$STREAMLINK_BIN"):$(dirname "$FFMPEG_BIN"):/usr/local/bin:/usr/bin:/bin"
for tool in "$STREAMLINK_BIN" "$FFMPEG_BIN"; do
  if [ "$(command -v "$(basename "$tool")")" != "$tool" ]; then
    echo "launch: $(basename "$tool") resolves to $(command -v "$(basename "$tool")"), want $tool" >&2
    exit 1
  fi
done

mkdir -p "$DIR/data"
unset BACKEND_SOCKET VIDEO_TOKEN_FILE
exec nice -n 10 taskset -c 0 "$DIR/vbt" \
  --source https://www.twitch.tv/valorant \
  --shadow --verbose \
  --events-file "$DIR/data/shadow-events.jsonl" \
  --tag-crops "$DIR/data/shadow-tag-crops" \
  >> "$DIR/data/shadow.log" 2>&1
