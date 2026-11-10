#!/bin/sh
# Downloads the test images, checks the archive's sha256 and unpacks it at the
# repository root. Pass a local copy of the archive to skip the download.
#
#   scripts/fetch-testdata.sh [ARCHIVE]
set -eu

url=https://files.akhilnarang.dev/cdn/valorant/ocr/v1/vbt-testdata-v1.tar.zst
sha256=2416a9eb6670480f3d964de460e49670bc20ab9445c9a94ff343d40a15cc3c8c

root=$(cd "$(dirname "$0")/.." && pwd)
if [ $# -gt 0 ]; then
    archive=$1
else
    archive=$(mktemp)
    trap 'rm -f "$archive"' EXIT
    curl -fL --retry 3 -o "$archive" "$url"
fi
echo "$sha256  $archive" | sha256sum -c --quiet -
tar --zstd -xf "$archive" -C "$root"
