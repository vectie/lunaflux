#!/bin/sh

set -eu
LC_ALL=C
TZ=UTC
COPYFILE_DISABLE=1
export LC_ALL TZ COPYFILE_DISABLE
umask 077

fail() {
  printf '%s\n' "LunaFlux portable source archive rejected: $1" >&2
  exit 1
}

usage() {
  printf '%s\n' \
    'usage: create-portable-source-archive.sh ABSOLUTE_NEW_ARCHIVE.tar.gz' >&2
  exit 2
}

[ "$#" -eq 1 ] || usage
output=$1
case "$output" in /*.tar.gz) ;; *) usage ;; esac
[ ! -e "$output" ] && [ ! -L "$output" ] ||
  fail 'refusing to overwrite an existing archive'

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
repo_parent=$(dirname -- "$repo_root")
repo_name=$(basename -- "$repo_root")
output_parent=$(CDPATH= cd -- "$(dirname -- "$output")" && pwd -P)
[ "$output_parent/$(basename -- "$output")" = "$output" ] ||
  fail 'archive path is not canonical'
case "$output" in "$repo_root"|"$repo_root"/*)
  fail 'archive must be outside the source tree'
esac

appledouble=$(find "$repo_root" \
  \( -path "$repo_root/.git" -o -path "$repo_root/_build" -o \
    -path "$repo_root/trace.json" \) -prune -o \
  -name '._*' -print | sed -n '1p')
[ -z "$appledouble" ] || fail "AppleDouble source entry exists: $appledouble"
special=$(find "$repo_root" \
  \( -path "$repo_root/.git" -o -path "$repo_root/_build" -o \
    -path "$repo_root/trace.json" \) -prune -o \
  ! -type d ! -type f ! -type l -print | sed -n '1p')
[ -z "$special" ] || fail "special source entry exists: $special"

scratch=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-portable-source.XXXXXX")
cleanup() { rm -rf -- "$scratch"; }
trap cleanup EXIT HUP INT TERM
archive=$scratch/source.tar.gz
listing=$scratch/source.list

tar --no-xattrs -czf "$archive" \
  --exclude="$repo_name/.git" \
  --exclude="$repo_name/_build" \
  --exclude="$repo_name/trace.json" \
  --exclude='*/.DS_Store' \
  -C "$repo_parent" "$repo_name" || fail 'tar creation failed'
tar -tzf "$archive" >"$listing" || fail 'archive listing failed'
if grep -Eq '(^|/)\._[^/]*$' "$listing"; then
  fail 'archive contains an AppleDouble entry'
fi
[ "$(sed -n '1p' "$listing")" = "$repo_name/" ] ||
  fail 'archive has the wrong top-level source directory'

mv "$archive" "$output"
trap - EXIT HUP INT TERM
rm -rf -- "$scratch"
if command -v sha256sum >/dev/null 2>&1; then
  digest=$(sha256sum "$output" | cut -d ' ' -f 1)
else
  digest=$(shasum -a 256 "$output" | cut -d ' ' -f 1)
fi
printf '%s\n' 'schema=lunaflux-portable-current-source-archive.v1'
printf 'archive=%s\n' "$output"
printf 'source_directory=%s\n' "$repo_name"
printf 'archive_sha256=%s\n' "$digest"
printf '%s\n' 'xattrs=excluded'
printf '%s\n' 'appledouble_entries=0'
