#!/usr/bin/env bash

set -euo pipefail

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
helper=$repo_root/scripts/immutable-evidence-directory.sh

fail() {
  printf 'immutable evidence-directory boundary failed: %s\n' "$1" >&2
  exit 1
}

mode_of() {
  if stat -c '%a' "$1" >/dev/null 2>&1; then
    stat -c '%a' "$1"
  else
    stat -f '%Lp' "$1"
  fi
}

bash -n "$helper" || fail 'helper syntax is invalid'
for anchor in \
  'lunaflux_prepare_evidence_manifest()' \
  'lunaflux_seal_evidence_directory()' \
  '! -path "$evidence_dir/RESULT.txt"' \
  'LC_ALL=C sort -z' \
  'find "$evidence_dir" -type f -exec chmod 0444 {} +' \
  'find "$evidence_dir" -type d -exec chmod 0555 {} +'; do
  grep -Fq "$anchor" "$helper" || fail "helper invariant is missing: $anchor"
done
if sed '/^[[:space:]]*#/d' "$helper" |
  rg -ni 'evidence_schema|outcome=|readiness=|runtime_authority|campaign_complete'; then
  fail 'filesystem helper acquired result-schema or promotion authority'
fi

scratch=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-evidence-helper.XXXXXX")
scratch=$(CDPATH= cd -- "$scratch" && pwd -P)
cleanup() {
  chmod -R u+rwX "$scratch" 2>/dev/null || true
  rm -rf -- "$scratch"
}
trap cleanup EXIT HUP INT TERM

evidence=$scratch/evidence
mkdir -p "$evidence/nested dir"
printf '%s' alpha >"$evidence/a file"
printf '%s' beta >"$evidence/nested dir/b"
printf '%s' tool >"$evidence/tool"
chmod 0755 "$evidence/tool"
printf '%s\n' ignored-result >"$evidence/RESULT.txt"
printf '%s\n' stale-manifest >"$evidence/FILES.sha256"

. "$helper"
lunaflux_prepare_evidence_manifest "$evidence" ||
  fail 'manifest preparation rejected ordinary evidence'
case "$lunaflux_evidence_manifest_sha256" in
  [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*) ;;
  *) fail 'manifest digest was not published' ;;
esac
[ "${#lunaflux_evidence_manifest_sha256}" -eq 64 ] ||
  fail 'manifest digest length is invalid'
if rg -q '(^|  )(FILES\.sha256|RESULT\.txt)$' "$evidence/FILES.sha256"; then
  fail 'manifest included its own file or the caller-owned result'
fi
expected=$scratch/expected.sha256
{
  printf '%s  %s\n' "$(lunaflux_evidence_sha256_file "$evidence/a file")" 'a file'
  printf '%s  %s\n' "$(lunaflux_evidence_sha256_file "$evidence/nested dir/b")" \
    'nested dir/b'
  printf '%s  %s\n' "$(lunaflux_evidence_sha256_file "$evidence/tool")" tool
} >"$expected"
cmp -s "$expected" "$evidence/FILES.sha256" ||
  fail 'manifest contents or bytewise ordering drifted'
[ "$(lunaflux_evidence_sha256_file "$evidence/FILES.sha256")" = \
  "$lunaflux_evidence_manifest_sha256" ] || fail 'manifest digest is inconsistent'

lunaflux_seal_evidence_directory "$evidence" "$evidence/tool" ||
  fail 'ordinary evidence sealing failed'
[ "$(mode_of "$evidence/a file")" = 444 ] || fail 'ordinary file is not mode 0444'
[ "$(mode_of "$evidence/tool")" = 555 ] || fail 'declared executable is not mode 0555'
[ "$(mode_of "$evidence")" = 555 ] || fail 'evidence root is not mode 0555'
[ "$(mode_of "$evidence/nested dir")" = 555 ] ||
  fail 'nested evidence directory is not mode 0555'

hostile=$scratch/hostile
outside=$scratch/outside
mkdir "$hostile"
printf '%s' retained >"$hostile/file"
printf '%s' outside >"$outside"
chmod 0600 "$hostile/file" "$outside"
if lunaflux_seal_evidence_directory "$hostile" "$outside"; then
  fail 'sealer accepted an executable outside the evidence root'
fi
[ "$(mode_of "$hostile/file")" = 600 ] ||
  fail 'rejected outside executable partially mutated evidence'
if lunaflux_seal_evidence_directory "$hostile" "$hostile/../outside"; then
  fail 'sealer accepted a non-canonical executable path'
fi
[ "$(mode_of "$outside")" = 600 ] ||
  fail 'non-canonical executable path escaped evidence ownership'

printf '%s\n' 'Immutable evidence-directory boundary passed.'
