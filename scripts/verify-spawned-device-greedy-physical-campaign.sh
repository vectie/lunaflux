#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
export LC_ALL
fail() { printf 'spawned device-greedy evidence rejected: %s\n' "$1" >&2; exit 1; }
[[ $# == 2 ]] || { printf 'usage: %s ABSOLUTE_EVIDENCE EXPECTED_OUTER_SHA256\n' "$0" >&2; exit 2; }
evidence=$1
expected_outer=$2
[[ $expected_outer =~ ^[0-9a-f]{64}$ ]] || fail 'outer pin is malformed'
[[ $evidence == /* && -d $evidence && ! -L $evidence && \
  $(realpath -- "$evidence") == "$evidence" ]] || fail 'evidence path is not canonical'
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
mode_of() {
  if stat -f '%Lp' "$1" >/dev/null 2>&1; then stat -f '%Lp' "$1"
  else stat -c '%a' "$1"; fi
}
files=$evidence/FILES.sha256
result=$evidence/RESULT.txt
outer=$evidence/OUTER_SEAL.sha256
for path in "$files" "$result" "$outer"; do
  [[ -f $path && ! -L $path ]] || fail 'seal transaction file is missing or aliased'
done
[[ $(sha256_file "$outer") == "$expected_outer" ]] || fail 'outer pin mismatch'
[[ $(wc -l <"$outer" | tr -d ' ') == 2 ]] || fail 'outer seal shape drifted'
files_sha=$(sed -n '1s/^\([0-9a-f]\{64\}\)  FILES\.sha256$/\1/p' "$outer")
result_sha=$(sed -n '2s/^\([0-9a-f]\{64\}\)  RESULT\.txt$/\1/p' "$outer")
[[ $files_sha =~ ^[0-9a-f]{64}$ && $result_sha =~ ^[0-9a-f]{64}$ && \
  $(sha256_file "$files") == "$files_sha" && \
  $(sha256_file "$result") == "$result_sha" ]] || fail 'outer-sealed digest mismatch'
[[ -z $(find "$evidence" -type l -print -quit) ]] || fail 'evidence contains a symlink'
tmp=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-device-greedy-verify.XXXXXX")
trap 'rm -rf -- "$tmp"' EXIT
: >"$tmp/declared"
previous=
while IFS= read -r line; do
  digest=${line%%  *}
  relative=${line#*  }
  [[ $digest =~ ^[0-9a-f]{64}$ && $line == "$digest  $relative" && \
    $relative =~ ^[A-Za-z0-9._%/-]+$ ]] || fail 'FILES entry is malformed'
  [[ $relative != /* && $relative != *'//'* && $relative != ../* && \
    $relative != */../* ]] || fail 'FILES path escapes evidence'
  [[ -z $previous || $previous < $relative ]] || fail 'FILES is not ordered and unique'
  path=$evidence/$relative
  [[ -f $path && ! -L $path && $(sha256_file "$path") == "$digest" ]] ||
    fail "FILES digest mismatch: $relative"
  [[ $(mode_of "$path") == 444 ]] || fail "evidence file is writable: $relative"
  printf '%s\n' "$relative" >>"$tmp/declared"
  previous=$relative
done <"$files"
(cd "$evidence" && find . -type f ! -name FILES.sha256 ! -name RESULT.txt \
  ! -name OUTER_SEAL.sha256 -print | sed 's#^./##' | LC_ALL=C sort) >"$tmp/actual"
cmp -s "$tmp/declared" "$tmp/actual" || fail 'FILES inventory is not exact'
[[ $(mode_of "$files") == 444 && $(mode_of "$result") == 444 && \
  $(mode_of "$outer") == 444 && $(mode_of "$evidence") == 555 ]] ||
  fail 'immutable evidence modes drifted'
field() {
  local count value
  count=$(grep -c "^$1=" "$result")
  [[ $count == 1 ]] || fail "RESULT field is absent or duplicated: $1"
  value=$(sed -n "s/^$1=//p" "$result")
  [[ -n $value ]] || fail "RESULT field is empty: $1"
  printf '%s\n' "$value"
}
fused=$(field fused_v2_runtime)
[[ $fused == optional-absent || $fused == required-present ]] || fail 'fused mode drifted'
[[ $(field schema) == lunaflux-spawned-device-greedy-physical-campaign.v1 && \
  $(field outcome) == spawned-device-greedy-physical-pass && \
  $(field request_plans) == 2 && $(field readback_bytes) == 16 && \
  $(field host_referee) == full-logits-production-route && \
  $(field memcheck_errors) == 0 && $(field racecheck_errors) == 0 && \
  $(field initcheck_errors) == 0 && $(field resources) == closed && \
  $(field adversarial_tie_policy) == source-and-host-contract-only && \
  $(field adversarial_nonfinite_policy) == source-and-host-contract-only && \
  $(field physical_cuda_observed) == true && $(field qualification_only) == true && \
  $(field promotion_authority) == absent && \
  $(field evidence_files_manifest_sha256) == "$files_sha" ]] ||
  fail 'RESULT semantic contract drifted'
printf 'outcome=spawned-device-greedy-physical-campaign-verified outer_seal_sha256=%s authority=qualification-only\n' "$expected_outer"
