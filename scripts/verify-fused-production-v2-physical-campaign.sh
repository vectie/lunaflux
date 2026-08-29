#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
export LC_ALL
fail() { printf 'fused production V2 evidence rejected: %s\n' "$1" >&2; exit 1; }
[[ $# == 3 ]] || { printf 'usage: %s ABSOLUTE_EVIDENCE_DIR EXPECTED_OUTER_SHA256 EXPECTED_POLICY_SHA256\n' "$0" >&2; exit 2; }
evidence=$1
expected_outer=$2
expected_policy=$3
[[ $expected_outer =~ ^[0-9a-f]{64}$ && $expected_policy =~ ^[0-9a-f]{64}$ ]] || fail 'expected digest is malformed'
[[ $evidence == /* && -d $evidence && ! -L $evidence && $(realpath -- "$evidence") == "$evidence" ]] || fail 'evidence is not an absolute canonical directory'
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
mode_of() { if stat -f '%Lp' "$1" >/dev/null 2>&1; then stat -f '%Lp' "$1"; else stat -c '%a' "$1"; fi; }
files=$evidence/FILES.sha256
result=$evidence/RESULT.txt
outer=$evidence/OUTER_SEAL.sha256
for path in "$files" "$result" "$outer"; do [[ -f $path && ! -L $path ]] || fail 'seal transaction file is absent or aliased'; done
[[ $(sha256_file "$outer") == "$expected_outer" ]] || fail 'outer seal digest differs from independent pin'
[[ $(wc -l <"$outer" | tr -d ' ') == 2 ]] || fail 'outer seal is not exactly two lines'
files_sha=$(sed -n '1s/^\([0-9a-f]\{64\}\)  FILES\.sha256$/\1/p' "$outer")
result_sha=$(sed -n '2s/^\([0-9a-f]\{64\}\)  RESULT\.txt$/\1/p' "$outer")
[[ $files_sha =~ ^[0-9a-f]{64}$ && $result_sha =~ ^[0-9a-f]{64}$ ]] || fail 'outer seal schema drifted'
[[ $(sha256_file "$files") == "$files_sha" && $(sha256_file "$result") == "$result_sha" ]] || fail 'outer-sealed file digest mismatch'
[[ -z $(find "$evidence" -type l -print -quit) ]] || fail 'evidence contains a symlink'
tmp=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-fused-v2-verify.XXXXXX")
trap 'rm -rf -- "$tmp"' EXIT
: >"$tmp/manifest-paths"
previous=
while IFS= read -r line; do
  digest=${line%%  *}
  relative=${line#*  }
  [[ $digest =~ ^[0-9a-f]{64}$ && $line == "$digest  $relative" && $relative =~ ^[A-Za-z0-9._/-]+$ ]] || fail 'FILES entry is malformed'
  [[ $relative != /* && $relative != *'//'* && $relative != '..' && $relative != ../* && $relative != */../* && $relative != */.. ]] || fail 'FILES path escapes evidence root'
  [[ -z $previous || $previous < $relative ]] || fail 'FILES paths are duplicated or unsorted'
  path=$evidence/$relative
  [[ -f $path && ! -L $path && $(sha256_file "$path") == "$digest" ]] || fail "FILES digest mismatch: $relative"
  printf '%s\n' "$relative" >>"$tmp/manifest-paths"
  previous=$relative
done <"$files"
(cd "$evidence" && find . -type f ! -name FILES.sha256 ! -name RESULT.txt ! -name OUTER_SEAL.sha256 -print | sed 's#^./##' | LC_ALL=C sort) >"$tmp/actual-paths"
cmp -s "$tmp/manifest-paths" "$tmp/actual-paths" || fail 'FILES inventory is incomplete or contains an extra path'
while IFS= read -r path; do [[ $(mode_of "$evidence/$path") == 444 ]] || fail "evidence file is writable: $path"; done <"$tmp/actual-paths"
[[ $(mode_of "$files") == 444 && $(mode_of "$result") == 444 && $(mode_of "$outer") == 444 && $(mode_of "$evidence") == 555 ]] || fail 'sealed transaction modes drifted'
field() {
  local count value
  count=$(grep -c "^$1=" "$result")
  [[ $count == 1 ]] || fail "RESULT $1 is absent or duplicated"
  value=$(sed -n "s/^$1=//p" "$result")
  [[ -n $value ]] || fail "RESULT $1 is empty"
  printf '%s\n' "$value"
}
join=$evidence/artifacts/SPAWN_JOIN.v1
[[ -f $join && ! -L $join && $(sha256_file "$join") == $(field spawn_join_sha256) ]] ||
  fail 'spawn join record is absent or substituted'
join_field() {
  local count value
  count=$(grep -c "^$1=" "$join")
  [[ $count == 1 ]] || fail "spawn join field is absent or duplicated: $1"
  value=$(sed -n "s/^$1=//p" "$join")
  [[ -n $value ]] || fail "spawn join field is empty: $1"
  printf '%s\n' "$value"
}
for family in qkv readonly residual; do
  relative=$(join_field "${family}_artifact_relative")
  digest=$(join_field "${family}_module_sha256")
  [[ $relative == artifacts/builds/$family-first/kernels/sha256/$digest.cubin && \
    $digest =~ ^[0-9a-f]{64}$ && -f $evidence/$relative && \
    $(sha256_file "$evidence/$relative") == "$digest" ]] ||
    fail "$family spawn join artifact is not exact"
  [[ $(join_field "${family}_source_sha256") =~ ^[0-9a-f]{64}$ && \
    -n $(join_field "${family}_symbol") ]] || fail "$family spawn join identity is invalid"
done
[[ $(join_field schema) == lunaflux-fused-production-v2-spawn-join.v1 && \
  $(join_field aggregate_runtime) == externally-prepared-required && \
  $(join_field normal_spawn_admission) == not-claimed && \
  $(join_field qualification_only) == true && \
  $(join_field promotion_authority) == absent ]] || fail 'spawn join authority drifted'
[[ $(field schema) == lunaflux-fused-production-v2-physical-campaign.v1 &&
   $(field outcome) == fused-production-v2-physical-campaign-pass &&
   $(field target) == sm_120 && $(field compiler_version) == 13.1.115 &&
   $(field approved_policy_sha256) == "$expected_policy" &&
   $(field production_abis) == qkv-v2,readonly-attention-v2,residual-rmsnorm-v2 &&
   $(field executor) == ordered-kernel-executor &&
   $(field graph_policy) == capture-required && $(field graph_mode) == captured &&
   $(field oracle) == standalone-cuda && $(field memcheck_errors) == 0 &&
   $(field racecheck_errors) == 0 && $(field initcheck_errors) == 0 &&
   $(field physical_cuda_observed) == true && $(field qualification_only) == true &&
   $(field manifest_bindable) == false && $(field promotion_authority) == absent &&
   $(field evidence_files_manifest_sha256) == "$files_sha" ]] || fail 'RESULT semantic contract drifted'
printf 'outcome=fused-production-v2-physical-campaign-verified outer_seal_sha256=%s authority=qualification-only promotion_authority=absent\n' "$expected_outer"
