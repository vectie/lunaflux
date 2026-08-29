#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL
umask 077

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/benchmark-campaign-common.sh"

[ "$#" -eq 6 ] || {
  printf '%s\n' \
    'usage: verify-openai-comparison-campaign.sh ABSOLUTE_CAMPAIGN CAMPAIGN_TOOL#sha256=HEX TRIAL_DRIVER#sha256=HEX CORRECTNESS_VERIFIER#sha256=HEX ENGINE_IDENTITY_VERIFIER#sha256=HEX PROCESS_SUPERVISOR#sha256=HEX' >&2
  exit 2
}
root=$1
campaign_argument=$2
driver_argument=$3
correctness_argument=$4
identity_argument=$5
supervisor_argument=$6
for argument in "$campaign_argument" "$driver_argument" "$correctness_argument" \
  "$identity_argument" "$supervisor_argument"; do
  case "$argument" in
    /*#sha256=*) ;;
    *) benchmark_campaign_fail 'verification tool is not digest suffixed' ;;
  esac
done
campaign_tool=${campaign_argument%#sha256=*}
campaign_tool_sha=${campaign_argument##*#sha256=}
driver=${driver_argument%#sha256=*}
expected_driver_sha=${driver_argument##*#sha256=}
correctness=${correctness_argument%#sha256=*}
correctness_sha=${correctness_argument##*#sha256=}
identity_verifier=${identity_argument%#sha256=*}
identity_verifier_sha=${identity_argument##*#sha256=}
supervisor=${supervisor_argument%#sha256=*}
supervisor_sha=${supervisor_argument##*#sha256=}
case "$root" in
  /|*//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
    benchmark_campaign_fail 'campaign root is not a safe canonical path'
    ;;
  /*) ;;
  *) benchmark_campaign_fail 'campaign root is not absolute' ;;
esac
[ -d "$root" ] && [ ! -L "$root" ] ||
  benchmark_campaign_fail 'campaign root is not a non-symlink directory'
[ "$(CDPATH= cd -- "$root" && pwd -P)" = "$root" ] ||
  benchmark_campaign_fail 'campaign root contains a directory alias'
for coordinate in \
  "$campaign_tool:$campaign_tool_sha:campaign-tool" \
  "$driver:$expected_driver_sha:trial-driver" \
  "$correctness:$correctness_sha:correctness-verifier" \
  "$identity_verifier:$identity_verifier_sha:engine-identity-verifier" \
  "$supervisor:$supervisor_sha:process-supervisor"; do
  path=${coordinate%%:*}
  tail=${coordinate#*:}
  digest=${tail%%:*}
  label=${tail#*:}
  benchmark_campaign_require_file "$path" "$label"
  benchmark_campaign_require_digest "$path" "$digest" "$label"
  [ -x "$path" ] || benchmark_campaign_fail "$label is not executable"
done

BENCHMARK_CAMPAIGN_SCRATCH=$(mktemp -d /tmp/lunaflux-openai-campaign-verify.XXXXXX) ||
  benchmark_campaign_fail 'could not create verifier scratch'
BENCHMARK_CAMPAIGN_SCRATCH=$(CDPATH= cd -- "$BENCHMARK_CAMPAIGN_SCRATCH" && pwd -P)
export BENCHMARK_CAMPAIGN_SCRATCH
BENCHMARK_CAMPAIGN_INVOCATION_SCRATCH=$BENCHMARK_CAMPAIGN_SCRATCH
export BENCHMARK_CAMPAIGN_INVOCATION_SCRATCH
trap 'rm -rf "$BENCHMARK_CAMPAIGN_SCRATCH"' EXIT HUP INT TERM
for coordinate in \
  "$campaign_tool:$campaign_tool_sha:campaign-tool" \
  "$driver:$expected_driver_sha:trial-driver" \
  "$correctness:$correctness_sha:correctness-verifier" \
  "$identity_verifier:$identity_verifier_sha:engine-identity-verifier" \
  "$supervisor:$supervisor_sha:process-supervisor"; do
  source=${coordinate%%:*}
  tail=${coordinate#*:}
  digest=${tail%%:*}
  name=${tail#*:}
  cp "$source" "$BENCHMARK_CAMPAIGN_SCRATCH/$name"
  chmod 500 "$BENCHMARK_CAMPAIGN_SCRATCH/$name"
  benchmark_campaign_require_digest "$BENCHMARK_CAMPAIGN_SCRATCH/$name" \
    "$digest" "private $name"
done

if find "$root" -type l -print | grep -q .; then
  benchmark_campaign_fail 'campaign contains a symbolic link'
fi
if find "$root" ! -type d ! -type f -print | grep -q .; then
  benchmark_campaign_fail 'campaign contains a special filesystem object'
fi
for directory in $(find "$root" -type d -print); do
  [ "$(benchmark_campaign_mode "$directory")" = 555 ] ||
    benchmark_campaign_fail 'campaign directory is not mode 555'
done
for file in $(find "$root" -type f -print); do
  [ "$(benchmark_campaign_mode "$file")" = 444 ] ||
    benchmark_campaign_fail 'campaign file is not mode 444'
  [ "$(benchmark_campaign_links "$file")" = 1 ] ||
    benchmark_campaign_fail 'campaign file has a hard-link alias'
done

handoff=$root/comparison-handoff.v1
[ "$(wc -l < "$handoff" | tr -d ' ')" -eq 25 ] ||
  benchmark_campaign_fail 'comparison handoff has the wrong shape'
[ "$(benchmark_campaign_field "$handoff" 1 schema)" = \
    'lunaflux.openai-comparison-handoff.v1' ] ||
  benchmark_campaign_fail 'comparison handoff schema is invalid'
declaration_sha=$(benchmark_campaign_field "$handoff" 2 declaration_sha256)
recipe_sha=$(benchmark_campaign_field "$handoff" 3 recipe_sha256)
source_identity=$(benchmark_campaign_field "$handoff" 4 source_identity_sha256)
recorded_campaign_tool=$(benchmark_campaign_field "$handoff" 5 campaign_tool_sha256)
driver_sha=$(benchmark_campaign_field "$handoff" 6 trial_driver_sha256)
recorded_correctness=$(benchmark_campaign_field "$handoff" 7 correctness_verifier_sha256)
recorded_identity=$(benchmark_campaign_field "$handoff" 8 engine_identity_verifier_sha256)
recorded_supervisor=$(benchmark_campaign_field "$handoff" 9 process_supervisor_sha256)
for digest in "$declaration_sha" "$recipe_sha" "$source_identity" \
  "$recorded_campaign_tool" "$driver_sha" "$recorded_correctness" \
  "$recorded_identity" "$recorded_supervisor"; do
  benchmark_campaign_is_sha256 "$digest" ||
    benchmark_campaign_fail 'comparison handoff contains an invalid identity'
done
[ "$recorded_campaign_tool" = "$campaign_tool_sha" ] &&
  [ "$driver_sha" = "$expected_driver_sha" ] &&
  [ "$recorded_correctness" = "$correctness_sha" ] &&
  [ "$recorded_identity" = "$identity_verifier_sha" ] &&
  [ "$recorded_supervisor" = "$supervisor_sha" ] ||
  benchmark_campaign_fail 'independently supplied verification tool identity changed'
benchmark_campaign_require_digest "$root/campaign.declaration.json" \
  "$declaration_sha" 'campaign declaration'
benchmark_campaign_require_digest "$root/campaign.recipe.v1" "$recipe_sha" 'campaign recipe'
trial_timeout=$(benchmark_campaign_field "$root/campaign.recipe.v1" 15 \
  trial_timeout_seconds)
tool_timeout=$(benchmark_campaign_field "$root/campaign.recipe.v1" 16 \
  tool_timeout_seconds)
grace=$(benchmark_campaign_field "$root/campaign.recipe.v1" 17 \
  cancellation_grace_seconds)
benchmark_campaign_require_digest "$root/tools/campaign-tool" "$campaign_tool_sha" \
  'recorded campaign tool'
benchmark_campaign_require_digest "$root/tools/trial-driver" "$driver_sha" \
  'recorded trial driver'
benchmark_campaign_require_digest "$root/tools/correctness-verifier" \
  "$correctness_sha" 'recorded correctness verifier'
benchmark_campaign_require_digest "$root/tools/engine-identity-verifier" \
  "$identity_verifier_sha" 'recorded engine identity verifier'
benchmark_campaign_require_digest "$root/tools/process-supervisor" \
  "$supervisor_sha" 'recorded process supervisor'

capture_paths=$BENCHMARK_CAMPAIGN_SCRATCH/capture-paths
correctness_paths=$BENCHMARK_CAMPAIGN_SCRATCH/correctness-paths
identity_paths=$BENCHMARK_CAMPAIGN_SCRATCH/identity-paths
process_paths=$BENCHMARK_CAMPAIGN_SCRATCH/process-paths
tool_paths=$BENCHMARK_CAMPAIGN_SCRATCH/tool-paths
find "$root" -maxdepth 1 -type f -name 'trial-*.capture.json' -print |
  sed "s#^$root/##" | LC_ALL=C sort > "$capture_paths"
find "$root/correctness" -type f -print | sed "s#^$root/##" |
  LC_ALL=C sort > "$correctness_paths"
find "$root/identity" -type f -print | sed "s#^$root/##" |
  LC_ALL=C sort > "$identity_paths"
find "$root/process" -type f -print | sed "s#^$root/##" |
  LC_ALL=C sort > "$process_paths"
find "$root/tools" -type f -print | sed "s#^$root/##" |
  LC_ALL=C sort > "$tool_paths"
[ "$(wc -l < "$capture_paths" | tr -d ' ')" -eq 81 ] &&
  [ "$(wc -l < "$correctness_paths" | tr -d ' ')" -eq 81 ] &&
  [ "$(wc -l < "$identity_paths" | tr -d ' ')" -eq 81 ] &&
  [ "$(wc -l < "$process_paths" | tr -d ' ')" -eq 652 ] ||
  benchmark_campaign_fail 'campaign matrix is incomplete'
expected_process_paths=$BENCHMARK_CAMPAIGN_SCRATCH/process-paths.expected
{
  printf '%s\n' process/preflight.invocation.v1 process/preflight.supervisor.v1
  for index in $(awk 'BEGIN { for (i = 0; i < 81; i += 1) print i }'); do
    for label in identity trial identity-post correctness; do
      printf 'process/%s-%s.invocation.v1\n' "$label" "$index"
      printf 'process/%s-%s.supervisor.v1\n' "$label" "$index"
    done
  done
  printf '%s\n' process/replay.invocation.v1 process/replay.supervisor.v1
} | LC_ALL=C sort > "$expected_process_paths"
cmp -s "$process_paths" "$expected_process_paths" ||
  benchmark_campaign_fail 'campaign process evidence path set is not exact'
benchmark_campaign_validate_inventory "$root" "$root/captures.files.sha256" \
  "$capture_paths"
benchmark_campaign_validate_inventory "$root" "$root/correctness.files.sha256" \
  "$correctness_paths"
benchmark_campaign_validate_inventory "$root" "$root/identity.files.sha256" \
  "$identity_paths"
benchmark_campaign_validate_inventory "$root" "$root/process.files.sha256" \
  "$process_paths"
benchmark_campaign_validate_inventory "$root" "$root/tools.files.sha256" \
  "$tool_paths"

all_paths=$BENCHMARK_CAMPAIGN_SCRATCH/all-paths
find "$root" -type f ! -path "$root/campaign.files.sha256" -print |
  sed "s#^$root/##" | LC_ALL=C sort > "$all_paths"
benchmark_campaign_validate_inventory "$root" "$root/campaign.files.sha256" \
  "$all_paths"
for coordinate in \
  "10:capture_inventory_sha256:captures.files.sha256" \
  "11:correctness_inventory_sha256:correctness.files.sha256" \
  "12:identity_inventory_sha256:identity.files.sha256" \
  "13:process_inventory_sha256:process.files.sha256" \
  "14:tool_inventory_sha256:tools.files.sha256" \
  "15:execution_order_sha256:execution-order.csv"; do
  line=${coordinate%%:*}
  tail=${coordinate#*:}
  key=${tail%%:*}
  relative=${tail#*:}
  expected=$(benchmark_campaign_field "$handoff" "$line" "$key")
  benchmark_campaign_require_digest "$root/$relative" "$expected" "$key"
done
[ "$(benchmark_campaign_field "$handoff" 17 trial_count)" = 81 ] &&
  [ "$(benchmark_campaign_field "$handoff" 18 correctness_artifact_count)" = 81 ] &&
  [ "$(benchmark_campaign_field "$handoff" 19 identity_artifact_count)" = 81 ] &&
  [ "$(benchmark_campaign_field "$handoff" 20 counterbalance)" = latin-square-v1 ] &&
  [ "$(benchmark_campaign_field "$handoff" 21 protocol)" = openai.responses.sse.v1 ] &&
  [ "$(benchmark_campaign_field "$handoff" 22 process_cleanup_complete)" = 1 ] &&
  [ "$(benchmark_campaign_field "$handoff" 23 comparison_admission)" = \
    external-correctness-join-required ] &&
  [ "$(benchmark_campaign_field "$handoff" 24 comparison_authority)" = none ] &&
  [ "$(benchmark_campaign_field "$handoff" 25 physical_measurement_claim)" = none ] ||
  benchmark_campaign_fail 'comparison handoff overclaims authority'

receipt_identities=$BENCHMARK_CAMPAIGN_SCRATCH/receipt-identities
: > "$receipt_identities"
while IFS= read -r relative; do
  case "$relative" in *.supervisor.v1) ;; *) continue ;; esac
  receipt=$root/$relative
  receipt_invocation=$(benchmark_campaign_field "$receipt" 2 invocation_sha256)
  label=${relative#process/}
  label=${label%.supervisor.v1}
  invocation=$root/${relative%.supervisor.v1}.invocation.v1
  expected_timeout=$tool_timeout
  expected_scope=none
  case "$label" in
    trial-*)
      trial_index=${label#trial-}
      expected_timeout=$trial_timeout
      expected_scope=fd-$((trial_index % 3 + 3))
      ;;
  esac
  benchmark_campaign_validate_invocation "$invocation" "$receipt_invocation" \
    "$label" "$expected_timeout" "$grace" "$expected_scope"
  benchmark_campaign_validate_supervisor_receipt "$receipt" \
    "$receipt_invocation" "$label"
  if grep -F -x "$receipt_invocation" "$receipt_identities" >/dev/null 2>&1; then
    benchmark_campaign_fail 'stored process cleanup receipt was replayed'
  fi
  printf '%s\n' "$receipt_invocation" >> "$receipt_identities"
done < "$process_paths"

preflight=$root/preflight.record.v1
preflight_value() {
  key=$1
  value=$(sed -n "s/^${key}=//p" "$preflight")
  [ -n "$value" ] && [ "$(grep -c "^${key}=" "$preflight")" -eq 1 ] ||
    benchmark_campaign_fail "preflight identity is missing: $key"
  printf '%s\n' "$value"
}
expected_order=$BENCHMARK_CAMPAIGN_SCRATCH/execution-order.expected
: > "$expected_order"
for profile_index in 0 1 2 3 4 5 6 7 8; do
  profile=$(benchmark_campaign_name "$profile_index")
  for ordinal in 1 2 3; do
    for position in 0 1 2; do
      engine_index=$(benchmark_engine_for_position "$ordinal" "$position")
      engine=$(benchmark_engine_name "$engine_index")
      index=$((profile_index * 9 + (ordinal - 1) * 3 + engine_index))
      printf '%s,%s,%s,%s,%s\n' \
        "$profile" "$ordinal" "$position" "$engine" "$index" \
        >> "$expected_order"
    done
  done
done
cmp -s "$expected_order" "$root/execution-order.csv" ||
  benchmark_campaign_fail 'execution order is not the fixed counterbalanced matrix'

for index in $(awk 'BEGIN { for (i = 0; i < 81; i += 1) print i }'); do
  engine_index=$((index % 3))
  engine=$(benchmark_engine_name "$engine_index")
  identity=$root/identity/trial-$index.identity.v1
  revision=$(preflight_value engine_${engine_index}_revision_sha256)
  image=$(preflight_value engine_${engine_index}_image_sha256)
  configuration=$(preflight_value engine_${engine_index}_configuration_sha256)
  executable_identity=$(preflight_value engine_${engine_index}_executable_sha256)
  endpoint=$(preflight_value endpoint_$engine_index)
  [ "$(wc -l < "$identity" | tr -d ' ')" -eq 10 ] &&
    [ "$(benchmark_campaign_field "$identity" 1 schema)" = \
      lunaflux.live-engine-identity.v1 ] &&
    [ "$(benchmark_campaign_field "$identity" 2 declaration_sha256)" = \
      "$declaration_sha" ] &&
    [ "$(benchmark_campaign_field "$identity" 3 trial_index)" = "$index" ] &&
    [ "$(benchmark_campaign_field "$identity" 4 engine)" = "$engine" ] &&
    [ "$(benchmark_campaign_field "$identity" 5 endpoint)" = "$endpoint" ] &&
    [ "$(benchmark_campaign_field "$identity" 6 revision_sha256)" = "$revision" ] &&
    [ "$(benchmark_campaign_field "$identity" 7 image_sha256)" = "$image" ] &&
    [ "$(benchmark_campaign_field "$identity" 8 configuration_sha256)" = \
      "$configuration" ] &&
    [ "$(benchmark_campaign_field "$identity" 9 executable_sha256)" = \
      "$executable_identity" ] &&
    [ "$(benchmark_campaign_field "$identity" 10 live_identity_verified)" = 1 ] ||
    benchmark_campaign_fail "live engine identity artifact changed: trial-$index"
done

run_verified() {
  label=$1
  stdout=$2
  stderr=$3
  receipt=$4
  shift 4
  invocation=$BENCHMARK_CAMPAIGN_SCRATCH/verify-$label.invocation.v1
  benchmark_campaign_write_invocation \
    "$invocation" "verify-$label" 86400 30 none "$@"
  invocation_sha=$(benchmark_campaign_sha256 "$invocation")
  env -i LC_ALL=C PATH=/usr/bin:/bin \
    "$BENCHMARK_CAMPAIGN_SCRATCH/process-supervisor" run 86400 30 \
    "$invocation" "$invocation_sha" "$stdout" "$stderr" "$receipt" -- "$@" \
    3<&- 4<&- 5<&- ||
    benchmark_campaign_fail "verification process failed: $label"
  [ ! -s "$stderr" ] || benchmark_campaign_fail "$label verification wrote stderr"
  [ "$(wc -l < "$receipt" | tr -d ' ')" -eq 9 ] &&
    [ "$(benchmark_campaign_field "$receipt" 1 schema)" = \
      lunaflux.external-process-supervisor.v2 ] &&
    [ "$(benchmark_campaign_field "$receipt" 2 invocation_sha256)" = \
      "$invocation_sha" ] &&
    [ "$(benchmark_campaign_field "$receipt" 3 outcome)" = completed ] &&
    [ "$(benchmark_campaign_field "$receipt" 4 exit_status)" = 0 ] &&
    [ "$(benchmark_campaign_field "$receipt" 5 timed_out)" = 0 ] &&
    [ "$(benchmark_campaign_field "$receipt" 6 cancelled)" = 0 ] &&
    [ "$(benchmark_campaign_field "$receipt" 7 process_group_empty)" = 1 ] &&
    [ "$(benchmark_campaign_field "$receipt" 8 stdout_closed)" = 1 ] &&
    [ "$(benchmark_campaign_field "$receipt" 9 stderr_closed)" = 1 ] ||
    benchmark_campaign_fail "$label verification cleanup is incomplete"
}

run_verified preflight "$BENCHMARK_CAMPAIGN_SCRATCH/preflight.stdout" \
  "$BENCHMARK_CAMPAIGN_SCRATCH/preflight.stderr" \
  "$BENCHMARK_CAMPAIGN_SCRATCH/preflight.receipt" \
  "$BENCHMARK_CAMPAIGN_SCRATCH/campaign-tool" --preflight \
  "$root/campaign.declaration.json" "$declaration_sha"
cmp -s "$BENCHMARK_CAMPAIGN_SCRATCH/preflight.stdout" "$preflight" ||
  benchmark_campaign_fail 'offline preflight record changed'

for index in $(awk 'BEGIN { for (i = 0; i < 81; i += 1) print i }'); do
  capture_name=$(benchmark_capture_name "$index")
  capture=$root/$capture_name
  capture_sha=$(benchmark_campaign_sha256 "$capture")
  regenerated=$BENCHMARK_CAMPAIGN_SCRATCH/correctness-$index.v1
  run_verified "correctness-$index" \
    "$BENCHMARK_CAMPAIGN_SCRATCH/correctness-$index.stdout" \
    "$BENCHMARK_CAMPAIGN_SCRATCH/correctness-$index.stderr" \
    "$BENCHMARK_CAMPAIGN_SCRATCH/correctness-$index.receipt" \
    "$BENCHMARK_CAMPAIGN_SCRATCH/correctness-verifier" verify \
    "$root/campaign.declaration.json" "$declaration_sha" "$capture" \
    "$capture_sha" "$index" "$regenerated"
  cmp -s "$regenerated" "$root/correctness/trial-$index.correctness.v1" ||
    benchmark_campaign_fail "correctness artifact replay changed: trial-$index"
done

run_verified replay "$BENCHMARK_CAMPAIGN_SCRATCH/replay.stdout" \
  "$BENCHMARK_CAMPAIGN_SCRATCH/replay.stderr" \
  "$BENCHMARK_CAMPAIGN_SCRATCH/replay.receipt" \
  "$BENCHMARK_CAMPAIGN_SCRATCH/campaign-tool" "$root" "$declaration_sha"
cmp -s "$BENCHMARK_CAMPAIGN_SCRATCH/replay.stdout" "$root/replay.record.v1" ||
  benchmark_campaign_fail 'offline replay record changed'
replay_record_sha=$(benchmark_campaign_field "$handoff" 16 replay_record_sha256)
grep -F -x "replay_record_sha256=$replay_record_sha" "$root/replay.record.v1" \
  >/dev/null 2>&1 || benchmark_campaign_fail 'replay handoff digest changed'

rm -rf "$BENCHMARK_CAMPAIGN_SCRATCH"
trap - EXIT HUP INT TERM
printf '%s\n' 'LunaFlux external-process campaign is sealed and replayable; comparison authority remains external.'
