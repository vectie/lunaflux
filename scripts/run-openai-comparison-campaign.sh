#!/bin/sh
set -eu
LC_ALL=C
export LC_ALL
umask 077
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/benchmark-campaign-common.sh"
[ "$#" -eq 2 ] || {
  printf '%s\n' \
    'usage: run-openai-comparison-campaign.sh ABSOLUTE_RECIPE#sha256=HEX ABSOLUTE_NEW_OUTPUT' >&2
  exit 2
}
recipe_argument=$1
output=$2
case "$recipe_argument" in /*#sha256=*) ;; *) benchmark_campaign_fail 'recipe is not digest suffixed' ;; esac
recipe=${recipe_argument%#sha256=*}
recipe_sha=${recipe_argument##*#sha256=}
benchmark_campaign_require_file "$recipe" 'campaign recipe'
benchmark_campaign_require_digest "$recipe" "$recipe_sha" 'campaign recipe'
benchmark_campaign_require_newline "$recipe" 'campaign recipe'
[ "$(wc -l < "$recipe" | tr -d ' ')" -eq 22 ] ||
  benchmark_campaign_fail 'campaign recipe must contain exactly 22 lines'
[ "$(benchmark_campaign_field "$recipe" 1 schema)" = \
    'lunaflux.openai-external-process-campaign.v1' ] ||
  benchmark_campaign_fail 'campaign recipe schema is invalid'

declaration=$(benchmark_campaign_field "$recipe" 2 declaration_source)
declaration_sha=$(benchmark_campaign_field "$recipe" 3 declaration_sha256)
campaign_tool=$(benchmark_campaign_field "$recipe" 4 campaign_tool_source)
campaign_tool_sha=$(benchmark_campaign_field "$recipe" 5 campaign_tool_sha256)
driver=$(benchmark_campaign_field "$recipe" 6 trial_driver_source)
driver_sha=$(benchmark_campaign_field "$recipe" 7 trial_driver_sha256)
correctness=$(benchmark_campaign_field "$recipe" 8 correctness_verifier_source)
correctness_sha=$(benchmark_campaign_field "$recipe" 9 correctness_verifier_sha256)
identity_verifier=$(benchmark_campaign_field "$recipe" 10 engine_identity_verifier_source)
identity_verifier_sha=$(benchmark_campaign_field "$recipe" 11 engine_identity_verifier_sha256)
supervisor=$(benchmark_campaign_field "$recipe" 12 process_supervisor_source)
supervisor_sha=$(benchmark_campaign_field "$recipe" 13 process_supervisor_sha256)
source_identity=$(benchmark_campaign_field "$recipe" 14 source_identity_sha256)
trial_timeout=$(benchmark_campaign_field "$recipe" 15 trial_timeout_seconds)
tool_timeout=$(benchmark_campaign_field "$recipe" 16 tool_timeout_seconds)
grace=$(benchmark_campaign_field "$recipe" 17 cancellation_grace_seconds)
[ "$(benchmark_campaign_field "$recipe" 18 credential_fds)" = '3,4,5' ] ||
  benchmark_campaign_fail 'credential descriptors are not the fixed inherited set'
[ "$(benchmark_campaign_field "$recipe" 19 protocol)" = \
    'openai.responses.sse.v1' ] || benchmark_campaign_fail 'protocol is not Responses SSE v1'
[ "$(benchmark_campaign_field "$recipe" 20 matrix)" = \
    '3-engines,9-profiles,3-trials' ] || benchmark_campaign_fail 'matrix is incomplete'
[ "$(benchmark_campaign_field "$recipe" 21 counterbalance)" = \
    'latin-square-v1' ] || benchmark_campaign_fail 'counterbalance policy is invalid'
[ "$(benchmark_campaign_field "$recipe" 22 authority)" = \
    'capture-only-no-comparison' ] || benchmark_campaign_fail 'campaign authority is overbroad'
if ! (: <&3) 2>/dev/null || ! (: <&4) 2>/dev/null ||
  ! (: <&5) 2>/dev/null; then
  benchmark_campaign_fail 'one or more inherited credential descriptors are closed'
fi
benchmark_campaign_is_sha256 "$source_identity" ||
  benchmark_campaign_fail 'source identity digest is invalid'
for seconds in "$trial_timeout" "$tool_timeout" "$grace"; do
  case "$seconds" in
    [1-9]|[1-9][0-9]*) ;;
    *) benchmark_campaign_fail 'timeout is not canonical decimal' ;;
  esac
  [ "$seconds" -ge 1 ] && [ "$seconds" -le 86400 ] ||
    benchmark_campaign_fail 'timeout is outside the fixed bound'
done

for coordinate in \
  "$declaration:$declaration_sha:declaration" \
  "$campaign_tool:$campaign_tool_sha:campaign-tool" \
  "$driver:$driver_sha:trial-driver" \
  "$correctness:$correctness_sha:correctness-verifier" \
  "$identity_verifier:$identity_verifier_sha:engine-identity-verifier" \
  "$supervisor:$supervisor_sha:process-supervisor"; do
  coordinate_path=${coordinate%%:*}
  coordinate_tail=${coordinate#*:}
  coordinate_sha=${coordinate_tail%%:*}
  coordinate_label=${coordinate_tail#*:}
  benchmark_campaign_require_file "$coordinate_path" "$coordinate_label"
  benchmark_campaign_require_digest "$coordinate_path" "$coordinate_sha" "$coordinate_label"
done
for executable in "$campaign_tool" "$driver" "$correctness" \
  "$identity_verifier" "$supervisor"; do
  [ -x "$executable" ] || benchmark_campaign_fail 'campaign tool is not executable'
done

case "$output" in
  /|*//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
    benchmark_campaign_fail 'output path is not a safe canonical absolute path'
    ;;
  /*) ;;
  *) benchmark_campaign_fail 'output path is not absolute' ;;
esac
[ ! -e "$output" ] || benchmark_campaign_fail 'refusing to overwrite an existing output'
output_parent=$(CDPATH= cd -- "$(dirname -- "$output")" && pwd -P)
[ "$output_parent/$(basename -- "$output")" = "$output" ] ||
  benchmark_campaign_fail 'output parent is not canonical'

stage=$(mktemp -d "$output_parent/.lunaflux-openai-campaign-stage.XXXXXX") ||
  benchmark_campaign_fail 'could not create output-adjacent campaign stage'
scratch=$(mktemp -d /tmp/lunaflux-openai-campaign.XXXXXX) || {
  rmdir "$stage"
  benchmark_campaign_fail 'could not create private campaign scratch'
}
scratch=$(CDPATH= cd -- "$scratch" && pwd -P)
BENCHMARK_CAMPAIGN_INVOCATION_SCRATCH=$scratch
export BENCHMARK_CAMPAIGN_INVOCATION_SCRATCH
claimed_output=
cleanup() {
  chmod -R u+w "$stage" "$scratch" 2>/dev/null || true
  rm -rf "$stage" "$scratch"
  if [ -n "$claimed_output" ]; then
    chmod -R u+w "$claimed_output" 2>/dev/null || true
    rm -rf "$claimed_output"
  fi
}
trap cleanup EXIT HUP INT TERM
mkdir "$stage/correctness" "$stage/identity" "$stage/process" "$stage/tools"

copy_tool() {
  boc_source=$1
  boc_name=$2
  boc_digest=$3
  cp "$boc_source" "$scratch/$boc_name"
  chmod 500 "$scratch/$boc_name"
  benchmark_campaign_require_digest "$scratch/$boc_name" "$boc_digest" "$boc_name private copy"
  cp "$scratch/$boc_name" "$stage/tools/$boc_name"
  chmod 444 "$stage/tools/$boc_name"
}
copy_tool "$campaign_tool" campaign-tool "$campaign_tool_sha"
copy_tool "$driver" trial-driver "$driver_sha"
copy_tool "$correctness" correctness-verifier "$correctness_sha"
copy_tool "$identity_verifier" engine-identity-verifier "$identity_verifier_sha"
copy_tool "$supervisor" process-supervisor "$supervisor_sha"
cp "$repo_root/scripts/run-openai-comparison-campaign.sh" "$stage/tools/orchestrator"
cp "$repo_root/scripts/benchmark-campaign-common.sh" "$stage/tools/orchestrator-common"
chmod 444 "$stage/tools/orchestrator" "$stage/tools/orchestrator-common"
cp "$recipe" "$stage/campaign.recipe.v1"
cp "$declaration" "$stage/campaign.declaration.json"
chmod 444 "$stage/campaign.recipe.v1" "$stage/campaign.declaration.json"
benchmark_campaign_require_digest "$stage/campaign.declaration.json" \
  "$declaration_sha" 'staged declaration'

run_supervised() {
  boc_label=$1
  boc_timeout=$2
  boc_stdout=$3
  boc_stderr=$4
  boc_receipt=$5
  boc_credential_scope=$6
  shift 6
  boc_invocation=$scratch/invocation-$boc_label.v1
  benchmark_campaign_write_invocation \
    "$boc_invocation" "$boc_label" "$boc_timeout" "$grace" \
    "$boc_credential_scope" "$@"
  boc_invocation_sha=$(benchmark_campaign_sha256 "$boc_invocation")
  env -i LC_ALL=C PATH=/usr/bin:/bin \
    "$scratch/process-supervisor" run "$boc_timeout" "$grace" \
    "$boc_invocation" "$boc_invocation_sha" \
    "$boc_stdout" "$boc_stderr" "$boc_receipt" -- "$@" ||
    benchmark_campaign_fail "supervised process failed: $boc_label"
  benchmark_campaign_validate_invocation "$boc_invocation" \
    "$boc_invocation_sha" "$boc_label" "$boc_timeout" "$grace" \
    "$boc_credential_scope"
  benchmark_campaign_validate_supervisor_receipt "$boc_receipt" \
    "$boc_invocation_sha" "$boc_label"
  boc_invocation_artifact=${boc_receipt%.supervisor.v1}.invocation.v1
  [ "$boc_invocation_artifact" != "$boc_receipt" ] ||
    benchmark_campaign_fail "$boc_label receipt path is not canonical"
  cp "$boc_invocation" "$boc_invocation_artifact"
  chmod 444 "$boc_invocation_artifact"
  chmod 444 "$boc_receipt"
  [ ! -s "$boc_stderr" ] || benchmark_campaign_fail "$boc_label wrote stderr"
}

# Each helper receives only its exact credential scope.
run_without_credentials() {
  boc_label=$1
  boc_timeout=$2
  boc_stdout=$3
  boc_stderr=$4
  boc_receipt=$5
  shift 5
  run_supervised "$boc_label" "$boc_timeout" "$boc_stdout" "$boc_stderr" \
    "$boc_receipt" none "$@" 3<&- 4<&- 5<&-
}
run_with_selected_credential() {
  boc_credential_fd=$1
  shift
  boc_label=$1
  boc_timeout=$2
  boc_stdout=$3
  boc_stderr=$4
  boc_receipt=$5
  shift 5
  case "$boc_credential_fd" in
    3) run_supervised "$boc_label" "$boc_timeout" "$boc_stdout" \
      "$boc_stderr" "$boc_receipt" fd-3 "$@" 4<&- 5<&- ;;
    4) run_supervised "$boc_label" "$boc_timeout" "$boc_stdout" \
      "$boc_stderr" "$boc_receipt" fd-4 "$@" 3<&- 5<&- ;;
    5) run_supervised "$boc_label" "$boc_timeout" "$boc_stdout" \
      "$boc_stderr" "$boc_receipt" fd-5 "$@" 3<&- 4<&- ;;
    *) benchmark_campaign_fail 'selected credential descriptor is invalid' ;;
  esac
}

preflight_stdout=$scratch/preflight.stdout
preflight_stderr=$scratch/preflight.stderr
preflight_receipt=$stage/process/preflight.supervisor.v1
run_without_credentials preflight "$tool_timeout" "$preflight_stdout" "$preflight_stderr" \
  "$preflight_receipt" "$scratch/campaign-tool" --preflight \
  "$stage/campaign.declaration.json" "$declaration_sha"
expected_preflight=$scratch/preflight.expected
cat > "$expected_preflight" <<EOF
schema=lunaflux-openai-responses-campaign-preflight.v1
declaration_sha256=$declaration_sha
engine_order=lunaflux,vllm,sglang
profile_order=latency,chat,long_prefill,decode_heavy,prefix_rich,prefix_cold,saturation,churn,mixed
trial_order=profile_then_ordinal_then_counterbalanced_engine
trial_count=81
credential_ingress=external.inherited-descriptor.v1
EOF
for required_line in comparison_authority=none endpoint_0=127.0.0.1: \
  endpoint_1=127.0.0.1: endpoint_2=127.0.0.1:; do
  case "$required_line" in
    endpoint_*) grep -F "$required_line" "$preflight_stdout" >/dev/null 2>&1 ||
      benchmark_campaign_fail 'campaign preflight did not retain loopback endpoints' ;;
    *) grep -F -x "$required_line" "$preflight_stdout" >/dev/null 2>&1 ||
      benchmark_campaign_fail 'campaign preflight gained comparison authority' ;;
  esac
done
head -n 7 "$preflight_stdout" > "$scratch/preflight.prefix"
cmp -s "$scratch/preflight.prefix" "$expected_preflight" ||
  benchmark_campaign_fail 'campaign preflight matrix or declaration identity changed'
cp "$preflight_stdout" "$stage/preflight.record.v1"
chmod 444 "$stage/preflight.record.v1"

preflight_value() {
  boc_key=$1
  boc_result=$(sed -n "s/^${boc_key}=//p" "$preflight_stdout")
  [ -n "$boc_result" ] &&
    [ "$(grep -c "^${boc_key}=" "$preflight_stdout")" -eq 1 ] ||
    benchmark_campaign_fail "campaign preflight identity is missing: $boc_key"
  printf '%s\n' "$boc_result"
}
for engine_index in 0 1 2; do
  [ "$(preflight_value engine_$engine_index)" = \
    "$(benchmark_engine_name "$engine_index")" ] ||
    benchmark_campaign_fail 'campaign preflight engine order changed'
  benchmark_campaign_is_sha256 \
    "$(preflight_value engine_${engine_index}_revision_sha256)" ||
    benchmark_campaign_fail 'campaign preflight revision identity is invalid'
  benchmark_campaign_is_sha256 \
    "$(preflight_value engine_${engine_index}_image_sha256)" ||
    benchmark_campaign_fail 'campaign preflight image identity is invalid'
  benchmark_campaign_is_sha256 \
    "$(preflight_value engine_${engine_index}_configuration_sha256)" ||
    benchmark_campaign_fail 'campaign preflight configuration identity is invalid'
  benchmark_campaign_is_sha256 \
    "$(preflight_value engine_${engine_index}_executable_sha256)" ||
    benchmark_campaign_fail 'campaign preflight executable identity is invalid'
done

execution_order=$scratch/execution-order
: > "$execution_order"
for profile_index in 0 1 2 3 4 5 6 7 8; do
  profile=$(benchmark_campaign_name "$profile_index")
  for ordinal in 1 2 3; do
    for position in 0 1 2; do
      engine_index=$(benchmark_engine_for_position "$ordinal" "$position")
      engine=$(benchmark_engine_name "$engine_index")
      trial_index=$((profile_index * 9 + (ordinal - 1) * 3 + engine_index))
      capture_name=$(benchmark_capture_name "$trial_index")
      capture=$stage/$capture_name
      credential_fd=$((engine_index + 3))
      endpoint=$(preflight_value endpoint_$engine_index)
      revision=$(preflight_value engine_${engine_index}_revision_sha256)
      image=$(preflight_value engine_${engine_index}_image_sha256)
      configuration=$(preflight_value engine_${engine_index}_configuration_sha256)
      executable_identity=$(preflight_value engine_${engine_index}_executable_sha256)
      identity_artifact=$stage/identity/trial-$trial_index.identity.v1
      identity_stdout=$scratch/identity-$trial_index.stdout
      identity_stderr=$scratch/identity-$trial_index.stderr
      identity_receipt=$stage/process/identity-$trial_index.supervisor.v1
      run_without_credentials "identity-$trial_index" "$tool_timeout" \
        "$identity_stdout" "$identity_stderr" "$identity_receipt" \
        "$scratch/engine-identity-verifier" verify \
        "$stage/campaign.declaration.json" "$declaration_sha" "$trial_index" \
        "$engine" "$endpoint" "$revision" "$image" "$configuration" \
        "$executable_identity" "$identity_artifact"
      [ ! -s "$identity_stdout" ] ||
        benchmark_campaign_fail "identity-$trial_index wrote stdout"
      benchmark_campaign_require_file "$identity_artifact" \
        "trial-$trial_index engine identity artifact"
      benchmark_campaign_require_newline "$identity_artifact" \
        "trial-$trial_index engine identity artifact"
      [ "$(wc -l < "$identity_artifact" | tr -d ' ')" -eq 10 ] &&
        [ "$(benchmark_campaign_field "$identity_artifact" 1 schema)" = \
          'lunaflux.live-engine-identity.v1' ] &&
        [ "$(benchmark_campaign_field "$identity_artifact" 2 declaration_sha256)" = \
          "$declaration_sha" ] &&
        [ "$(benchmark_campaign_field "$identity_artifact" 3 trial_index)" = \
          "$trial_index" ] &&
        [ "$(benchmark_campaign_field "$identity_artifact" 4 engine)" = "$engine" ] &&
        [ "$(benchmark_campaign_field "$identity_artifact" 5 endpoint)" = "$endpoint" ] &&
        [ "$(benchmark_campaign_field "$identity_artifact" 6 revision_sha256)" = \
          "$revision" ] &&
        [ "$(benchmark_campaign_field "$identity_artifact" 7 image_sha256)" = \
          "$image" ] &&
        [ "$(benchmark_campaign_field "$identity_artifact" 8 configuration_sha256)" = \
          "$configuration" ] &&
        [ "$(benchmark_campaign_field "$identity_artifact" 9 executable_sha256)" = \
          "$executable_identity" ] &&
        [ "$(benchmark_campaign_field "$identity_artifact" 10 live_identity_verified)" = 1 ] ||
        benchmark_campaign_fail "trial-$trial_index live engine identity was not established"
      chmod 444 "$identity_artifact"
      stdout=$scratch/trial-$trial_index.stdout
      stderr=$scratch/trial-$trial_index.stderr
      receipt=$stage/process/trial-$trial_index.supervisor.v1
      run_with_selected_credential "$credential_fd" \
        "trial-$trial_index" "$trial_timeout" "$stdout" "$stderr" \
        "$receipt" "$scratch/trial-driver" run \
        "$stage/campaign.declaration.json" "$declaration_sha" "$trial_index" \
        "$profile" "$engine" "$ordinal" "$position" "$capture" \
        "$credential_fd" "$revision" "$image" "$configuration" \
        "$executable_identity"
      [ ! -s "$stdout" ] || benchmark_campaign_fail "trial-$trial_index wrote stdout"
      benchmark_campaign_require_file "$capture" "trial-$trial_index capture"
      benchmark_campaign_require_size "$capture" 67108864 "trial-$trial_index capture"
      chmod 444 "$capture"
      capture_sha=$(benchmark_campaign_sha256 "$capture")
      post_identity=$scratch/identity-post-$trial_index.v1
      post_identity_stdout=$scratch/identity-post-$trial_index.stdout
      post_identity_stderr=$scratch/identity-post-$trial_index.stderr
      post_identity_receipt=$stage/process/identity-post-$trial_index.supervisor.v1
      run_without_credentials "identity-post-$trial_index" "$tool_timeout" \
        "$post_identity_stdout" "$post_identity_stderr" \
        "$post_identity_receipt" \
        "$scratch/engine-identity-verifier" verify \
        "$stage/campaign.declaration.json" "$declaration_sha" "$trial_index" \
        "$engine" "$endpoint" "$revision" "$image" "$configuration" \
        "$executable_identity" "$post_identity"
      [ ! -s "$post_identity_stdout" ] ||
        benchmark_campaign_fail "identity-post-$trial_index wrote stdout"
      benchmark_campaign_require_file "$post_identity" \
        "trial-$trial_index post-trial engine identity artifact"
      cmp -s "$identity_artifact" "$post_identity" ||
        benchmark_campaign_fail \
          "trial-$trial_index engine identity changed across measurement"
      correctness_artifact=$stage/correctness/trial-$trial_index.correctness.v1
      correctness_stdout=$scratch/correctness-$trial_index.stdout
      correctness_stderr=$scratch/correctness-$trial_index.stderr
      correctness_receipt=$stage/process/correctness-$trial_index.supervisor.v1
      run_without_credentials "correctness-$trial_index" "$tool_timeout" \
        "$correctness_stdout" "$correctness_stderr" "$correctness_receipt" \
        "$scratch/correctness-verifier" verify \
        "$stage/campaign.declaration.json" "$declaration_sha" "$capture" \
        "$capture_sha" "$trial_index" "$correctness_artifact"
      [ ! -s "$correctness_stdout" ] ||
        benchmark_campaign_fail "correctness-$trial_index wrote stdout"
      benchmark_campaign_require_file "$correctness_artifact" \
        "trial-$trial_index correctness artifact"
      benchmark_campaign_require_newline "$correctness_artifact" \
        "trial-$trial_index correctness artifact"
      [ "$(wc -l < "$correctness_artifact" | tr -d ' ')" -eq 6 ] &&
        [ "$(benchmark_campaign_field "$correctness_artifact" 1 schema)" = \
          'lunaflux.external-correctness-observation.v1' ] &&
        [ "$(benchmark_campaign_field "$correctness_artifact" 2 declaration_sha256)" = \
          "$declaration_sha" ] &&
        [ "$(benchmark_campaign_field "$correctness_artifact" 3 trial_index)" = \
          "$trial_index" ] &&
        [ "$(benchmark_campaign_field "$correctness_artifact" 4 capture_sha256)" = \
          "$capture_sha" ] &&
        benchmark_campaign_is_sha256 \
          "$(benchmark_campaign_field "$correctness_artifact" 5 reference_sha256)" &&
        [ "$(benchmark_campaign_field "$correctness_artifact" 6 correctness_passed)" = 1 ] ||
        benchmark_campaign_fail "trial-$trial_index correctness was not established"
      chmod 444 "$correctness_artifact"
      printf '%s,%s,%s,%s,%s\n' \
        "$profile" "$ordinal" "$position" "$engine" "$trial_index" \
        >> "$execution_order"
    done
  done
done
cp "$execution_order" "$stage/execution-order.csv"
chmod 444 "$stage/execution-order.csv"

replay_stdout=$scratch/replay.stdout
replay_stderr=$scratch/replay.stderr
replay_receipt=$stage/process/replay.supervisor.v1
run_without_credentials replay "$tool_timeout" "$replay_stdout" "$replay_stderr" \
  "$replay_receipt" "$scratch/campaign-tool" "$stage" "$declaration_sha"
for replay_line in \
  trial_count=81 \
  observation_protocol=openai.responses.sse.v1 \
  replay_authority=offline_admission_only \
  comparison_authority=none \
  correctness_authority=none; do
  grep -F -x "$replay_line" "$replay_stdout" >/dev/null 2>&1 ||
    benchmark_campaign_fail "offline replay lost boundary: $replay_line"
done
replay_record_sha=$(sed -n 's/^replay_record_sha256=//p' "$replay_stdout")
benchmark_campaign_is_sha256 "$replay_record_sha" ||
  benchmark_campaign_fail 'offline replay record digest is invalid'
cp "$replay_stdout" "$stage/replay.record.v1"
chmod 444 "$stage/replay.record.v1"

capture_paths=$scratch/capture-paths
correctness_paths=$scratch/correctness-paths
identity_paths=$scratch/identity-paths
process_paths=$scratch/process-paths
tool_paths=$scratch/tool-paths
find "$stage" -maxdepth 1 -type f -name 'trial-*.capture.json' -print |
  sed "s#^$stage/##" | LC_ALL=C sort > "$capture_paths"
find "$stage/correctness" -type f -print | sed "s#^$stage/##" |
  LC_ALL=C sort > "$correctness_paths"
find "$stage/identity" -type f -print | sed "s#^$stage/##" |
  LC_ALL=C sort > "$identity_paths"
find "$stage/process" -type f -print | sed "s#^$stage/##" |
  LC_ALL=C sort > "$process_paths"
find "$stage/tools" -type f -print | sed "s#^$stage/##" |
  LC_ALL=C sort > "$tool_paths"
[ "$(wc -l < "$capture_paths" | tr -d ' ')" -eq 81 ] ||
  benchmark_campaign_fail 'raw capture matrix is incomplete'
[ "$(wc -l < "$correctness_paths" | tr -d ' ')" -eq 81 ] ||
  benchmark_campaign_fail 'correctness matrix is incomplete'
[ "$(wc -l < "$identity_paths" | tr -d ' ')" -eq 81 ] ||
  benchmark_campaign_fail 'live engine identity matrix is incomplete'
[ "$(wc -l < "$process_paths" | tr -d ' ')" -eq 652 ] ||
  benchmark_campaign_fail 'process cleanup receipt matrix is incomplete'
benchmark_campaign_write_inventory "$stage" "$capture_paths" \
  "$stage/captures.files.sha256"
benchmark_campaign_write_inventory "$stage" "$correctness_paths" \
  "$stage/correctness.files.sha256"
benchmark_campaign_write_inventory "$stage" "$identity_paths" \
  "$stage/identity.files.sha256"
benchmark_campaign_write_inventory "$stage" "$process_paths" \
  "$stage/process.files.sha256"
benchmark_campaign_write_inventory "$stage" "$tool_paths" \
  "$stage/tools.files.sha256"
for inventory in "$stage/"*.files.sha256; do chmod 444 "$inventory"; done

cat > "$stage/comparison-handoff.v1" <<EOF
schema=lunaflux.openai-comparison-handoff.v1
declaration_sha256=$declaration_sha
recipe_sha256=$recipe_sha
source_identity_sha256=$source_identity
campaign_tool_sha256=$campaign_tool_sha
trial_driver_sha256=$driver_sha
correctness_verifier_sha256=$correctness_sha
engine_identity_verifier_sha256=$identity_verifier_sha
process_supervisor_sha256=$supervisor_sha
capture_inventory_sha256=$(benchmark_campaign_sha256 "$stage/captures.files.sha256")
correctness_inventory_sha256=$(benchmark_campaign_sha256 "$stage/correctness.files.sha256")
identity_inventory_sha256=$(benchmark_campaign_sha256 "$stage/identity.files.sha256")
process_inventory_sha256=$(benchmark_campaign_sha256 "$stage/process.files.sha256")
tool_inventory_sha256=$(benchmark_campaign_sha256 "$stage/tools.files.sha256")
execution_order_sha256=$(benchmark_campaign_sha256 "$stage/execution-order.csv")
replay_record_sha256=$replay_record_sha
trial_count=81
correctness_artifact_count=81
identity_artifact_count=81
counterbalance=latin-square-v1
protocol=openai.responses.sse.v1
process_cleanup_complete=1
comparison_admission=external-correctness-join-required
comparison_authority=none
physical_measurement_claim=none
EOF
chmod 444 "$stage/comparison-handoff.v1"

all_paths=$scratch/all-paths
find "$stage" -type f -print | sed "s#^$stage/##" | LC_ALL=C sort > "$all_paths"
benchmark_campaign_write_inventory "$stage" "$all_paths" \
  "$stage/campaign.files.sha256"
chmod 444 "$stage/campaign.files.sha256"
find "$stage" -type d -exec chmod 700 {} \;
mkdir "$output" || benchmark_campaign_fail 'could not claim new campaign output'
claimed_output=$output
for entry in campaign.declaration.json campaign.recipe.v1 campaign.files.sha256 \
  captures.files.sha256 comparison-handoff.v1 correctness correctness.files.sha256 \
  execution-order.csv identity identity.files.sha256 preflight.record.v1 \
  process process.files.sha256 \
  replay.record.v1 tools tools.files.sha256; do
  mv "$stage/$entry" "$output/$entry"
done
for capture in "$stage"/trial-*.capture.json; do
  mv "$capture" "$output/$(basename -- "$capture")"
done
rmdir "$stage"
find "$output" -type d -exec chmod 555 {} \;
for file in $(find "$output" -type f -print); do chmod 444 "$file"; done
claimed_output=
rm -rf "$scratch"
trap - EXIT HUP INT TERM
printf '%s\n' 'LunaFlux external-process comparison campaign captured and sealed; comparison authority remains external.'
