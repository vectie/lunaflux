#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

fail() {
  printf '%s\n' "FP8 reusable paged v3 boundary: $1" >&2
  exit 1
}

package=engine/device_step
interface="$package/pkg.generated.mbti"

for source in \
  fp8_blueprint_admit.mbt \
  fp8_blueprint_types.mbt \
  fp8_executor_prepare.mbt \
  fp8_frame_admission.mbt \
  fp8_scale_evidence.mbt; do
  lines=$(wc -l < "$package/$source" | tr -d ' ')
  [ "$lines" -lt 500 ] || fail "$source exceeds the 499-line budget"
done

rg -F -q \
  'pub fn prepare_fp8_paged_graph_executor_v3(@device.Context, @worker_protocol.ModelPlanGeneration, Fp8PagedExecutionBlueprintV3, @numeric_weight_file.AuthenticatedNumericWeightAuthority' \
  "$interface" || fail 'opaque executor preparation surface drifted'
rg -F -q '@fp8_release_authority.Fp8ReusablePagedEnvelopeAuthorityV3' \
  "$interface" || fail 'executor lost opaque reusable release authority'

for invariant in \
  'fp8_reusable_token_capacity_contains(model_rows, model_tokens, max_tokens)' \
  'workspace_offset += workspace_bytes' \
  'workspace_base + launch.workspace_offset()' \
  'fill_fp8_scale_sentinel_v3(evidence.host)' \
  'fp8_scale_evidence: scale_evidence' \
  'ProductionFastPath => None' \
  'validate_fp8_scale_evidence_v3(self)' \
  'descriptor.lifecycle = Poisoned' \
  'abort_paged_ordered_executor(ordered)'; do
  rg -F -q "$invariant" \
    engine/fp8_runtime_recipe/v3_admit.mbt \
    "$package/fp8_blueprint_admit.mbt" \
    "$package/execution_diagnostics.mbt" \
    "$package/fp8_executor_prepare.mbt" \
    "$package/fp8_scale_evidence.mbt" \
    "$package/paged_executor_run.mbt" || \
    fail "missing evidence invariant: $invariant"
done

rg -F -q \
  'test "production execution policy allocates no diagnostic host state"' \
  "$package/execution_diagnostics_wbtest.mbt" ||
  fail 'production no-observer-state test is missing'

if rg -F -n 'descriptor_fingerprint' \
  "$package/fp8_frame_admission.mbt" \
  "$package/paged_executor_types.mbt"; then
  fail 'diagnostic-only frame fingerprint leaked into reusable authority'
fi

if rg -n 'Array::|FixedArray::make|Bytes::' \
  "$package/fp8_frame_admission.mbt" \
  "$package/fp8_scale_evidence.mbt" \
  "$package/paged_executor_run.mbt"; then
  fail 'reusable frame admission/execution path constructs managed storage'
fi

moon check --target native --deny-warn --warn-list +73 \
  engine/fp8_runtime_recipe kernels/luna_capability_manifest \
  kernels/fp8_release_authority engine/device_step
moon test --target native --deny-warn --warn-list +73 \
  engine/fp8_runtime_recipe engine/device_step

printf '%s\n' 'FP8 reusable paged v3 boundary validation passed.'
