#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"
fail() { printf 'spawned device-greedy campaign boundary rejected: %s\n' "$1" >&2; exit 1; }
for script in \
  scripts/run-spawned-device-greedy-physical-campaign.sh \
  scripts/verify-spawned-device-greedy-physical-campaign.sh \
  scripts/test-spawned-device-greedy-evidence-verifier.sh \
  scripts/augment-luna-kernel-root-plan-with-fused-runtime.sh \
  scripts/materialize-approved-tiny-bf16-launch.sh; do
  case "$(sed -n '1p' "$script")" in
    *bash) bash -n "$script" || fail "shell syntax rejected: $script" ;;
    *) sh -n "$script" || fail "shell syntax rejected: $script" ;;
  esac
done
for anchor in \
  'inspect-fused-production-runtime' \
  'device-greedy-fused-v2' \
  'spawn_boundary=descriptor_file,worker_wire,child_bootstrap,paged_device_executor' \
  'sampling_result_row_bytes=8' \
  'sampling_result_layout=token_i32,status_i32' \
  'sampling_success_status=-1' \
  'nonfinite_status=first_nonfinite_token_id' \
  'tie_policy=lowest_token_id' \
  'graph_policy_interaction=authenticated-and-stable' \
  'promotion_authority=absent'; do
  rg -Fq "$anchor" tests/approved_model_spawned_physical scripts ||
    fail "literal route anchor is absent: $anchor"
done
for anchor in \
  '--target-processes all' '--tool "$check"' '--error-exitcode 99' \
  'memcheck racecheck initcheck' 'sampling_readback_total_bytes=16' \
  'child_closed=2' 'lunaflux_prepare_evidence_manifest' \
  'OUTER_SEAL.sha256' 'output appeared before publication'; do
  rg -Fq -- "$anchor" scripts/run-spawned-device-greedy-physical-campaign.sh ||
    fail "physical runner anchor is absent: $anchor"
done
for anchor in \
  'fused-v2-device-greedy-v1' \
  'verify-fused-production-v2-physical-campaign.sh' \
  'aggregate fused production runtime did not pass normal admission' \
  'aggregate runtime substituted a lower-campaign module' \
  'augment-luna-kernel-root-plan-with-fused-runtime.sh' \
  '"fused_runtime":{"locator":"fused-production.runtime.v1"'; do
  rg -Fq "$anchor" scripts/materialize-approved-tiny-bf16-launch.sh ||
    fail "aggregate consumer anchor is absent: $anchor"
done
if rg -n 'manifest_bindable=true|promotion_authority=(present|granted)|production_readiness=ready' \
  scripts/run-spawned-device-greedy-physical-campaign.sh \
  tests/approved_model_spawned_physical/device_greedy_campaign.mbt; then
  fail 'qualification route gained promotion or readiness language'
fi
moon check --target native --deny-warn tests/approved_model_spawned_physical
moon test --target native --deny-warn tests/approved_model_spawned_physical
moon test --target native --deny-warn sampling
moon test --target native --deny-warn kernels/luna_cuda_sampling_aot
scripts/test-spawned-device-greedy-evidence-verifier.sh
printf '%s\n' 'spawned device-greedy physical campaign local gates: pass'
