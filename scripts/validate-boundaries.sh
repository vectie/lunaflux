#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

failed=0

for validator in \
  scripts/validate-luna-phase5-aot.sh \
  scripts/validate-fused-parallel-physical-campaign.sh \
  scripts/validate-fused-candidate-benchmark-qualification.sh \
  scripts/validate-benchmark-evidence-boundaries.sh \
  scripts/validate-openai-comparison-campaign.sh \
  scripts/validate-phase4-prefix-benchmark-boundaries.sh \
  scripts/validate-luna-profile-priority-boundary.sh \
  scripts/validate-luna-tile-optimizer-promotion-boundary.sh \
  scripts/validate-numeric-contract-boundaries.sh \
  scripts/validate-dense-llama-fp8-builder.sh \
  scripts/validate-fp8-launch-abi-boundary.sh \
  scripts/validate-fp8-launch-abi-v2.sh \
  scripts/validate-fp8-numeric-weight-materialization.sh \
  scripts/validate-authenticated-device-materialization-boundary-hostile.sh \
  scripts/validate-fp8-startup-admission.sh \
  scripts/validate-fp8-v3-frame-allocations.sh \
  scripts/validate-fp8-v2-compile-evidence.sh \
  kernels/fp8_release_authority/validate-boundary.sh \
  engine/fp8_device_executor/validate-boundary.sh \
  scripts/validate-luna-fp8-projection-aot.sh \
  scripts/validate-luna-fp8-projection-aot-v2.sh \
  scripts/validate-luna-kernel-bundle.sh \
  engine/fp8_runtime_recipe/validate-boundary.sh \
  scripts/validate-instance-metrics-allocations.sh \
  scripts/validate-operator-cli-convergence.sh \
  scripts/validate-debt-policy.sh \
  scripts/validate-moonbit-annotations.sh \
  scripts/validate-boundaries-capabilities.sh \
  scripts/validate-boundaries-core.sh \
  scripts/validate-service-boundaries.sh \
  scripts/validate-release-evidence-boundaries.sh \
  scripts/validate-final-release-inventory.sh \
  scripts/validate-i8-inert-capability-admission-boundaries.sh \
  scripts/validate-mistral-family-boundary.sh \
  scripts/validate-mistral-weight-boundary.sh \
  scripts/validate-numeric-schema-scaling.sh \
  scripts/validate-catalog-numeric-v4-boundary.sh \
  scripts/validate-dense-llama-i8-builder.sh \
  scripts/validate-paged-bf16-graph-probe.sh \
  scripts/validate-paged-bf16-shape-matrix-probe.sh \
  scripts/validate-paged-attention-parallel-candidate.sh \
  scripts/validate-fused-parallel-candidates.sh \
  scripts/validate-numeric-device-weight-layout-boundary.sh \
  scripts/validate-paged-numeric-v4-launch-boundary.sh \
  scripts/validate-paged-v4-artifact-boundary.sh \
  scripts/validate-i8-production-execution-boundary.sh \
  scripts/validate-inherited-drain-abi.sh \
  scripts/validate-inherited-drain-cli.sh \
  scripts/validate-inference-credential-abi.sh \
  scripts/validate-inference-credential-cli.sh \
  scripts/validate-promotion-verifier-key-boundary.sh \
  scripts/validate-promotion-verifier-key-allocations.sh \
  scripts/validate-runtime-startup-measurements.sh \
  scripts/validate-worker-executable-boundaries.sh \
  scripts/validate-worker-executable-activation-allocations.sh \
  scripts/validate-boundaries-native-owners.sh \
  scripts/validate-boundaries-interface-leaks.sh \
  scripts/validate-boundaries-online-composition.sh \
  scripts/validate-boundaries-request-composition.sh \
  scripts/validate-boundaries-authority-surfaces.sh \
  scripts/validate-boundaries-token-runtime.sh
do
  if ! "$validator"; then
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  exit 1
fi

printf '%s\n' 'LunaFlux dependency and debt boundaries are valid.'
