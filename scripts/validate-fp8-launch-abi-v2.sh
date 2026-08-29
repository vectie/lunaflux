#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

moon check kernels/fp8_launch_abi kernels/numeric_capability_manifest \
  --target native --deny-warn
moon test kernels/fp8_launch_abi kernels/numeric_capability_manifest \
  --target native --deny-warn

rg -q 'lunaflux\.fp8\.dynamic-launch-abi\.v1' \
  kernels/fp8_launch_abi/canonical.mbt
rg -q 'lunaflux\.fp8\.staged-dynamic-launch-abi\.v2' \
  kernels/fp8_launch_abi/v2_canonical.mbt
rg -q 'admit_staged_dynamic_activation_scale_launch_v2' \
  kernels/fp8_launch_abi/v2_admit.mbt
rg -q 'policy\.stage_at\(ordinal\)' kernels/fp8_launch_abi/v2_admit.mbt

for anchor in \
  'guard launches is PagedV4(contracts)' \
  'WeightScaleInput(weight, scale)' \
  'source_version: PagedNumericV4' \
  'region.alignment < operand.alignment()' \
  'append_v2_launch_contract(output, selected)' \
  'dimensions.shared_memory_bytes().to_string()' \
  'pub fn Fp8StagedDynamicLaunchAbi::raw_operands' \
  'v2 canonical identity binds exact dimensions and rejects reordered raw operands' \
  'v2 gated rejects legacy stateless source despite valid compound policy'; do
  rg -F -q "$anchor" kernels/fp8_launch_abi --glob '*.mbt' || {
    echo "FP8 launch ABI v2 missing exact raw-contract evidence: $anchor" >&2
    exit 1
  }
done

if rg -F -n 'select_stateless_contract(' kernels/fp8_launch_abi/v2_admit.mbt ||
  rg -F -n 'select_paged_contract(' kernels/fp8_launch_abi/v2_admit.mbt; then
  echo "FP8 launch ABI v2 still accepts a legacy scale-less contract source" >&2
  exit 1
fi

if rg -n 'internal/cuda|engine/|cmd/' \
  kernels/fp8_launch_abi/v2_*.mbt; then
  echo "FP8 launch ABI v2 crossed into device, engine, or CLI authority" >&2
  exit 1
fi

while IFS= read -r source_file; do
  line_count="$(wc -l < "$source_file")"
  if (( line_count >= 500 )); then
    echo "FP8 launch ABI v2 source exceeds the under-500-line boundary: $source_file" >&2
    exit 1
  fi
done < <(find kernels/fp8_launch_abi -type f -name 'v2_*.mbt' -print)

echo "FP8 whole-Workspace launch ABI v2 boundary is valid."
