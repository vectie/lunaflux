#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f 1
  else
    shasum -a 256 "$1" | cut -d ' ' -f 1
  fi
}

while read -r expected source; do
  [ -n "$source" ] || continue
  actual=$(sha256_file "$source")
  [ "$actual" = "$expected" ] || {
    echo "I8 physical probe source digest mismatch: $source" >&2
    exit 1
  }
done < tests/i8_weight_scale_cuda_probe/SOURCE_SHA256SUMS

if rg -n 'vectie/lunaflux/internal/cuda|extern\s+"[cC]"|#external' \
  --glob '*.mbt' tests/i8_weight_scale_cuda_probe >/dev/null; then
  echo 'I8 physical probe bypasses the public device boundary' >&2
  exit 1
fi

for anchor in \
  'admit_symmetric_i8_weight_only_per_output_channel_v1' \
  'build_paged_symmetric_i8_weight_only_v1' \
  'admit_production_llama_symmetric_i8_weight_only_v1' \
  'CatalogVersion::v4()' \
  'admit_paged_graph_v4' \
  'artifact.admit_paged_v4' \
  'WeightInput(weight)' \
  'WeightScaleInput(scale_weight, scale)' \
  'tensor.scale() == Some(scale)' \
  'OperationExecutionContract::symmetric_i8_weight_only_v1()'; do
  rg -Fq "$anchor" tests/i8_weight_scale_cuda_probe || {
    echo "production I8 contract anchor is missing: $anchor" >&2
    exit 1
  }
done

for anchor in \
  'create_ordered_kernel_executor' \
  'OrderedKernelCaptureRequired' \
  'executor.launch_captured()' \
  'executor.record_completion()' \
  'executor.wait_completion()' \
  'executor.reset()'; do
  rg -Fq "$anchor" tests/i8_weight_scale_cuda_probe/device_run.mbt || {
    echo "public ordered-execution anchor is missing: $anchor" >&2
    exit 1
  }
done

for anchor in \
  'const int8_t *weight' \
  'const float *scale' \
  '__fmul_rn((float)weight[weight_base + inner]' \
  '__float2bfloat16_rn(accumulator)'; do
  rg -Fq "$anchor" \
    tests/i8_weight_scale_cuda_probe/fixtures/i8_weight_scale_projection.cu || {
    echo "I8 CUDA ABI/numeric anchor is missing: $anchor" >&2
    exit 1
  }
done

for anchor in \
  'probe_weight_codes.contains(-127)' \
  'probe_weight_codes.contains(127)' \
  'assert_false(probe_weight_codes.contains(-128))' \
  'expected_i8_projection_bf16()' \
  'error <= 0.0F'; do
  rg -Fq "$anchor" tests/i8_weight_scale_cuda_probe/cpu_referee.mbt || {
    echo "independent numeric referee anchor is missing: $anchor" >&2
    exit 1
  }
done

if rg -ni 'nvrtc|\.ptx|--ptx|compute_[0-9]+,code=compute|runtime.{0,12}compil' \
  --glob '*.mbt' --glob '*.cu' \
  tests/i8_weight_scale_cuda_probe scripts/probe-i8-weight-scale-cuda.sh \
  >/dev/null; then
  echo 'I8 physical probe introduced PTX or runtime compilation' >&2
  exit 1
fi

rg -Fq 'sm_89|sm_90|sm_120' scripts/probe-i8-weight-scale-cuda.sh || {
  echo 'closed production-I8 architecture allowlist is missing' >&2
  exit 1
}
rg -Fq 'args[2] == observed_target' \
  tests/i8_weight_scale_cuda_probe/main.mbt || {
  echo 'offline compile target is not bound to observed device capability' >&2
  exit 1
}
rg -Fq \
  'capability.compute_major() == 12 && capability.compute_minor() == 0' \
  tests/i8_weight_scale_cuda_probe/main.mbt || {
  echo 'sm120 is absent from the live I8 probe target gate' >&2
  exit 1
}
rg -Fq \
  'probe admits sm120 as I8 software target without a physical claim' \
  tests/i8_weight_scale_cuda_probe/contract_admission.mbt || {
  echo 'sm120 software admission lacks a non-physical focused test' >&2
  exit 1
}
rg -Fq 'scope=single-output-projection' \
  tests/i8_weight_scale_cuda_probe/main.mbt || {
  echo 'scoped physical evidence label is missing' >&2
  exit 1
}
rg -Fq 'outcome=i8-physical-rejected reason=' \
  tests/i8_weight_scale_cuda_probe/main.mbt || {
  echo 'typed physical rejection evidence is missing' >&2
  exit 1
}
rg -Fq 'tests/i8_weight_scale_cuda_probe/contract_admission.mbt' \
  scripts/validate-paged-v4-artifact-boundary.sh || {
  echo 'physical probe is absent from the aggregate catalog-v4 consumer audit' >&2
  exit 1
}
rg -Fq 'tests/i8_weight_scale_cuda_probe/contract_operands.mbt' \
  scripts/validate-paged-v4-artifact-boundary.sh || {
  echo 'physical probe operand consumer is absent from the aggregate audit' >&2
  exit 1
}

echo 'LunaFlux I8 weight+scale physical probe boundary passed.'
