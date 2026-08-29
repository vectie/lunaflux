#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
cd "$repo_root"

package=kernels/luna_tensor_core_candidate
proof_files='kernels/luna_tile_ir/tensor_core_view.mbt kernels/luna_tile_ir/tensor_core_view_types.mbt'

moon fmt --check kernels/luna_tile_ir "$package"
moon check --target native --deny-warn --warn-list +73 -p kernels/luna_tile_ir
moon test --target native --deny-warn --warn-list +73 -p kernels/luna_tile_ir
moon check --target native --deny-warn --warn-list +73 -p "$package"
moon test --target native --deny-warn --warn-list +73 -p "$package"
moon info --target native -p kernels/luna_tile_ir >/dev/null
moon info --target native -p "$package" >/dev/null

for source in $(rg --files "$package" -g '*.mbt') $proof_files
do
  lines=$(wc -l < "$source")
  if [ "$lines" -ge 500 ]
  then
    echo "$source exceeds the focused-file limit: $lines" >&2
    exit 1
  fi
done

if rg -n '@(fs|process|sys|async|device)|dlopen|nvrtc|cudaMalloc|cuModule|cudaLaunch|manifest\.write|runtime_jit' \
  "$package" $proof_files --glob '*.mbt' --glob 'moon.pkg'
then
  echo 'tensor-core candidate crossed its inert source-only boundary' >&2
  exit 1
fi

for required in \
  'compute_major != 12 || compute_minor != 0' \
  'm != 16 || n != 16 || k != 16' \
  'left_alignment_bytes != 32' \
  'pub fn admit_luna_tile_bf16_mma16x16x16_view(' \
  'tensor.alignment_bytes >= minimum_alignment_bytes' \
  'parallel_plan.compute_capability() !=' \
  '#include <mma.h>' \
  '__CUDA_ARCH__) && __CUDA_ARCH__ != 1200' \
  'nvcuda::wmma::fragment' \
  'nvcuda::wmma::mma_sync' \
  'sass_tensor_core_instruction_required=true' \
  'sass_qualification_observed=false' \
  'numeric_qualification_observed=false' \
  'performance_qualification_observed=false' \
  'manifest_bindable=false' \
  'promotion_authority=none' \
  'serial_oracle_sha256=' \
  'parallel_plan_sha256=' \
  '16-byte global operand alignment was accepted' \
  'foreign-target parallel plan was accepted'
do
  if ! rg -F -q "$required" "$package" $proof_files kernels/luna_tile_ir/parallel_types.mbt
  then
    echo "tensor-core candidate invariant missing: $required" >&2
    exit 1
  fi
done

if rg -F -q 'asm(' "$package/source.mbt" || \
  rg -F -q 'mma.sync' "$package/source.mbt"
then
  echo 'tensor-core candidate fabricated a typed PTX/SASS mapping' >&2
  exit 1
fi

if rg -n 'vectie/lunaflux/kernels/luna_tensor_core_candidate' \
  cmd engine runtime service kernels/luna_kernel_bundle \
  kernels/luna_artifact_admission --glob '*.mbt' --glob 'moon.pkg'
then
  echo 'production imported an unqualified tensor-core source candidate' >&2
  exit 1
fi

rg -F -q \
  'if policy.tensor_core_policy == RequireExternallyQualifiedTensorCore {' \
  kernels/luna_tile_ir/parallel_specialize.mbt
rg -F -q 'raise InvalidProgram(TensorCoreQualification)' \
  kernels/luna_tile_ir/parallel_specialize.mbt
rg -F -q 'caller-selected tensor-core qualification was accepted' \
  kernels/luna_tile_ir/parallel_test.mbt

echo 'LunaTile typed tensor-core source-candidate boundary: PASS'
