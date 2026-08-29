#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"
probe_dir=tests/paged_bf16_shape_matrix_probe

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f 1
  else
    shasum -a 256 "$1" | cut -d ' ' -f 1
  fi
}

check_hash() {
  expected=$1
  path=$2
  actual=$(sha256_file "$path")
  [ "$actual" = "$expected" ] || {
    echo "shape-matrix fixture digest mismatch: $path" >&2
    exit 1
  }
}

check_hash a93252e0181653e1b90d535f1c28a79b065dd5cdeecc638bbddea94f927eb937 \
  "$probe_dir/fixtures/graph_contract.v1"
check_hash 58bb572f9df2f58f6af83023a57fabd4f29eb9992c106e319520fb2e63d9a704 \
  tests/paged_bf16_graph_probe/fixtures/graph_pointwise.cu
check_hash e655964ca086f8904b8995d0230261120166c48ab9f36a2758391aecb9e1a159 \
  "$probe_dir/fixtures/pointwise_h4.recipe"
check_hash 8bf155ac2033793128053d7d28d96054e8b607b8169c5378daf258e297616b4a \
  kernels/luna_cuda_projection_aot/fixtures/physical_sm120/qkv.cu
check_hash 07b0ea49b1d230dfe5de1c1a9542976a5ff773afe276cb7bece3b7f58a84cdfd \
  kernels/luna_cuda_projection_aot/fixtures/physical_sm120/qkv.recipe
check_hash 7cb12e0e7b31bc4dce14bf3659019de35ead821ce7cd1faf7c3d1be8ac6b8e29 \
  kernels/luna_cuda_projection_aot/fixtures/physical_sm120/dense.cu
check_hash fa15150d817abdcd77881bfc2d84003753320b8f5287c8bbd140b27a2610f323 \
  kernels/luna_cuda_projection_aot/fixtures/physical_sm120/dense.recipe
check_hash 5c22598f9d24ee9a6dbecf70d858814d040a579a565ea414b691099d9815aa3f \
  kernels/luna_cuda_projection_aot/fixtures/physical_sm120/mlp.cu
check_hash 14a076c53c1570310c0754257752770e5273b501c65bb552217a8904dcbccfbe \
  kernels/luna_cuda_projection_aot/fixtures/physical_sm120/mlp.recipe
check_hash 54b51660051c22e98fa8be726a3c3408b40635d458e4ab087a43f6665a41b44f \
  kernels/luna_cuda_projection_aot/fixtures/physical_sm120/lm_head.cu
check_hash 71296d1314c61d4c94eae4e0efbe327f1cb424b2bcedb9bd4ae8b402d7a82ede \
  kernels/luna_cuda_projection_aot/fixtures/physical_sm120/lm_head.recipe
check_hash 437bc28e313c554cfedfe1b775a646413ad08c5599b632a3d7d87777c69989e1 \
  kernels/luna_cuda_paged_attention_aot/fixtures/generated_reference_v1_ep3.cu
check_hash 8e3cdc25e8a41d64c20ac29cde60abd25f6ffb19177f9a0dd5348d23d852730c \
  kernels/luna_cuda_paged_attention_aot/fixtures/generated_reference_v1_ep3.recipe

for anchor in \
  'prefill-single' \
  'prefill-page-edge' \
  'decode-cached-prefix' \
  'mixed-prefill-decode' \
  'two-decode-rows'; do
  rg -Fq "$anchor" "$probe_dir/types.mbt" || {
    echo "shape-matrix case is missing: $anchor" >&2
    exit 1
  }
done

for anchor in \
  'OrderedKernelCaptureRequired' \
  'OrderedKernelCaptured' \
  'matrix_authenticated_module' \
  'executor.launch_captured()' \
  'executor.record_completion()' \
  'executor.wait_completion()' \
  'executor.reset()'; do
  rg -Fq "$anchor" "$probe_dir/device_graph.mbt" || {
    echo "shape-matrix lifecycle anchor is missing: $anchor" >&2
    exit 1
  }
done

if rg -n 'vectie/lunaflux/internal/cuda|extern\s+"[cC]"|#external' \
  "$probe_dir" --glob '*.mbt' --glob 'moon.pkg' >/dev/null; then
  echo 'shape-matrix probe bypasses the public device boundary' >&2
  exit 1
fi

if rg -n 'nvrtc|--ptx|code=compute_[0-9]+' \
  "$probe_dir" scripts/probe-paged-bf16-shape-matrix-cuda.sh \
  --glob '*.mbt' --glob '*.sh' >/dev/null; then
  echo 'shape-matrix probe introduced runtime compilation or PTX' >&2
  exit 1
fi

for source in "$probe_dir"/*.mbt; do
  lines=$(wc -l <"$source" | tr -d ' ')
  [ "$lines" -le 500 ] || {
    echo "shape-matrix source exceeds 500 lines: $source ($lines)" >&2
    exit 1
  }
done

sh -n scripts/probe-paged-bf16-shape-matrix-cuda.sh
moon check --target native --deny-warn "$probe_dir"
moon test --target native --deny-warn "$probe_dir"
echo 'LunaFlux bounded paged-BF16 shape-matrix boundary passed.'
