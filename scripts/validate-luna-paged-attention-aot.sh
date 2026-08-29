#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
package_dir="$repo_root/kernels/luna_cuda_paged_attention_aot"
fixture_source="$package_dir/fixtures/generated_reference_v1_ep3.cu"
fixture_recipe="$package_dir/fixtures/generated_reference_v1_ep3.recipe"
expected_source=437bc28e313c554cfedfe1b775a646413ad08c5599b632a3d7d87777c69989e1
expected_recipe=8e3cdc25e8a41d64c20ac29cde60abd25f6ffb19177f9a0dd5348d23d852730c

test -d "$package_dir"
test -f "$package_dir/moon.pkg"
test "$(shasum -a 256 "$fixture_source" | awk '{print $1}')" = "$expected_source"
test "$(shasum -a 256 "$fixture_recipe" | awk '{print $1}')" = "$expected_recipe"
grep -qx 'language_standard=c++17' "$fixture_recipe"
grep -qx 'optimization=3' "$fixture_recipe"
grep -qx 'fmad=false' "$fixture_recipe"
grep -qx 'reassociate=false' "$fixture_recipe"
grep -qx 'max_registers=128' "$fixture_recipe"
grep -qx 'manifest_bindable=false' "$fixture_recipe"
if grep -Eq '^(module_sha256|family_id|workspace)=' "$fixture_recipe"; then
  echo "paged-attention candidate recipe contains final-contract identity" >&2
  exit 1
fi

if rg -n 'moonbitlang/(x/)?(fs|process)|internal/cuda|runtime_jit|nvrtc' \
  "$package_dir" -g '*.mbt' -g 'moon.pkg'; then
  echo "paged-attention AOT package crosses the offline typed-lowering boundary" >&2
  exit 1
fi

if rg -n '(__syncthreads|malloc|new char|cudaLaunchKernel)' \
  "$package_dir/source.mbt"; then
  echo "reference paged-attention source contains a forbidden synchronization/allocation path" >&2
  exit 1
fi

sh -n "$repo_root/scripts/probe-luna-paged-attention-cuda.sh"
moon check --target native --deny-warn kernels/luna_cuda_paged_attention_aot
moon test --target native --deny-warn kernels/luna_cuda_paged_attention_aot
