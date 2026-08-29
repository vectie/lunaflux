#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
test_tmp=$(realpath -- "$(mktemp -d /tmp/lunaflux-luna-cuda-aot-test.XXXXXX)")
trap 'rm -rf -- "$test_tmp"' EXIT HUP INT TERM
mkdir -p "$test_tmp/toolchain" "$test_tmp/input" "$test_tmp/kernel-root" "$test_tmp/evidence"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

assert_fused_rejected() {
  label=$1
  recipe=$2
  mkdir "$test_tmp/$label-kernel" "$test_tmp/$label-evidence"
  if "$root/scripts/build-luna-cuda-aot.sh" \
    "$test_tmp/toolchain/nvcc" \
    "$test_tmp/input/toolchain.manifest" \
    "$test_tmp/input/kernel.cu" \
    "$recipe" \
    "$test_tmp/$label-kernel" \
    "$test_tmp/$label-evidence" >/dev/null 2>&1; then
    printf 'hostile fused recipe was accepted: %s\n' "$label" >&2
    exit 1
  fi
}

cp "$root/scripts/fixtures/luna-cuda-aot/fake-nvcc.sh" "$test_tmp/toolchain/nvcc"
cp "$root/scripts/fixtures/luna-cuda-aot/fake-ptxas.sh" "$test_tmp/toolchain/ptxas"
chmod 555 "$test_tmp/toolchain/nvcc" "$test_tmp/toolchain/ptxas"
printf '%s\n' '__global__ void fixture() {}' >"$test_tmp/input/kernel.cu"

driver_report=$test_tmp/driver.txt
"$root/scripts/inspect-luna-cuda-aot-driver.sh" \
  "$test_tmp/toolchain/nvcc" >"$driver_report"
driver_identity_sha=$(sed -n '6s/^driver_identity_sha256=//p' "$driver_report")
{
  printf '%s\n' 'schema=test-approved-cuda-toolchain-v1'
  printf '%s\n' 'fixture=complete-toolkit-image'
  printf 'driver_identity_sha256=%s\n' "$driver_identity_sha"
} >"$test_tmp/input/toolchain.manifest"
if command -v sha256sum >/dev/null 2>&1; then
  source_sha=$(sha256sum "$test_tmp/input/kernel.cu" | awk '{print $1}')
  toolchain_sha=$(sha256sum "$test_tmp/input/toolchain.manifest" | awk '{print $1}')
else
  source_sha=$(shasum -a 256 "$test_tmp/input/kernel.cu" | awk '{print $1}')
  toolchain_sha=$(shasum -a 256 "$test_tmp/input/toolchain.manifest" | awk '{print $1}')
fi

{
  printf '%s\n' 'schema=lunaflux-luna-cuda-aot-recipe-v1'
  printf 'source_sha256=%s\n' "$source_sha"
  printf 'toolchain_sha256=%s\n' "$toolchain_sha"
  printf '%s\n' 'compiler_version=13.1.0'
  printf '%s\n' 'target=sm_90'
  printf '%s\n' 'output=cubin'
  printf '%s\n' 'optimization=3'
  printf '%s\n' 'fmad=false'
  printf '%s\n' 'reassociate=false'
  printf '%s\n' 'max_registers=128'
  printf '%s\n' 'grid=1,1,1'
  printf '%s\n' 'block=256,1,1'
  printf '%s\n' 'function_symbol=lunaflux_luna_residual_add_v1_ep_7'
  printf '%s\n' 'abi=pointer,pointer,pointer'
} >"$test_tmp/input/recipe.txt"

mkdir -p "$test_tmp/unapproved-toolchain"
cp "$root/scripts/fixtures/luna-cuda-aot/fake-nvcc.sh" \
  "$test_tmp/unapproved-toolchain/nvcc"
cp "$root/scripts/fixtures/luna-cuda-aot/fake-ptxas.sh" \
  "$test_tmp/unapproved-toolchain/ptxas"
printf '%s\n' '# different binary identity' >>"$test_tmp/unapproved-toolchain/nvcc"
chmod 555 "$test_tmp/unapproved-toolchain/nvcc" \
  "$test_tmp/unapproved-toolchain/ptxas"
if "$root/scripts/build-luna-cuda-aot.sh" \
  "$test_tmp/unapproved-toolchain/nvcc" \
  "$test_tmp/input/toolchain.manifest" \
  "$test_tmp/input/kernel.cu" \
  "$test_tmp/input/recipe.txt" \
  "$test_tmp/kernel-root" \
  "$test_tmp/evidence" >/dev/null 2>&1; then
  printf '%s\n' 'unapproved CUDA driver identity was accepted' >&2
  exit 1
fi

"$root/scripts/build-luna-cuda-aot.sh" \
  "$test_tmp/toolchain/nvcc" \
  "$test_tmp/input/toolchain.manifest" \
  "$test_tmp/input/kernel.cu" \
  "$test_tmp/input/recipe.txt" \
  "$test_tmp/kernel-root" \
  "$test_tmp/evidence" >"$test_tmp/result.txt"
module_relative=$(sed -n 's/^module_relative_path=//p' "$test_tmp/result.txt")
[ -f "$test_tmp/kernel-root/$module_relative" ]
grep -qx 'deterministic=1' "$test_tmp/evidence/luna-cuda-aot-evidence-v1.txt"
grep -qx "driver_identity_sha256=$driver_identity_sha" \
  "$test_tmp/evidence/luna-cuda-aot-evidence-v1.txt"

printf '%s\n' 'tamper' >>"$test_tmp/input/kernel.cu"
if "$root/scripts/build-luna-cuda-aot.sh" \
  "$test_tmp/toolchain/nvcc" \
  "$test_tmp/input/toolchain.manifest" \
  "$test_tmp/input/kernel.cu" \
  "$test_tmp/input/recipe.txt" \
  "$test_tmp/kernel-root" \
  "$test_tmp/evidence" >/dev/null 2>&1; then
  printf '%s\n' 'tampered source was accepted' >&2
  exit 1
fi

printf '%s\n' '__global__ void fixture() {}' >"$test_tmp/input/kernel.cu"
if FAKE_LUNA_CUDA_NONDETERMINISTIC=1 "$root/scripts/build-luna-cuda-aot.sh" \
  "$test_tmp/toolchain/nvcc" \
  "$test_tmp/input/toolchain.manifest" \
  "$test_tmp/input/kernel.cu" \
  "$test_tmp/input/recipe.txt" \
  "$test_tmp/kernel-root" \
  "$test_tmp/evidence" >/dev/null 2>&1; then
  printf '%s\n' 'nondeterministic compiler output was accepted' >&2
  exit 1
fi

mkdir -p "$test_tmp/redirected-kernel-root" "$test_tmp/redirected-evidence" "$test_tmp/redirect-target"
ln -s "$test_tmp/redirect-target" "$test_tmp/redirected-kernel-root/sha256"
if "$root/scripts/build-luna-cuda-aot.sh" \
  "$test_tmp/toolchain/nvcc" \
  "$test_tmp/input/toolchain.manifest" \
  "$test_tmp/input/kernel.cu" \
  "$test_tmp/input/recipe.txt" \
  "$test_tmp/redirected-kernel-root" \
  "$test_tmp/redirected-evidence" >/dev/null 2>&1; then
  printf '%s\n' 'redirected sha256 output child was accepted' >&2
  exit 1
fi

printf '%s\n' '__global__ void fixture() {}' >"$test_tmp/input/kernel.cu"
source_sha=$(sha256_file "$test_tmp/input/kernel.cu")
qkv_recipe=$test_tmp/input/fused-qkv.recipe
{
  printf '%s\n' \
    'schema=lunaflux-fused-parallel-cuda-aot-candidate.v1' \
    'family=qkv-positioned-rope-paged-kvwrite-block128' \
    'target=sm_120' \
    'profile_id=1' \
    'max_query_rows=1' \
    'max_query_tokens=4' \
    'max_page_table_entries=4' \
    "source_sha256=$source_sha" \
    'function_symbol=lunaflux_fused_qkv_positioned_rope_paged_kvwrite_bf16_block128_strict_v1' \
    'block=128,1,1' \
    'grid=4,1,1' \
    'shared_memory_bytes=0' \
    'dispatch_canary=1364350539' \
    'numerics=fused-qkv-positioned-rope-paged-kvwrite-ordered-f32-tolerance-v1' \
    'maximum_absolute_error_ppb=250000' \
    'maximum_relative_error_ppb=500000' \
    "toolchain_sha256=$toolchain_sha" \
    'compiler_version=13.1.0' \
    'optimization=3' \
    'fmad=false' \
    'reassociate=false' \
    'max_registers=128' \
    "model_content_sha256=$source_sha" \
    "model_plan_sha256=$toolchain_sha" \
    'qkv_operation_id=1' \
    'rope_operation_id=2' \
    'attention_operation_id=3' \
    'kv_layer=1' \
    'shape=4,12,2,1,4' \
    'rope_theta=10000' \
    'tokens_per_page=4' \
    'total_page_count=8' \
    'page_stride_bytes=32' \
    'component_stride_bytes=16' \
    "qkv_fallback_source_sha256=$source_sha" \
    "qkv_fallback_recipe_sha256=$toolchain_sha" \
    "rope_fallback_source_sha256=$source_sha" \
    "rope_fallback_recipe_sha256=$toolchain_sha" \
    "kv_write_fallback_source_sha256=$source_sha" \
    "kv_write_fallback_recipe_sha256=$toolchain_sha" \
    'abi=step_counts,query_positions,query_row_offsets,sequence_lengths,page_table_offsets,page_table_page_indices,input,q_weight,k_weight,v_weight,rotated_qkv,key_cache,value_cache,dispatch_canary' \
    'reference_fallback=three-distinct-correctness-kernels' \
    'aot_only=true' \
    'runtime_jit=false' \
    'release_binding=candidate-only-promotion-required' \
    'manifest_bindable=false'
} >"$qkv_recipe"
mkdir "$test_tmp/fused-qkv-kernel" "$test_tmp/fused-qkv-evidence"
"$root/scripts/build-luna-cuda-aot.sh" \
  "$test_tmp/toolchain/nvcc" \
  "$test_tmp/input/toolchain.manifest" \
  "$test_tmp/input/kernel.cu" \
  "$qkv_recipe" \
  "$test_tmp/fused-qkv-kernel" \
  "$test_tmp/fused-qkv-evidence" >"$test_tmp/fused-qkv-result.txt"
grep -qx 'target=sm_120' \
  "$test_tmp/fused-qkv-evidence/luna-cuda-aot-evidence-v1.txt"
grep -qx \
  'function_symbol=lunaflux_fused_qkv_positioned_rope_paged_kvwrite_bf16_block128_strict_v1' \
  "$test_tmp/fused-qkv-evidence/luna-cuda-aot-evidence-v1.txt"

residual_recipe=$test_tmp/input/fused-residual.recipe
{
  printf '%s\n' \
    'schema=lunaflux-fused-parallel-cuda-aot-candidate.v1' \
    'family=residual-rmsnorm-block128' \
    'target=sm_120' \
    'profile_id=1' \
    'max_query_rows=1' \
    'max_query_tokens=4' \
    'max_page_table_entries=4' \
    "source_sha256=$source_sha" \
    'function_symbol=lunaflux_fused_residual_rmsnorm_bf16_block128_tree_strict_v1' \
    'block=128,1,1' \
    'grid=4,1,1' \
    'shared_memory_bytes=512' \
    'dispatch_canary=1380864582' \
    'numerics=fused-residual-rmsnorm-block128-f32-tree-tolerance-v1' \
    'maximum_absolute_error_ppb=100000' \
    'maximum_relative_error_ppb=250000' \
    "toolchain_sha256=$toolchain_sha" \
    'compiler_version=13.1.0' \
    'optimization=3' \
    'fmad=false' \
    'reassociate=false' \
    'max_registers=128' \
    "model_content_sha256=$source_sha" \
    "model_plan_sha256=$toolchain_sha" \
    'residual_operation_id=4' \
    'norm_operation_id=5' \
    'width=4' \
    'epsilon=0.00001' \
    "residual_fallback_source_sha256=$source_sha" \
    "residual_fallback_recipe_sha256=$toolchain_sha" \
    "norm_fallback_source_sha256=$source_sha" \
    "norm_fallback_recipe_sha256=$toolchain_sha" \
    'abi=step_counts,residual,branch,norm_weight,residual_output,norm_output,dispatch_canary' \
    'reference_fallback=two-distinct-correctness-kernels' \
    'aot_only=true' \
    'runtime_jit=false' \
    'release_binding=candidate-only-promotion-required' \
    'manifest_bindable=false'
} >"$residual_recipe"
mkdir "$test_tmp/fused-residual-kernel" "$test_tmp/fused-residual-evidence"
"$root/scripts/build-luna-cuda-aot.sh" \
  "$test_tmp/toolchain/nvcc" \
  "$test_tmp/input/toolchain.manifest" \
  "$test_tmp/input/kernel.cu" \
  "$residual_recipe" \
  "$test_tmp/fused-residual-kernel" \
  "$test_tmp/fused-residual-evidence" >"$test_tmp/fused-residual-result.txt"
grep -qx \
  'function_symbol=lunaflux_fused_residual_rmsnorm_bf16_block128_tree_strict_v1' \
  "$test_tmp/fused-residual-evidence/luna-cuda-aot-evidence-v1.txt"

production_residual_recipe=$test_tmp/input/fused-residual-production.recipe
{
  printf '%s\n' \
    'schema=lunaflux-fused-parallel-cuda-aot-candidate.v2' \
    'family=residual-rmsnorm-production-block128' \
    'execution_policy=production-fast-path' \
    "qualification_candidate_sha256=$source_sha" \
    "source_sha256=$source_sha" \
    'function_symbol=lunaflux_fused_residual_rmsnorm_bf16_block128_tree_production_v2' \
    'target=sm_120' \
    'profile_id=1' \
    'max_query_rows=1' \
    'max_query_tokens=4' \
    'max_page_table_entries=4' \
    'block=128,1,1' \
    'grid=4,1,1' \
    'shared_memory_bytes=512' \
    'diagnostic_canary=absent' \
    'numerics=fused-residual-rmsnorm-block128-f32-tree-tolerance-v1' \
    'maximum_absolute_error_ppb=100000' \
    'maximum_relative_error_ppb=250000' \
    "toolchain_sha256=$toolchain_sha" \
    'compiler_version=13.1.0' \
    'optimization=3' \
    'fmad=false' \
    'reassociate=false' \
    'max_registers=128' \
    "model_content_sha256=$source_sha" \
    "model_plan_sha256=$toolchain_sha" \
    'residual_operation_id=4' \
    'norm_operation_id=5' \
    'width=4' \
    'epsilon=0.00001' \
    "residual_fallback_source_sha256=$source_sha" \
    "residual_fallback_recipe_sha256=$toolchain_sha" \
    "norm_fallback_source_sha256=$source_sha" \
    "norm_fallback_recipe_sha256=$toolchain_sha" \
    'abi=step_counts,residual,branch,norm_weight,residual_output,norm_output' \
    'reference_fallback=two-distinct-correctness-kernels' \
    'aot_only=true' \
    'runtime_jit=false' \
    'release_binding=external-approval-and-production-physical-gate-required' \
    'manifest_bindable=false'
} >"$production_residual_recipe"
mkdir "$test_tmp/fused-residual-production-kernel" \
  "$test_tmp/fused-residual-production-evidence"
"$root/scripts/build-luna-cuda-aot.sh" \
  "$test_tmp/toolchain/nvcc" \
  "$test_tmp/input/toolchain.manifest" \
  "$test_tmp/input/kernel.cu" \
  "$production_residual_recipe" \
  "$test_tmp/fused-residual-production-kernel" \
  "$test_tmp/fused-residual-production-evidence" \
  >"$test_tmp/fused-residual-production-result.txt"
grep -qx \
  'function_symbol=lunaflux_fused_residual_rmsnorm_bf16_block128_tree_production_v2' \
  "$test_tmp/fused-residual-production-evidence/luna-cuda-aot-evidence-v1.txt"
production_hostile=$test_tmp/input/fused-residual-production-hostile.recipe
sed 's/diagnostic_canary=absent/dispatch_canary=17/' \
  "$production_residual_recipe" >"$production_hostile"
assert_fused_rejected fused-production-canary "$production_hostile"

hostile_recipe=$test_tmp/input/fused-hostile.recipe
sed 's/shared_memory_bytes=512/shared_memory_bytes=0/' \
  "$residual_recipe" >"$hostile_recipe"
assert_fused_rejected fused-shared-memory "$hostile_recipe"
for mutation in block abi manifest runtime-jit symbol fmad grid decimal; do
  mutated=$test_tmp/input/fused-$mutation.recipe
  case "$mutation" in
    block) sed 's/block=128,1,1/block=256,1,1/' "$qkv_recipe" >"$mutated" ;;
    abi) sed 's/,dispatch_canary$/,extra,dispatch_canary/' "$qkv_recipe" >"$mutated" ;;
    manifest) sed 's/manifest_bindable=false/manifest_bindable=true/' "$qkv_recipe" >"$mutated" ;;
    runtime-jit) sed 's/runtime_jit=false/runtime_jit=true/' "$qkv_recipe" >"$mutated" ;;
    symbol) sed 's/function_symbol=lunaflux_/function_symbol=alternate_/' "$qkv_recipe" >"$mutated" ;;
    fmad) sed 's/fmad=false/fmad=true/' "$qkv_recipe" >"$mutated" ;;
    grid) sed 's/grid=4,1,1/grid=3,1,1/' "$qkv_recipe" >"$mutated" ;;
    decimal) sed 's/max_query_tokens=4/max_query_tokens=04/' "$qkv_recipe" >"$mutated" ;;
  esac
  assert_fused_rejected "fused-$mutation" "$mutated"
done

printf '%s\n' 'Luna CUDA AOT offline builder gate passed'
