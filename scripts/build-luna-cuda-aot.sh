#!/bin/sh
set -eu

fail() {
  printf 'luna CUDA AOT build failed: %s\n' "$1" >&2
  exit 1
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    fail 'sha256sum or shasum is required'
  fi
}

recipe_value() {
  key=$1
  line_number=$2
  line=$(sed -n "${line_number}p" "$recipe_snapshot")
  case "$line" in
    "$key"=*) printf '%s\n' "${line#*=}" ;;
    *) fail "recipe line $line_number must be $key" ;;
  esac
}

[ "$#" -eq 6 ] ||
  fail 'usage: build-luna-cuda-aot.sh ABSOLUTE_NVCC APPROVED_TOOLCHAIN_MANIFEST SOURCE RECIPE KERNEL_ROOT EVIDENCE_ROOT'
nvcc=$1
toolchain_manifest=$2
source_input=$3
recipe_input=$4
kernel_root=$5
evidence_root=$6

for approved_path in "$nvcc" "$toolchain_manifest" "$source_input" "$recipe_input" "$kernel_root" "$evidence_root"; do
  case "$approved_path" in /*) ;; *) fail 'all input and root paths must be absolute' ;; esac
  [ "$(realpath -- "$approved_path")" = "$approved_path" ] ||
    fail 'all input and root paths must be canonical'
done
[ -f "$nvcc" ] && [ -x "$nvcc" ] && [ ! -L "$nvcc" ] ||
  fail 'NVCC must be an executable non-symlink file'
[ -f "$toolchain_manifest" ] && [ ! -L "$toolchain_manifest" ] ||
  fail 'toolchain manifest must be a regular non-symlink file'
[ -f "$source_input" ] && [ ! -L "$source_input" ] ||
  fail 'source must be a regular non-symlink file'
[ -f "$recipe_input" ] && [ ! -L "$recipe_input" ] ||
  fail 'recipe must be a regular non-symlink file'
[ -d "$kernel_root" ] && [ ! -L "$kernel_root" ] ||
  fail 'kernel root must be an existing non-symlink directory'
[ -d "$evidence_root" ] && [ ! -L "$evidence_root" ] ||
  fail 'evidence root must be an existing non-symlink directory'
if [ -e "$kernel_root/sha256" ] || [ -L "$kernel_root/sha256" ]; then
  [ -d "$kernel_root/sha256" ] && [ ! -L "$kernel_root/sha256" ] ||
    fail 'kernel sha256 child must be a non-symlink directory'
  [ "$(realpath -- "$kernel_root/sha256")" = "$kernel_root/sha256" ] ||
    fail 'kernel sha256 child must be canonical'
else
  mkdir "$kernel_root/sha256"
fi
[ -d "$kernel_root/sha256" ] && [ ! -L "$kernel_root/sha256" ] &&
  [ "$(realpath -- "$kernel_root/sha256")" = "$kernel_root/sha256" ] ||
  fail 'kernel sha256 child changed during validation'

build_tmp=$(mktemp -d /tmp/lunaflux-luna-cuda-aot.XXXXXX)
trap 'rm -rf -- "$build_tmp"' EXIT HUP INT TERM
mkdir -p "$build_tmp/first" "$build_tmp/second"
cp "$source_input" "$build_tmp/kernel.cu"
cp "$recipe_input" "$build_tmp/recipe.txt"
cp "$toolchain_manifest" "$build_tmp/toolchain-manifest.txt"
chmod 444 "$build_tmp/kernel.cu" "$build_tmp/recipe.txt"
recipe_snapshot=$build_tmp/recipe.txt

recipe_lines=$(wc -l <"$recipe_snapshot" | tr -d ' ')
schema=$(recipe_value schema 1)
case "$schema" in
  lunaflux-luna-cuda-aot-recipe-v1)
    [ "$recipe_lines" -eq 14 ] ||
      fail 'v1 recipe must contain exactly 14 newline-terminated fields'
    expected_source_sha=$(recipe_value source_sha256 2)
    expected_toolchain_sha=$(recipe_value toolchain_sha256 3)
    compiler_version=$(recipe_value compiler_version 4)
    target=$(recipe_value target 5)
    output=$(recipe_value output 6)
    optimization=$(recipe_value optimization 7)
    fmad=$(recipe_value fmad 8)
    reassociate=$(recipe_value reassociate 9)
    max_registers=$(recipe_value max_registers 10)
    grid=$(recipe_value grid 11)
    block=$(recipe_value block 12)
    function_symbol=$(recipe_value function_symbol 13)
    abi=$(recipe_value abi 14)
    [ "$block" = 256,1,1 ] || fail 'block is invalid'
    [ "$abi" = pointer,pointer,pointer ] || fail 'unsupported kernel ABI'
    ;;
  lunaflux-fused-parallel-cuda-aot-candidate.v1)
    family=$(recipe_value family 2)
    case "$family" in
      qkv-positioned-rope-paged-kvwrite-block128)
        [ "$recipe_lines" -eq 46 ] ||
          fail 'fused QKV recipe must contain exactly 46 newline-terminated fields'
        expected_symbol=lunaflux_fused_qkv_positioned_rope_paged_kvwrite_bf16_block128_strict_v1
        expected_numerics=fused-qkv-positioned-rope-paged-kvwrite-ordered-f32-tolerance-v1
        expected_shared_memory=0
        expected_abi=step_counts,query_positions,query_row_offsets,sequence_lengths,page_table_offsets,page_table_page_indices,input,q_weight,k_weight,v_weight,rotated_qkv,key_cache,value_cache,dispatch_canary
        recipe_value model_content_sha256 23 >/dev/null
        recipe_value model_plan_sha256 24 >/dev/null
        recipe_value qkv_operation_id 25 >/dev/null
        recipe_value rope_operation_id 26 >/dev/null
        recipe_value attention_operation_id 27 >/dev/null
        recipe_value kv_layer 28 >/dev/null
        recipe_value shape 29 >/dev/null
        recipe_value rope_theta 30 >/dev/null
        recipe_value tokens_per_page 31 >/dev/null
        recipe_value total_page_count 32 >/dev/null
        recipe_value page_stride_bytes 33 >/dev/null
        recipe_value component_stride_bytes 34 >/dev/null
        recipe_value qkv_fallback_source_sha256 35 >/dev/null
        recipe_value qkv_fallback_recipe_sha256 36 >/dev/null
        recipe_value rope_fallback_source_sha256 37 >/dev/null
        recipe_value rope_fallback_recipe_sha256 38 >/dev/null
        recipe_value kv_write_fallback_source_sha256 39 >/dev/null
        recipe_value kv_write_fallback_recipe_sha256 40 >/dev/null
        abi=$(recipe_value abi 41)
        [ "$(recipe_value reference_fallback 42)" = three-distinct-correctness-kernels ] ||
          fail 'fused QKV reference fallback is invalid'
        aot_line=43
        ;;
      residual-rmsnorm-block128)
        [ "$recipe_lines" -eq 38 ] ||
          fail 'fused residual/RMSNorm recipe must contain exactly 38 newline-terminated fields'
        expected_symbol=lunaflux_fused_residual_rmsnorm_bf16_block128_tree_strict_v1
        expected_numerics=fused-residual-rmsnorm-block128-f32-tree-tolerance-v1
        expected_shared_memory=512
        expected_abi=step_counts,residual,branch,norm_weight,residual_output,norm_output,dispatch_canary
        recipe_value model_content_sha256 23 >/dev/null
        recipe_value model_plan_sha256 24 >/dev/null
        recipe_value residual_operation_id 25 >/dev/null
        recipe_value norm_operation_id 26 >/dev/null
        recipe_value width 27 >/dev/null
        recipe_value epsilon 28 >/dev/null
        recipe_value residual_fallback_source_sha256 29 >/dev/null
        recipe_value residual_fallback_recipe_sha256 30 >/dev/null
        recipe_value norm_fallback_source_sha256 31 >/dev/null
        recipe_value norm_fallback_recipe_sha256 32 >/dev/null
        abi=$(recipe_value abi 33)
        [ "$(recipe_value reference_fallback 34)" = two-distinct-correctness-kernels ] ||
          fail 'fused residual/RMSNorm reference fallback is invalid'
        aot_line=35
        ;;
      *) fail 'unsupported fused candidate family' ;;
    esac
    target=$(recipe_value target 3)
    profile_id=$(recipe_value profile_id 4)
    max_query_rows=$(recipe_value max_query_rows 5)
    max_query_tokens=$(recipe_value max_query_tokens 6)
    max_page_table_entries=$(recipe_value max_page_table_entries 7)
    expected_source_sha=$(recipe_value source_sha256 8)
    function_symbol=$(recipe_value function_symbol 9)
    block=$(recipe_value block 10)
    grid=$(recipe_value grid 11)
    shared_memory_bytes=$(recipe_value shared_memory_bytes 12)
    dispatch_canary=$(recipe_value dispatch_canary 13)
    numerics=$(recipe_value numerics 14)
    maximum_absolute_error_ppb=$(recipe_value maximum_absolute_error_ppb 15)
    maximum_relative_error_ppb=$(recipe_value maximum_relative_error_ppb 16)
    expected_toolchain_sha=$(recipe_value toolchain_sha256 17)
    compiler_version=$(recipe_value compiler_version 18)
    optimization=$(recipe_value optimization 19)
    fmad=$(recipe_value fmad 20)
    reassociate=$(recipe_value reassociate 21)
    max_registers=$(recipe_value max_registers 22)
    output=cubin
    [ "$block" = 128,1,1 ] || fail 'fused block is invalid'
    [ "$grid" = "$max_query_tokens,1,1" ] || fail 'fused grid is invalid'
    [ "$function_symbol" = "$expected_symbol" ] ||
      fail 'fused function symbol is invalid'
    [ "$numerics" = "$expected_numerics" ] ||
      fail 'fused numerical policy is invalid'
    [ "$shared_memory_bytes" = "$expected_shared_memory" ] ||
      fail 'fused shared-memory claim is invalid'
    [ "$abi" = "$expected_abi" ] || fail 'unsupported fused kernel ABI'
    [ "$(recipe_value aot_only "$aot_line")" = true ] ||
      fail 'fused recipe is not AOT-only'
    [ "$(recipe_value runtime_jit "$((aot_line + 1))")" = false ] ||
      fail 'fused recipe enables runtime JIT'
    [ "$(recipe_value release_binding "$((aot_line + 2))")" = candidate-only-promotion-required ] ||
      fail 'fused recipe broadened release binding'
    [ "$(recipe_value manifest_bindable "$((aot_line + 3))")" = false ] ||
      fail 'fused candidate became manifest-bindable'
    for bounded_integer in "$profile_id" "$max_query_rows" "$max_query_tokens" \
      "$max_page_table_entries" "$dispatch_canary" \
      "$maximum_absolute_error_ppb" "$maximum_relative_error_ppb"; do
      case "$bounded_integer" in
        [1-9]|[1-9][0-9]*) ;;
        *) fail 'fused bounded integer is invalid' ;;
      esac
    done
    [ "$fmad" = false ] || fail 'fused candidates require strict fmad policy'
    ;;
  lunaflux-fused-parallel-cuda-aot-candidate.v2)
    family=$(recipe_value family 2)
    [ "$(recipe_value execution_policy 3)" = production-fast-path ] ||
      fail 'production fused execution policy is invalid'
    recipe_value qualification_candidate_sha256 4 >/dev/null
    expected_source_sha=$(recipe_value source_sha256 5)
    function_symbol=$(recipe_value function_symbol 6)
    target=$(recipe_value target 7)
    profile_id=$(recipe_value profile_id 8)
    max_query_rows=$(recipe_value max_query_rows 9)
    max_query_tokens=$(recipe_value max_query_tokens 10)
    max_page_table_entries=$(recipe_value max_page_table_entries 11)
    block=$(recipe_value block 12)
    grid=$(recipe_value grid 13)
    shared_memory_bytes=$(recipe_value shared_memory_bytes 14)
    [ "$(recipe_value diagnostic_canary 15)" = absent ] ||
      fail 'production fused recipe retained diagnostic canary'
    numerics=$(recipe_value numerics 16)
    maximum_absolute_error_ppb=$(recipe_value maximum_absolute_error_ppb 17)
    maximum_relative_error_ppb=$(recipe_value maximum_relative_error_ppb 18)
    expected_toolchain_sha=$(recipe_value toolchain_sha256 19)
    compiler_version=$(recipe_value compiler_version 20)
    optimization=$(recipe_value optimization 21)
    fmad=$(recipe_value fmad 22)
    reassociate=$(recipe_value reassociate 23)
    max_registers=$(recipe_value max_registers 24)
    recipe_value model_content_sha256 25 >/dev/null
    recipe_value model_plan_sha256 26 >/dev/null
    case "$family" in
      residual-rmsnorm-production-block128)
        [ "$recipe_lines" -eq 40 ] || fail 'production residual recipe field count is invalid'
        for spec in 'residual_operation_id 27' 'norm_operation_id 28' 'width 29' 'epsilon 30' 'residual_fallback_source_sha256 31' 'residual_fallback_recipe_sha256 32' 'norm_fallback_source_sha256 33' 'norm_fallback_recipe_sha256 34'; do recipe_value "${spec% *}" "${spec#* }" >/dev/null; done
        abi_line=35; tail_line=36; expected_fallback=two-distinct-correctness-kernels
        expected_symbol=lunaflux_fused_residual_rmsnorm_bf16_block128_tree_production_v2
        expected_shared=512; expected_numerics=fused-residual-rmsnorm-block128-f32-tree-tolerance-v1
        expected_abi=step_counts,residual,branch,norm_weight,residual_output,norm_output ;;
      qkv-positioned-rope-paged-kvwrite-production-block128)
        [ "$recipe_lines" -eq 45 ] || fail 'production fused QKV recipe field count is invalid'
        for spec in 'qkv_operation_id 27' 'rope_operation_id 28' 'attention_operation_id 29' 'kv_layer 30' 'shape 31' 'rope_theta 32' 'tokens_per_page 33' 'total_page_count 34' 'page_stride_bytes 35' 'component_stride_bytes 36' 'qkv_fallback_source_sha256 37' 'rope_fallback_source_sha256 38' 'kv_write_fallback_source_sha256 39'; do recipe_value "${spec% *}" "${spec#* }" >/dev/null; done
        abi_line=40; tail_line=41; expected_fallback=three-distinct-correctness-kernels
        expected_symbol=lunaflux_fused_qkv_positioned_rope_paged_kvwrite_bf16_block128_production_v2
        expected_shared=0; expected_numerics=fused-qkv-positioned-rope-paged-kvwrite-ordered-f32-tolerance-v1
        expected_abi=step_counts,query_positions,query_row_offsets,sequence_lengths,page_table_offsets,page_table_page_indices,input,q_weight,k_weight,v_weight,rotated_qkv,key_cache,value_cache ;;
      *) fail 'unsupported production fused candidate family' ;;
    esac
    abi=$(recipe_value abi "$abi_line")
    [ "$(recipe_value reference_fallback "$tail_line")" = "$expected_fallback" ] ||
      fail 'production fused reference fallback is invalid'
    [ "$(recipe_value aot_only "$((tail_line + 1))")" = true ] ||
      fail 'production fused recipe is not AOT-only'
    [ "$(recipe_value runtime_jit "$((tail_line + 2))")" = false ] ||
      fail 'production fused recipe enables runtime JIT'
    [ "$(recipe_value release_binding "$((tail_line + 3))")" = external-approval-and-production-physical-gate-required ] ||
      fail 'production fused release gate is invalid'
    [ "$(recipe_value manifest_bindable "$((tail_line + 4))")" = false ] ||
      fail 'production fused candidate became manifest-bindable'
    output=cubin
    [ "$function_symbol" = "$expected_symbol" ] ||
      fail 'production fused function symbol is invalid'
    [ "$block" = 128,1,1 ] || fail 'production fused block is invalid'
    [ "$grid" = "$max_query_tokens,1,1" ] || fail 'production fused grid is invalid'
    [ "$shared_memory_bytes" = "$expected_shared" ] || fail 'production fused shared-memory claim is invalid'
    [ "$numerics" = "$expected_numerics" ] ||
      fail 'production fused numerical policy is invalid'
    [ "$abi" = "$expected_abi" ] ||
      fail 'unsupported production fused kernel ABI'
    for bounded_integer in "$profile_id" "$max_query_rows" "$max_query_tokens" \
      "$max_page_table_entries" "$maximum_absolute_error_ppb" \
      "$maximum_relative_error_ppb"; do
      case "$bounded_integer" in
        [1-9]|[1-9][0-9]*) ;;
        *) fail 'production fused bounded integer is invalid' ;;
      esac
    done
    [ "$fmad" = false ] || fail 'production fused candidate requires strict fmad policy'
    ;;
  lunaflux-positioned-paged-kv-write-candidate.v1)
    [ "$recipe_lines" -eq 30 ] ||
      fail 'positioned paged-KV-write recipe must contain exactly 30 newline-terminated fields'
    [ "$(recipe_value qualification_only 2)" = true ] ||
      fail 'paged-KV-write recipe is not qualification-only'
    [ "$(recipe_value manifest_bindable 3)" = false ] ||
      fail 'paged-KV-write recipe became manifest-bindable'
    [ "$(recipe_value promotion_authority 4)" = false ] ||
      fail 'paged-KV-write recipe gained promotion authority'
    [ "$(recipe_value bounds_authority 5)" = runtime-validated-step-descriptor ] ||
      fail 'paged-KV-write bounds authority drifted'
    recipe_value model_content_sha256 6 >/dev/null
    recipe_value model_plan_sha256 7 >/dev/null
    recipe_value attention_operation 8 >/dev/null
    recipe_value positioned_qkv_activation 9 >/dev/null
    recipe_value kv_layer 10 >/dev/null
    target=$(recipe_value target 11)
    expected_toolchain_sha=$(recipe_value toolchain_sha256 12)
    compiler_version=$(recipe_value compiler 13)
    optimization=$(recipe_value optimization 14)
    fmad=$(recipe_value fmad 15)
    reassociate=$(recipe_value reassociate 16)
    max_registers=$(recipe_value max_registers 17)
    recipe_value profile 18 >/dev/null
    recipe_value layout 19 >/dev/null
    function_symbol=$(recipe_value symbol 20)
    recipe_value dispatch_canary 21 >/dev/null
    abi=$(recipe_value abi 22)
    grid=$(recipe_value grid 23)
    block=$(recipe_value block 24)
    recipe_value operands 25 >/dev/null
    expected_source_sha=$(recipe_value source_sha256 26)
    [ "$(recipe_value fallback_family 27)" = monolithic-paged-write-and-attend ] ||
      fail 'paged-KV-write fallback family drifted'
    recipe_value fallback_operation 28 >/dev/null
    recipe_value fallback_source_sha256 29 >/dev/null
    recipe_value fallback_recipe_sha256 30 >/dev/null
    [ "$block" = 256,1,1 ] || fail 'paged-KV-write block is invalid'
    [ "$abi" = step_counts,query_positions,query_row_offsets,sequence_lengths,page_table_offsets,page_table_page_indices,positioned_qkv,key_cache,value_cache,dispatch_canary ] ||
      fail 'unsupported paged-KV-write kernel ABI'
    output=cubin
    ;;
  lunaflux-luna-cuda-projection-aot-recipe-v1)
    [ "$recipe_lines" -eq 25 ] || fail 'projection recipe must contain exactly 25 fields'
    recipe_value family 2 >/dev/null
    recipe_value operation_id 3 >/dev/null
    recipe_value profile_id 4 >/dev/null
    recipe_value entry_point_id 5 >/dev/null
    function_symbol=$(recipe_value function_symbol 6)
    expected_source_sha=$(recipe_value source_sha256 7)
    expected_toolchain_sha=$(recipe_value toolchain_sha256 8)
    compiler_version=$(recipe_value compiler_version 9)
    target=$(recipe_value target 10)
    output=$(recipe_value output 11)
    optimization=$(recipe_value optimization 12)
    fmad=$(recipe_value fmad 13)
    reassociate=$(recipe_value reassociate 14)
    max_registers=$(recipe_value max_registers 15)
    recipe_value input_width 16 >/dev/null
    recipe_value output_width 17 >/dev/null
    recipe_value intermediate_width 18 >/dev/null
    recipe_value max_query_tokens 19 >/dev/null
    grid=$(recipe_value grid 20)
    block=$(recipe_value block 21)
    [ "$(recipe_value manifest_bindable 22)" = false ] ||
      fail 'standalone projection oracle became manifest-bindable'
    recipe_value vendor_fast_path 23 >/dev/null
    recipe_value numerical_contract 24 >/dev/null
    recipe_value operands 25 >/dev/null
    [ "$block" = 256,1,1 ] || fail 'projection block is invalid'
    ;;
  lunaflux-luna-cuda-pointwise-aot-recipe-v1)
    [ "$recipe_lines" -eq 19 ] || fail 'pointwise recipe must contain exactly 19 fields'
    recipe_value family 2 >/dev/null
    recipe_value operation_id 3 >/dev/null
    recipe_value entry_point_id 4 >/dev/null
    function_symbol=$(recipe_value function_symbol 5)
    expected_source_sha=$(recipe_value source_sha256 6)
    expected_toolchain_sha=$(recipe_value toolchain_sha256 7)
    compiler_version=$(recipe_value compiler_version 8)
    target=$(recipe_value target 9)
    output=$(recipe_value output 10)
    optimization=$(recipe_value optimization 11)
    fmad=$(recipe_value fmad 12)
    reassociate=$(recipe_value reassociate 13)
    max_registers=$(recipe_value max_registers 14)
    grid=$(recipe_value grid 15)
    block=$(recipe_value block 16)
    [ "$(recipe_value manifest_bindable 17)" = false ] ||
      fail 'standalone pointwise oracle became manifest-bindable'
    recipe_value numerical_contract 18 >/dev/null
    recipe_value operands 19 >/dev/null
    [ "$block" = 256,1,1 ] || fail 'pointwise block is invalid'
    ;;
  *) fail 'unsupported recipe schema' ;;
esac

printf '%s\n' "$expected_source_sha" "$expected_toolchain_sha" |
  awk 'length != 64 || $0 !~ /^[0-9a-f]+$/ { exit 1 }' || fail 'recipe digests are invalid'
printf '%s\n' "$compiler_version" |
  awk '$0 !~ /^[0-9]+\.[0-9]+\.[0-9]+$/ { exit 1 }' ||
  fail 'compiler version is invalid'
case "$target" in
  sm_[1-9][0-9]|sm_[1-9][0-9][0-9]) ;;
  *) fail 'target is invalid' ;;
esac
[ "$output" = cubin ] || fail 'only CUBIN output is supported'
case "$optimization" in 0|1|2|3) ;; *) fail 'optimization is invalid' ;; esac
case "$fmad" in true|false) ;; *) fail 'fmad is invalid' ;; esac
[ "$reassociate" = false ] || fail 'reassociation is unsupported'
case "$max_registers" in
  ''|*[!0-9]*) fail 'max_registers is invalid' ;;
esac
[ "$max_registers" -ge 1 ] && [ "$max_registers" -le 255 ] ||
  fail 'max_registers is outside 1..255'
printf '%s\n' "$grid" |
  awk '$0 !~ /^[1-9][0-9]*,1,1$/ { exit 1 }' || fail 'grid is invalid'
printf '%s\n' "$function_symbol" |
  awk '$0 !~ /^[A-Za-z_][A-Za-z0-9_]*$/ || length > 96 { exit 1 }' ||
  fail 'function symbol is invalid'

actual_source_sha=$(sha256_file "$build_tmp/kernel.cu")
[ "$actual_source_sha" = "$expected_source_sha" ] || fail 'source digest mismatch'
actual_toolchain_sha=$(sha256_file "$build_tmp/toolchain-manifest.txt")
[ "$actual_toolchain_sha" = "$expected_toolchain_sha" ] ||
  fail 'approved toolchain manifest digest mismatch'
driver_report=$build_tmp/driver.txt
"$(dirname -- "$0")/inspect-luna-cuda-aot-driver.sh" "$nvcc" >"$driver_report"
driver_identity_sha=$(sed -n '6s/^driver_identity_sha256=//p' "$driver_report")
manifest_driver_count=$(grep -c '^driver_identity_sha256=' \
  "$build_tmp/toolchain-manifest.txt" || true)
[ "$manifest_driver_count" -eq 1 ] ||
  fail 'approved toolchain manifest must contain one driver identity'
approved_driver_identity=$(sed -n \
  's/^driver_identity_sha256=//p' "$build_tmp/toolchain-manifest.txt")
printf '%s\n' "$approved_driver_identity" |
  awk 'length != 64 || $0 !~ /^[0-9a-f]+$/ { exit 1 }' ||
  fail 'approved driver identity is invalid'
[ "$driver_identity_sha" = "$approved_driver_identity" ] ||
  fail 'invoked CUDA driver is not bound by the approved toolchain manifest'
actual_compiler_version=$(sed -n '7s/^compiler_version=//p' "$driver_report")
[ "$actual_compiler_version" = "$compiler_version" ] ||
  fail 'compiler version mismatch'

compute=${target#sm_}
common_flags="--cubin --std=c++14 --generate-code=arch=compute_${compute},code=${target} -O${optimization} --fmad=${fmad} --ftz=false --prec-div=true --prec-sqrt=true --maxrregcount=${max_registers} --Werror all-warnings"
for build in first second; do
  cp "$build_tmp/kernel.cu" "$build_tmp/$build/kernel.cu"
  (
    cd "$build_tmp/$build"
    LC_ALL=C TZ=UTC SOURCE_DATE_EPOCH=0 CUDA_CACHE_DISABLE=1 \
      "$nvcc" $common_flags kernel.cu -o kernel.cubin >compiler.log 2>&1
  ) || fail "$build compiler invocation failed"
  [ -s "$build_tmp/$build/kernel.cubin" ] || fail "$build produced an empty CUBIN"
done

first_sha=$(sha256_file "$build_tmp/first/kernel.cubin")
second_sha=$(sha256_file "$build_tmp/second/kernel.cubin")
[ "$first_sha" = "$second_sha" ] || fail 'independent CUBIN digests differ'
cmp -s "$build_tmp/first/kernel.cubin" "$build_tmp/second/kernel.cubin" ||
  fail 'independent CUBIN bytes differ'

recipe_sha=$(sha256_file "$recipe_snapshot")
{
  printf '%s\n' 'schema=lunaflux-luna-cuda-aot-evidence-v1'
  printf 'source_sha256=%s\n' "$actual_source_sha"
  printf 'recipe_sha256=%s\n' "$recipe_sha"
  printf 'toolchain_sha256=%s\n' "$actual_toolchain_sha"
  printf 'driver_identity_sha256=%s\n' "$driver_identity_sha"
  printf 'target=%s\n' "$target"
  printf 'function_symbol=%s\n' "$function_symbol"
  printf 'artifact_sha256=%s\n' "$first_sha"
  printf 'first_build_sha256=%s\n' "$first_sha"
  printf 'second_build_sha256=%s\n' "$second_sha"
  printf '%s\n' 'deterministic=1'
} >"$build_tmp/evidence.txt"

[ -d "$kernel_root/sha256" ] && [ ! -L "$kernel_root/sha256" ] &&
  [ "$(realpath -- "$kernel_root/sha256")" = "$kernel_root/sha256" ] ||
  fail 'kernel sha256 child changed before publication'
module_path=$kernel_root/sha256/$first_sha.cubin
evidence_path=$evidence_root/luna-cuda-aot-evidence-v1.txt
if [ -e "$module_path" ]; then
  [ -f "$module_path" ] && [ ! -L "$module_path" ] ||
    fail 'existing content-addressed module is not a regular file'
  cmp -s "$build_tmp/first/kernel.cubin" "$module_path" ||
    fail 'existing content-addressed module has different bytes'
else
  cp "$build_tmp/first/kernel.cubin" "$module_path"
fi
if [ -e "$evidence_path" ]; then
  [ -f "$evidence_path" ] && [ ! -L "$evidence_path" ] ||
    fail 'existing evidence is not a regular file'
  cmp -s "$build_tmp/evidence.txt" "$evidence_path" ||
    fail 'existing evidence does not match this build'
else
  cp "$build_tmp/evidence.txt" "$evidence_path"
fi
chmod 444 "$module_path" "$evidence_path"
printf 'module_relative_path=sha256/%s.cubin\n' "$first_sha"
printf 'artifact_sha256=%s\n' "$first_sha"
printf 'evidence_path=%s\n' "$evidence_path"
