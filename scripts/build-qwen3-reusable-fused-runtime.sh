#!/bin/sh

set -eu
LC_ALL=C
TZ=UTC
export LC_ALL TZ
umask 077

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/luna-bf16-kernel-producer-common.sh"

usage() {
  lbf_fail 'usage: build-qwen3-reusable-fused-runtime.sh ABSOLUTE_NVCC ABSOLUTE_TOOLCHAIN_MANIFEST#sha256=HEX ABSOLUTE_QWEN_CANDIDATE_OUTPUT MODEL_CONTENT_SHA256 MODEL_PLAN_SHA256 ABSOLUTE_BUNDLE_EXPORTER#sha256=HEX ABSOLUTE_NEW_OUTPUT'
}

[ "$#" -eq 7 ] || usage
nvcc=$1
toolchain_argument=$2
candidate_output=$3
model_content_sha=$4
model_plan_sha=$5
exporter_argument=$6
output=$7

case "$toolchain_argument" in /*#sha256=*) ;; *) usage ;; esac
toolchain_manifest=${toolchain_argument%#sha256=*}
toolchain_sha=${toolchain_argument##*#sha256=}
case "$exporter_argument" in /*#sha256=*) ;; *) usage ;; esac
exporter=${exporter_argument%#sha256=*}
exporter_sha=${exporter_argument##*#sha256=}

lbf_require_absolute_file "$nvcc"
[ -x "$nvcc" ] || lbf_fail 'NVCC is not executable'
lbf_require_absolute_file "$toolchain_manifest"
lbf_require_absolute_directory "$candidate_output"
lbf_require_absolute_file "$exporter"
[ -x "$exporter" ] || lbf_fail 'runtime-bundle exporter is not executable'
for digest in "$toolchain_sha" "$model_content_sha" "$model_plan_sha" \
  "$exporter_sha"; do
  lbf_is_sha256 "$digest" || lbf_fail 'an input digest is invalid'
done
[ "$(lbf_sha256_file "$toolchain_manifest")" = "$toolchain_sha" ] ||
  lbf_fail 'toolchain-manifest digest mismatch'
[ "$(lbf_sha256_file "$exporter")" = "$exporter_sha" ] ||
  lbf_fail 'runtime-bundle exporter digest mismatch'

case "$output" in /*) ;; *) lbf_fail 'output path must be absolute' ;; esac
case "$output" in /|*//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
  lbf_fail 'output path is not a safe canonical absolute path'
  ;;
esac
[ ! -e "$output" ] && [ ! -L "$output" ] ||
  lbf_fail 'refusing to overwrite an existing output'
output_parent=$(CDPATH= cd -- "$(dirname -- "$output")" && pwd -P)
[ "$output_parent/$(basename -- "$output")" = "$output" ] ||
  lbf_fail 'output parent is not canonical'

driver_report=$(mktemp /tmp/lunaflux-qwen-fused-driver.XXXXXX)
scratch=$(mktemp -d /tmp/lunaflux-qwen-fused-build.XXXXXX)
publish_container=$(mktemp -d "$output_parent/.lunaflux-qwen-fused.XXXXXX")
published=0
cleanup() {
  rm -f -- "$driver_report"
  rm -rf -- "$scratch"
  if [ "$published" -ne 1 ]; then
    rm -rf -- "$publish_container"
  fi
}
trap cleanup EXIT HUP INT TERM

"$repo_root/scripts/inspect-luna-cuda-aot-driver.sh" "$nvcc" >"$driver_report"
driver_identity=$(sed -n 's/^driver_identity_sha256=//p' "$driver_report")
compiler_version=$(sed -n 's/^compiler_version=//p' "$driver_report")
approved_driver=$(sed -n 's/^driver_identity_sha256=//p' "$toolchain_manifest")
[ "$driver_identity" = "$approved_driver" ] ||
  lbf_fail 'NVCC driver identity is not admitted by the toolchain manifest'

stage=$publish_container/result
mkdir "$stage"
table=$scratch/modules.table
: >"$table"

validate_module() {
  name=$1
  relative=$2
  expected_schema=$3
  expected_family=$4
  operation_field=$5
  symbol_field=$6
  identity_mode=$7

  root=$candidate_output/$relative
  source_file=$root/kernel.cu
  recipe=$root/kernel.recipe
  lbf_require_absolute_file "$source_file"
  lbf_require_absolute_file "$recipe"
  [ "$(lbf_recipe_value schema "$recipe")" = "$expected_schema" ] ||
    lbf_fail "$name recipe schema mismatch"
  [ "$(lbf_recipe_value family "$recipe")" = "$expected_family" ] ||
    lbf_fail "$name recipe family mismatch"
  case "$identity_mode" in
    model-bound)
      [ "$(lbf_recipe_value model_content_sha256 "$recipe")" = "$model_content_sha" ] ||
        lbf_fail "$name model-content identity mismatch"
      [ "$(lbf_recipe_value model_plan_sha256 "$recipe")" = "$model_plan_sha" ] ||
        lbf_fail "$name model-plan identity mismatch"
      ;;
    reusable-generic)
      ! grep -q '^model_content_sha256=' "$recipe" ||
        lbf_fail "$name unexpectedly declares model-content identity"
      ! grep -q '^model_plan_sha256=' "$recipe" ||
        lbf_fail "$name unexpectedly declares model-plan identity"
      ;;
    *) lbf_fail "$name identity mode is invalid" ;;
  esac
  [ "$(lbf_recipe_value target "$recipe")" = sm_120 ] ||
    lbf_fail "$name target mismatch"
  [ "$(lbf_recipe_value toolchain_sha256 "$recipe")" = "$toolchain_sha" ] ||
    lbf_fail "$name toolchain identity mismatch"
  if grep -q '^compiler_version=' "$recipe"; then
    recipe_compiler=$(lbf_recipe_value compiler_version "$recipe")
  else
    recipe_compiler=$(lbf_recipe_value compiler "$recipe")
  fi
  [ "$recipe_compiler" = "$compiler_version" ] ||
    lbf_fail "$name compiler version mismatch"
  source_sha=$(lbf_recipe_value source_sha256 "$recipe")
  [ "$(lbf_sha256_file "$source_file")" = "$source_sha" ] ||
    lbf_fail "$name source digest mismatch"
  optimization=$(lbf_recipe_value optimization "$recipe")
  fmad=$(lbf_recipe_value fmad "$recipe")
  reassociate=$(lbf_recipe_value reassociate "$recipe")
  max_registers=$(lbf_recipe_value max_registers "$recipe")
  [ "$reassociate" = false ] || lbf_fail "$name enables reassociation"
  case "$optimization" in 0|1|2|3) ;; *) lbf_fail "$name optimization is invalid" ;; esac
  case "$fmad" in true|false) ;; *) lbf_fail "$name FMAD policy is invalid" ;; esac
  lbf_is_uint "$max_registers" && [ "$max_registers" -ge 1 ] &&
    [ "$max_registers" -le 255 ] || lbf_fail "$name register ceiling is invalid"
  operation=$(lbf_recipe_value "$operation_field" "$recipe")
  lbf_is_uint "$operation" || lbf_fail "$name operation identity is invalid"
  symbol=$(lbf_recipe_value "$symbol_field" "$recipe")
  grep -Fq "void $symbol(" "$source_file" ||
    lbf_fail "$name source does not define its declared symbol"
  grid=$(lbf_recipe_value grid "$recipe")
  block=$(lbf_recipe_value block "$recipe")
  shared=$(lbf_recipe_value shared_memory_bytes "$recipe")
  grid_x=${grid%%,*}
  grid_tail=${grid#*,}
  grid_y=${grid_tail%%,*}
  block_x=${block%%,*}
  for value in "$grid_x" "$grid_y" "$block_x"; do
    lbf_is_uint "$value" && [ "$value" -gt 0 ] ||
      lbf_fail "$name launch geometry is invalid"
  done
  lbf_is_uint "$shared" || lbf_fail "$name shared-memory size is invalid"

  common_flags="--cubin --std=c++14 --generate-code=arch=compute_120,code=sm_120 -O$optimization --fmad=$fmad --ftz=false --prec-div=true --prec-sqrt=true --maxrregcount=$max_registers --Werror all-warnings"
  module_root=$scratch/$name
  mkdir "$module_root" "$module_root/first" "$module_root/second"
  for pass in first second; do
    cp "$source_file" "$module_root/$pass/kernel.cu"
    (
      cd "$module_root/$pass"
      LC_ALL=C TZ=UTC SOURCE_DATE_EPOCH=0 CUDA_CACHE_DISABLE=1 \
        "$nvcc" $common_flags kernel.cu -o kernel.cubin >compiler.log 2>&1
    ) || lbf_fail "$name compiler invocation failed"
    [ -s "$module_root/$pass/kernel.cubin" ] ||
      lbf_fail "$name compiler produced an empty CUBIN"
    [ ! -s "$module_root/$pass/compiler.log" ] ||
      lbf_fail "$name compiler emitted diagnostics"
  done
  cmp -s "$module_root/first/kernel.cubin" "$module_root/second/kernel.cubin" ||
    lbf_fail "$name compiler output is not deterministic"
  cp "$module_root/first/kernel.cubin" "$stage/$name.cubin"
  printf '%s,%s,%s,%s,%s,%s\n' \
    "$name" "$operation" "$grid_x" "$grid_y" "$block_x" "$shared" >>"$table"
}

validate_module residual reusable-fused-residual \
  lunaflux-fused-parallel-cuda-aot-candidate.v2 \
  residual-rmsnorm-production-block128 residual_operation_id function_symbol \
  model-bound
validate_module ingress reusable-qwen-ingress \
  lunaflux-qknorm-rope-paged-kv-write-candidate.v2 \
  qknorm-positioned-rope-paged-kv-write qkv_operation symbol model-bound
validate_module prefill reusable-qwen-prefill-attention \
  lunaflux-attention-tile-compiler-cuda-aot-candidate.v1 \
  paged-attention-prefill-functional-tile \
  model_operation_id function_symbol reusable-generic
validate_module attention reusable-qwen-readonly-attention \
  lunaflux-paged-attention-readonly-cuda-aot-candidate.v2 \
  paged-attention-rotated-q-paged-kv-readonly-production \
  model_operation_id function_symbol reusable-generic

split_root=$candidate_output/reusable-qwen-decode-split-attention
split_source=$split_root/kernel.cu
split_recipe=$split_root/kernel.recipe
lbf_require_absolute_file "$split_source"
lbf_require_absolute_file "$split_recipe"
[ "$(lbf_recipe_value schema "$split_recipe")" = \
  lunaflux-paged-attention-decode-split-cuda-aot-recipe-v1 ] ||
  lbf_fail 'decode split recipe schema mismatch'
[ "$(lbf_recipe_value target "$split_recipe")" = sm_120 ] ||
  lbf_fail 'decode split target mismatch'
[ "$(lbf_recipe_value toolchain_sha256 "$split_recipe")" = "$toolchain_sha" ] ||
  lbf_fail 'decode split toolchain identity mismatch'
[ "$(lbf_recipe_value compiler_version "$split_recipe")" = "$compiler_version" ] ||
  lbf_fail 'decode split compiler version mismatch'
[ "$(lbf_recipe_value reassociate "$split_recipe")" = false ] ||
  lbf_fail 'decode split enables reassociation'
[ "$(lbf_recipe_value fmad "$split_recipe")" = false ] ||
  lbf_fail 'decode split FMAD policy mismatch'
[ "$(lbf_sha256_file "$split_source")" = \
  "$(lbf_recipe_value source_sha256 "$split_recipe")" ] ||
  lbf_fail 'decode split source digest mismatch'
split_operation=$(lbf_recipe_value operation_id "$split_recipe")
split_partial_symbol=$(lbf_recipe_value partial_function_symbol "$split_recipe")
split_merge_symbol=$(lbf_recipe_value merge_function_symbol "$split_recipe")
lbf_is_uint "$split_operation" || lbf_fail 'decode split operation is invalid'
grep -Fq "void $split_partial_symbol(" "$split_source" ||
  lbf_fail 'decode split partial symbol is absent'
grep -Fq "void $split_merge_symbol(" "$split_source" ||
  lbf_fail 'decode split merge symbol is absent'

# The readonly and decode split entries deliberately share one reusable CUBIN.
# The runtime ABI distinguishes this V3 module from older readonly-only V2
# bundles, while prefill continues to launch the readonly entry.
combined_source=$scratch/attention-combined.cu
cat "$candidate_output/reusable-qwen-readonly-attention/kernel.cu" \
  "$split_source" >"$combined_source"
combined_root=$scratch/attention-combined
mkdir "$combined_root" "$combined_root/first" "$combined_root/second"
for pass in first second; do
  cp "$combined_source" "$combined_root/$pass/kernel.cu"
  (
    cd "$combined_root/$pass"
    LC_ALL=C TZ=UTC SOURCE_DATE_EPOCH=0 CUDA_CACHE_DISABLE=1 \
      "$nvcc" $common_flags kernel.cu -o kernel.cubin >compiler.log 2>&1
  ) || lbf_fail 'combined attention compiler invocation failed'
  [ -s "$combined_root/$pass/kernel.cubin" ] ||
    lbf_fail 'combined attention compiler produced an empty CUBIN'
  [ ! -s "$combined_root/$pass/compiler.log" ] ||
    lbf_fail 'combined attention compiler emitted diagnostics'
done
cmp -s "$combined_root/first/kernel.cubin" \
  "$combined_root/second/kernel.cubin" ||
  lbf_fail 'combined attention compiler output is not deterministic'
cp "$combined_root/first/kernel.cubin" "$stage/attention.cubin"

module_row() {
  awk -F, -v name="$1" '$1 == name { print; found=1 } END { exit !found }' "$table"
}
residual_row=$(module_row residual)
ingress_row=$(module_row ingress)
attention_row=$(module_row attention)
prefill_row=$(module_row prefill)
IFS=, read -r _ residual_operation residual_grid_x residual_grid_y \
  residual_block_x residual_shared <<EOF
$residual_row
EOF
IFS=, read -r _ ingress_operation ingress_grid_x ingress_grid_y \
  ingress_block_x ingress_shared <<EOF
$ingress_row
EOF
IFS=, read -r _ attention_operation attention_grid_x attention_grid_y \
  attention_block_x attention_shared <<EOF
$attention_row
EOF
IFS=, read -r _ prefill_operation prefill_grid_x prefill_grid_y \
  prefill_block_x prefill_shared <<EOF
$prefill_row
EOF
[ "$split_operation" = "$attention_operation" ] ||
  lbf_fail 'decode split and readonly operation identities differ'
[ "$prefill_operation" = "$attention_operation" ] ||
  lbf_fail 'compiler prefill and readonly operation identities differ'

runtime=$stage/reusable-fused-runtime-bundle.v3
"$exporter" "$model_content_sha" "$model_plan_sha" \
  "$residual_operation" "$residual_grid_x" "$residual_grid_y" \
  "$residual_block_x" "$residual_shared" "$stage/residual.cubin" \
  "$ingress_operation" "$ingress_grid_x" "$ingress_grid_y" \
  "$ingress_block_x" "$ingress_shared" "$stage/ingress.cubin" \
  "$prefill_operation" "$prefill_grid_x" "$prefill_grid_y" \
  "$prefill_block_x" "$prefill_shared" "$stage/prefill.cubin" \
  "$attention_operation" "$attention_grid_x" "$attention_grid_y" \
  "$attention_block_x" "$attention_shared" "$stage/attention.cubin" \
  "$runtime" >"$scratch/export.stdout" 2>"$scratch/export.stderr" ||
  lbf_fail 'reusable fused runtime bundle export failed'
[ ! -s "$scratch/export.stderr" ] ||
  lbf_fail 'reusable fused runtime bundle exporter emitted stderr'

for module in residual ingress prefill attention; do
  printf '%s  %s.cubin\n' "$(lbf_sha256_file "$stage/$module.cubin")" "$module"
done >"$stage/FILES.sha256"
printf '%s  reusable-fused-runtime-bundle.v3\n' \
  "$(lbf_sha256_file "$runtime")" >>"$stage/FILES.sha256"
chmod 0444 "$stage"/*
mv "$stage" "$output"
published=1
trap - EXIT HUP INT TERM
rm -f -- "$driver_report"
rm -rf -- "$scratch" "$publish_container"

printf '%s\n' 'schema=lunaflux-qwen3-reusable-fused-runtime-build.v1'
printf 'runtime=%s/reusable-fused-runtime-bundle.v3\n' "$output"
printf 'runtime_sha256=%s\n' \
  "$(lbf_sha256_file "$output/reusable-fused-runtime-bundle.v3")"
printf '%s\n' 'compiler_invocations=10'
printf '%s\n' 'device_opened=0'
