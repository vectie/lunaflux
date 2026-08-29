#!/bin/sh

set -eu
LC_ALL=C
TZ=UTC
export LC_ALL TZ
umask 077

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/luna-bf16-kernel-producer-common.sh"

usage() {
  lbf_fail 'usage: build-luna-bf16-kernel-set.sh ABSOLUTE_NVCC ABSOLUTE_TOOLCHAIN_MANIFEST#sha256=HEX ABSOLUTE_CANDIDATE_ROOT ABSOLUTE_CANDIDATE_INVENTORY#sha256=HEX ABSOLUTE_NEW_OUTPUT'
}

[ "$#" -eq 5 ] || usage
nvcc=$1
toolchain_argument=$2
candidate_root=$3
inventory_argument=$4
output=$5

case "$toolchain_argument" in /*#sha256=*) ;; *) usage ;; esac
toolchain_manifest=${toolchain_argument%#sha256=*}
toolchain_sha=${toolchain_argument##*#sha256=}
case "$inventory_argument" in /*#sha256=*) ;; *) usage ;; esac
candidate_inventory=${inventory_argument%#sha256=*}
candidate_inventory_sha=${inventory_argument##*#sha256=}

lbf_require_absolute_file "$nvcc"
[ -x "$nvcc" ] || lbf_fail 'NVCC is not executable'
lbf_require_absolute_file "$toolchain_manifest"
lbf_require_absolute_directory "$candidate_root"
lbf_require_absolute_file "$candidate_inventory"
lbf_is_sha256 "$toolchain_sha" || lbf_fail 'toolchain-manifest digest is invalid'
lbf_is_sha256 "$candidate_inventory_sha" || lbf_fail 'candidate-inventory digest is invalid'
[ "$(lbf_sha256_file "$toolchain_manifest")" = "$toolchain_sha" ] ||
  lbf_fail 'toolchain-manifest digest does not match its bytes'
[ "$(lbf_sha256_file "$candidate_inventory")" = "$candidate_inventory_sha" ] ||
  lbf_fail 'candidate-inventory digest does not match its bytes'

case "$candidate_inventory" in "$candidate_root"/*)
  lbf_fail 'candidate inventory must be independent of the candidate root'
  ;;
esac
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

scratch=$(mktemp -d /tmp/lunaflux-bf16-producer-input.XXXXXX) ||
  lbf_fail 'could not create private validation scratch'
publish_container=$(mktemp -d "$output_parent/.lunaflux-bf16-producer.XXXXXX") ||
  lbf_fail 'could not create atomic publication scratch'
published=0
cleanup() {
  rm -rf -- "$scratch"
  if [ "$published" -ne 1 ]; then
    rm -rf -- "$publish_container"
  fi
}
trap cleanup EXIT HUP INT TERM

lbf_validate_candidate_inventory "$candidate_inventory" "$candidate_root" "$scratch"
candidate_set=$candidate_root/candidate-set.v1
lbf_require_newline "$candidate_set" 'candidate set'
[ "$(sed -n '1p' "$candidate_set")" = 'schema=lunaflux-bf16-candidate-set.v1' ] ||
  lbf_fail 'unsupported candidate-set schema'
target=$(sed -n '2s/^target=//p' "$candidate_set")
operation_count=$(sed -n '3s/^operation_count=//p' "$candidate_set")
lbf_is_target "$target" || lbf_fail 'candidate-set target is invalid'
lbf_is_uint "$operation_count" && [ "$operation_count" -ge 9 ] ||
  lbf_fail 'candidate set must contain at least the nine BF16 semantic families'
[ "$(wc -l <"$candidate_set" | tr -d ' ')" -eq "$((operation_count + 3))" ] ||
  lbf_fail 'candidate-set line count does not match operation_count'
[ "$(wc -l <"$candidate_inventory" | tr -d ' ')" -eq "$((operation_count * 2 + 1))" ] ||
  lbf_fail 'candidate inventory does not contain exactly one source/recipe pair per operation'

table=$scratch/candidates.table
: >"$table"
expected_id=0
while [ "$expected_id" -lt "$operation_count" ]; do
  line_number=$((expected_id + 4))
  line=$(sed -n "${line_number}p" "$candidate_set")
  case "$line" in candidate=*) value=${line#candidate=} ;; *)
    lbf_fail "candidate-set line $line_number is not canonical"
    ;;
  esac
  key=${value%%,*}
  remainder=${value#*,}
  operation_id=${remainder%%,*}
  family=${remainder#*,}
  [ "$family" != "$remainder" ] || lbf_fail 'candidate row has too few fields'
  case "$family" in *,*) lbf_fail 'candidate row has too many fields' ;; esac
  lbf_is_key "$key" || lbf_fail "candidate key is invalid: $key"
  [ "$operation_id" = "$expected_id" ] ||
    lbf_fail 'candidate operation IDs must be contiguous and ordered from zero'
  lbf_is_family "$family" || lbf_fail "candidate family is invalid: $family"
  if awk -F, -v key="$key" '$1 == key { found=1 } END { exit !found }' "$table"; then
    lbf_fail "candidate key is duplicated: $key"
  fi
  printf '%s,%s,%s\n' "$key" "$operation_id" "$family" >>"$table"
  expected_id=$((expected_id + 1))
done

for required_family in embedding_lookup rms_norm positioned_rotary residual_add \
  qkv_projection dense_projection gated_mlp language_model_head paged_attention; do
  awk -F, -v family="$required_family" '$3 == family { found=1 } END { exit !found }' "$table" ||
    lbf_fail "complete BF16 set is missing family: $required_family"
done

while IFS=, read -r key operation_id family; do
  grep -qx "[0-9a-f][0-9a-f]*  sources/$key.cu" "$candidate_inventory" ||
    lbf_fail "candidate source is absent from inventory: $key"
  grep -qx "[0-9a-f][0-9a-f]*  recipes/$key.recipe" "$candidate_inventory" ||
    lbf_fail "candidate recipe is absent from inventory: $key"
done <"$table"

driver_report=$scratch/driver.txt
"$repo_root/scripts/inspect-luna-cuda-aot-driver.sh" "$nvcc" >"$driver_report"
driver_identity_sha=$(sed -n '6s/^driver_identity_sha256=//p' "$driver_report")
compiler_version=$(sed -n '7s/^compiler_version=//p' "$driver_report")
[ "$(grep -c '^driver_identity_sha256=' "$toolchain_manifest")" -eq 1 ] ||
  lbf_fail 'approved toolchain manifest must contain one driver identity'
approved_driver_identity=$(sed -n 's/^driver_identity_sha256=//p' "$toolchain_manifest")
lbf_is_sha256 "$approved_driver_identity" ||
  lbf_fail 'approved driver identity is invalid'
[ "$driver_identity_sha" = "$approved_driver_identity" ] ||
  lbf_fail 'invoked CUDA driver is not bound by the approved toolchain manifest'

stage=$publish_container/result
mkdir "$stage"
mkdir "$stage/sha256" "$stage/receipts"
mkdir "$scratch/compile-cache"
operations_output=$scratch/operations.output
: >"$operations_output"

while IFS=, read -r key operation_id family; do
  source_file=$candidate_root/sources/$key.cu
  recipe_file=$candidate_root/recipes/$key.recipe
  source_sha=$(lbf_recipe_value source_sha256 "$recipe_file")
  recipe_toolchain_sha=$(lbf_recipe_value toolchain_sha256 "$recipe_file")
  recipe_compiler_version=$(lbf_recipe_value compiler_version "$recipe_file")
  recipe_target=$(lbf_recipe_value target "$recipe_file")
  recipe_output=$(lbf_recipe_value output "$recipe_file")
  optimization=$(lbf_recipe_value optimization "$recipe_file")
  fmad=$(lbf_recipe_value fmad "$recipe_file")
  reassociate=$(lbf_recipe_value reassociate "$recipe_file")
  max_registers=$(lbf_recipe_value max_registers "$recipe_file")
  recipe_operation_id=$(lbf_recipe_value operation_id "$recipe_file")
  function_symbol=$(lbf_recipe_value function_symbol "$recipe_file")
  manifest_bindable=$(lbf_recipe_value manifest_bindable "$recipe_file")
  schema=$(lbf_recipe_value schema "$recipe_file")

  case "$family" in
    embedding_lookup|rms_norm|qk_rms_norm|positioned_rotary|residual_add)
      [ "$schema" = lunaflux-luna-cuda-pointwise-aot-recipe-v1 ] ||
        lbf_fail "pointwise candidate has the wrong recipe schema: $key"
      [ "$(lbf_recipe_value family "$recipe_file")" = "$family" ] ||
        lbf_fail "pointwise family mismatch: $key"
      language_standard=c++14
      ;;
    qkv_projection|dense_projection|gated_mlp|language_model_head)
      [ "$schema" = lunaflux-luna-cuda-projection-aot-recipe-v1 ] ||
        lbf_fail "projection candidate has the wrong recipe schema: $key"
      [ "$(lbf_recipe_value family "$recipe_file")" = "$family" ] ||
        lbf_fail "projection family mismatch: $key"
      language_standard=c++14
      ;;
    paged_attention)
      [ "$schema" = lunaflux-paged-attention-cuda-aot-recipe-v1 ] ||
        lbf_fail "attention candidate has the wrong recipe schema: $key"
      language_standard=$(lbf_recipe_value language_standard "$recipe_file")
      ;;
  esac

  lbf_is_sha256 "$source_sha" || lbf_fail "source digest is invalid: $key"
  [ "$(lbf_sha256_file "$source_file")" = "$source_sha" ] ||
    lbf_fail "source digest mismatch: $key"
  [ "$recipe_toolchain_sha" = "$toolchain_sha" ] ||
    lbf_fail "toolchain digest mismatch: $key"
  [ "$recipe_compiler_version" = "$compiler_version" ] ||
    lbf_fail "compiler version mismatch: $key"
  [ "$recipe_target" = "$target" ] || lbf_fail "target mismatch: $key"
  [ "$recipe_output" = cubin ] || lbf_fail "non-CUBIN output is forbidden: $key"
  case "$optimization" in 0|1|2|3) ;; *) lbf_fail "optimization is invalid: $key" ;; esac
  case "$fmad" in true|false) ;; *) lbf_fail "fmad policy is invalid: $key" ;; esac
  [ "$reassociate" = false ] || lbf_fail "reassociation is forbidden: $key"
  lbf_is_uint "$max_registers" && [ "$max_registers" -ge 1 ] &&
    [ "$max_registers" -le 255 ] || lbf_fail "max_registers is invalid: $key"
  [ "$recipe_operation_id" = "$operation_id" ] ||
    lbf_fail "operation identity mismatch: $key"
  printf '%s\n' "$function_symbol" |
    awk '$0 !~ /^[A-Za-z_][A-Za-z0-9_]*$/ || length > 96 { exit 1 }' ||
    lbf_fail "function symbol is invalid: $key"
  grep -Fq "void $function_symbol(" "$source_file" ||
    lbf_fail "source does not define the recipe function symbol: $key"
  [ "$manifest_bindable" = false ] ||
    lbf_fail "pre-compile candidate claims final manifest authority: $key"
  case "$language_standard" in c++14|c++17) ;; *)
    lbf_fail "language standard is invalid: $key"
    ;;
  esac
  if grep -E -i '(^|[^a-z])(module_sha256|family_id|nvrtc|runtime[_-]?jit|developer[_-]?jit|output=ptx|\.ptx)([^a-z]|$)' \
    "$recipe_file" >/dev/null 2>&1; then
    lbf_fail "candidate recipe contains final-module, PTX, or JIT authority: $key"
  fi

  build_root=$scratch/build-$operation_id
  mkdir "$build_root"
  cp "$source_file" "$build_root/kernel.cu"
  chmod 444 "$build_root/kernel.cu"
  compute=${target#sm_}
  common_flags="--cubin --std=$language_standard --generate-code=arch=compute_${compute},code=$target -O$optimization --fmad=$fmad --ftz=false --prec-div=true --prec-sqrt=true --maxrregcount=$max_registers --Werror all-warnings"
  compile_identity=$build_root/compile-identity.v1
  {
    printf '%s\n' 'schema=lunaflux-bf16-compile-identity.v1'
    printf 'source_sha256=%s\n' "$source_sha"
    printf 'toolchain_sha256=%s\n' "$toolchain_sha"
    printf 'driver_identity_sha256=%s\n' "$driver_identity_sha"
    printf 'compiler_version=%s\n' "$compiler_version"
    printf 'target=%s\n' "$target"
    printf 'language_standard=%s\n' "$language_standard"
    printf 'output=%s\n' "$recipe_output"
    printf 'optimization=%s\n' "$optimization"
    printf 'fmad=%s\n' "$fmad"
    printf 'reassociate=%s\n' "$reassociate"
    printf 'max_registers=%s\n' "$max_registers"
    printf 'flags=%s\n' "$common_flags"
  } >"$compile_identity"
  compile_identity_sha=$(lbf_sha256_file "$compile_identity")
  cache_root=$scratch/compile-cache/$compile_identity_sha
  if [ -d "$cache_root" ]; then
    [ -f "$cache_root/compile-identity.v1" ] &&
      [ ! -L "$cache_root/compile-identity.v1" ] &&
      cmp -s "$compile_identity" "$cache_root/compile-identity.v1" ||
      lbf_fail "compile-cache identity collision: $key"
    [ -s "$cache_root/kernel.cubin" ] && [ ! -L "$cache_root/kernel.cubin" ] ||
      lbf_fail "compile-cache artifact is unavailable: $key"
    compiled_artifact=$cache_root/kernel.cubin
    first_sha=$(lbf_sha256_file "$compiled_artifact")
    second_sha=$first_sha
  else
    mkdir "$build_root/first" "$build_root/second"
    for build in first second; do
      cp "$build_root/kernel.cu" "$build_root/$build/kernel.cu"
      (
        cd "$build_root/$build"
        LC_ALL=C TZ=UTC SOURCE_DATE_EPOCH=0 CUDA_CACHE_DISABLE=1 \
          "$nvcc" $common_flags kernel.cu -o kernel.cubin >compiler.log 2>&1
      ) || lbf_fail "compiler invocation failed: $key/$build"
      [ -s "$build_root/$build/kernel.cubin" ] ||
        lbf_fail "compiler produced an empty CUBIN: $key/$build"
      [ ! -s "$build_root/$build/compiler.log" ] ||
        lbf_fail "compiler emitted diagnostics: $key/$build"
    done
    first_sha=$(lbf_sha256_file "$build_root/first/kernel.cubin")
    second_sha=$(lbf_sha256_file "$build_root/second/kernel.cubin")
    [ "$first_sha" = "$second_sha" ] &&
      cmp -s "$build_root/first/kernel.cubin" "$build_root/second/kernel.cubin" ||
      lbf_fail "independent compiler outputs differ: $key"
    mkdir "$cache_root"
    cp "$compile_identity" "$cache_root/compile-identity.v1"
    cp "$build_root/first/kernel.cubin" "$cache_root/kernel.cubin"
    compiled_artifact=$cache_root/kernel.cubin
  fi

  module_relative=sha256/$first_sha.cubin
  module_path=$stage/$module_relative
  if [ -e "$module_path" ]; then
    [ -f "$module_path" ] && [ ! -L "$module_path" ] &&
      cmp -s "$module_path" "$compiled_artifact" ||
      lbf_fail "content-addressed module collision: $key"
  else
    cp "$compiled_artifact" "$module_path"
  fi
  recipe_sha=$(lbf_sha256_file "$recipe_file")
  receipt_relative=receipts/$key.receipt.v1
  receipt=$stage/$receipt_relative
  {
    printf '%s\n' 'schema=lunaflux-bf16-offline-compile-receipt.v1'
    printf 'candidate_set_sha256=%s\n' "$(lbf_sha256_file "$candidate_set")"
    printf 'operation_id=%s\n' "$operation_id"
    printf 'family=%s\n' "$family"
    printf 'source_sha256=%s\n' "$source_sha"
    printf 'recipe_sha256=%s\n' "$recipe_sha"
    printf 'toolchain_sha256=%s\n' "$toolchain_sha"
    printf 'driver_identity_sha256=%s\n' "$driver_identity_sha"
    printf 'target=%s\n' "$target"
    printf 'first_artifact_sha256=%s\n' "$first_sha"
    printf 'second_artifact_sha256=%s\n' "$second_sha"
    printf 'module_relative_path=%s\n' "$module_relative"
    printf '%s\n' 'deterministic=1'
  } >"$receipt"
  receipt_sha=$(lbf_sha256_file "$receipt")
  printf 'operation=%s,%s,%s,%s,%s\n' \
    "$operation_id" "$key" "$family" "$first_sha" "$receipt_sha" >>"$operations_output"
done <"$table"

module_count=$(find "$stage/sha256" -type f -name '*.cubin' -print | wc -l | tr -d ' ')
{
  printf '%s\n' 'schema=lunaflux-bf16-compiled-kernel-set.v1'
  printf 'candidate_set_sha256=%s\n' "$(lbf_sha256_file "$candidate_set")"
  printf 'candidate_inventory_sha256=%s\n' "$candidate_inventory_sha"
  printf 'toolchain_sha256=%s\n' "$toolchain_sha"
  printf 'driver_identity_sha256=%s\n' "$driver_identity_sha"
  printf 'target=%s\n' "$target"
  printf 'operation_count=%s\n' "$operation_count"
  printf 'module_count=%s\n' "$module_count"
  cat "$operations_output"
} >"$stage/compiled-set.v1"
lbf_write_inventory "$stage" "$stage/files.sha256"
find "$stage" -type f -exec chmod 444 {} \;
find "$stage" -type d -exec chmod 555 {} \;
chmod 755 "$stage"

"$repo_root/scripts/verify-luna-bf16-kernel-set.sh" "$stage"
mv "$stage" "$output"
rmdir "$publish_container"
published=1
trap - EXIT HUP INT TERM
rm -rf -- "$scratch"
printf '%s\n' "Luna BF16 compiled kernel set produced: $output"
printf 'compiled_set_sha256=%s\n' "$(lbf_sha256_file "$output/compiled-set.v1")"
printf 'files_inventory_sha256=%s\n' "$(lbf_sha256_file "$output/files.sha256")"
