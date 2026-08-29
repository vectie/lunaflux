#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
export LC_ALL

if [[ $# -ne 2 ]]; then
  printf '%s\n' "usage: $0 ABSOLUTE_NVCC_13_1_115 ABSOLUTE_NEW_OUTPUT" >&2
  exit 2
fi
nvcc=$1
output=$2

fail() {
  printf 'paged-attention read-only source campaign rejected: %s\n' "$1" >&2
  exit 1
}
[[ $nvcc == /* && -f $nvcc && ! -L $nvcc && -x $nvcc ]] || fail 'NVCC is not an absolute executable file'
[[ $(realpath -- "$nvcc") == "$nvcc" ]] || fail 'NVCC is not canonical'
ptxas=$(dirname -- "$nvcc")/ptxas
[[ -f $ptxas && ! -L $ptxas && -x $ptxas && $(realpath -- "$ptxas") == "$ptxas" ]] ||
  fail 'sibling ptxas is not canonical executable'
[[ $output == /* ]] || fail 'output is not absolute'
parent=${output%/*}
name=${output##*/}
[[ $name =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail 'output name is not bounded'
[[ -d $parent && ! -L $parent && $(realpath -- "$parent") == "$parent" ]] || fail 'output parent is not canonical'
[[ $output == "$parent/$name" && ! -e $output && ! -L $output ]] || fail 'output is not canonical and new'

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}
nvcc_sha=$(sha256_file "$nvcc")
ptxas_sha=$(sha256_file "$ptxas")
stage=$(mktemp -d "$parent/.${name}.partial.XXXXXX")
stage=$(CDPATH= cd -- "$stage" && pwd -P)
cleanup() {
  if [[ -n ${stage:-} && -d $stage ]]; then
    chmod -R u+rwX "$stage" 2>/dev/null || true
    rm -rf -- "$stage"
  fi
}
trap cleanup EXIT HUP INT TERM
mkdir -p "$stage/export" "$stage/builds/first" "$stage/builds/second" "$stage/measurements"

"$nvcc" --version >"$stage/measurements/nvcc-version-before.stdout" \
  2>"$stage/measurements/nvcc-version-before.stderr"
"$ptxas" --version >"$stage/measurements/ptxas-version-before.stdout" \
  2>"$stage/measurements/ptxas-version-before.stderr"
[[ ! -s $stage/measurements/nvcc-version-before.stderr &&
   ! -s $stage/measurements/ptxas-version-before.stderr ]] || fail 'tool version emitted stderr'
grep -F 'release 13.1,' "$stage/measurements/nvcc-version-before.stdout" >/dev/null || fail 'NVCC release mismatch'
scripts/inspect-luna-cuda-aot-driver.sh "$nvcc" \
  >"$stage/measurements/driver-before.v1" 2>"$stage/measurements/driver-before.stderr"
[[ ! -s $stage/measurements/driver-before.stderr ]] || fail 'driver inspection emitted stderr'
compiler_version=$(sed -n 's/^compiler_version=//p' "$stage/measurements/driver-before.v1")
[[ $compiler_version == 13.1.115 ]] || fail 'compiler is not exact 13.1.115'
driver_identity=$(sed -n 's/^driver_identity_sha256=//p' "$stage/measurements/driver-before.v1")

moon build tests/paged_attention_readonly_source_export --target native --deny-warn --warn-list +73 \
  >"$stage/measurements/moon-build.stdout" 2>"$stage/measurements/moon-build.stderr"
exporter=$root/_build/native/debug/build/tests/paged_attention_readonly_source_export/paged_attention_readonly_source_export.exe
[[ -f $exporter && ! -L $exporter && -x $exporter ]] || fail 'source exporter executable is missing'
"$exporter" export "$stage/export" "$driver_identity" "$compiler_version" >"$stage/measurements/export.stdout" \
  2>"$stage/measurements/export.stderr"
[[ ! -s $stage/measurements/export.stderr ]] || fail 'source exporter emitted stderr'
record=$stage/export/export.v1
source=$stage/export/readonly.cu
candidate_sha=$(sed -n 's/^candidate_sha256=//p' "$record")
source_sha=$(sed -n 's/^source_sha256=//p' "$record")
recipe_sha=$(sed -n 's/^recipe_sha256=//p' "$record")
fallback_source_sha=$(sed -n 's/^fallback_source_sha256=//p' "$record")
fallback_recipe_sha=$(sed -n 's/^fallback_recipe_sha256=//p' "$record")
toolchain_sha=$(sed -n 's/^toolchain_sha256=//p' "$stage/export/readonly.recipe")
symbol=$(sed -n 's/^function_symbol=//p' "$record")
abi=$(sed -n 's/^raw_pointer_abi=//p' "$record")
input_row_width=$(sed -n 's/^input_row_width=//p' "$record")
input_activation=$(sed -n 's/^input_activation_ref=//p' "$record")
dispatch_canary=$(sed -n 's/^dispatch_canary_per_token=//p' "$record")
canary_cells=$(sed -n 's/^dispatch_canary_cell_count=//p' "$record")
for digest in "$candidate_sha" "$source_sha" "$recipe_sha" \
  "$fallback_source_sha" "$fallback_recipe_sha" "$toolchain_sha" "$driver_identity"; do
  [[ $digest =~ ^[0-9a-f]{64}$ ]] || fail 'export identity digest is malformed'
done
[[ $(sha256_file "$source") == "$source_sha" &&
   $(sha256_file "$stage/export/readonly.recipe") == "$recipe_sha" &&
   $(sha256_file "$stage/export/fallback.cu") == "$fallback_source_sha" &&
   $(sha256_file "$stage/export/fallback.recipe") == "$fallback_recipe_sha" ]] ||
  fail 'export digest mismatch'
grep -F "void $symbol(" "$source" >/dev/null || fail 'function symbol is absent'
grep -F 'const __nv_bfloat16 *key_cache' "$source" >/dev/null
grep -F 'const __nv_bfloat16 *value_cache' "$source" >/dev/null
if grep -E '(key_cache|value_cache)\[[^]]+\][[:space:]]*=' "$source"; then
  fail 'read-only source contains KV mutation'
fi

for build in first second; do
  cp "$source" "$stage/builds/$build/kernel.cu"
  (
    cd "$stage/builds/$build"
    TZ=UTC SOURCE_DATE_EPOCH=0 CUDA_CACHE_DISABLE=1 \
      "$nvcc" --cubin --std=c++17 --generate-code=arch=compute_120,code=sm_120 \
      -O3 --fmad=false --ftz=false --prec-div=true --prec-sqrt=true \
      --maxrregcount=128 --Werror all-warnings kernel.cu -o kernel.cubin \
      >compiler.stdout 2>compiler.stderr
  ) || fail "$build compilation failed"
  [[ -s $stage/builds/$build/kernel.cubin ]] || fail "$build CUBIN is empty"
done
first=$stage/builds/first/kernel.cubin
second=$stage/builds/second/kernel.cubin
cmp -s "$first" "$second" || fail 'independent CUBIN bytes differ'
module_sha=$(sha256_file "$first")
printf '%s\n' \
  'schema=lunaflux-paged-attention-readonly-aot-evidence.v1' \
  "candidate_sha256=$candidate_sha" "source_sha256=$source_sha" \
  "recipe_sha256=$recipe_sha" "fallback_source_sha256=$fallback_source_sha" \
  "fallback_recipe_sha256=$fallback_recipe_sha" "toolchain_sha256=$toolchain_sha" \
  "driver_identity_sha256=$driver_identity" 'target=sm_120' \
  "function_symbol=$symbol" "raw_pointer_abi=$abi" \
  "input_row_width=$input_row_width" "input_activation_ref=$input_activation" \
  "dispatch_canary_per_token=$dispatch_canary" \
  "dispatch_canary_cell_count=$canary_cells" \
  "artifact_sha256=$module_sha" "first_build_sha256=$module_sha" \
  "second_build_sha256=$module_sha" 'deterministic=1' 'kv_cache_mutation=none' \
  >"$stage/measurements/compile-receipt.v1"
receipt_sha=$(sha256_file "$stage/measurements/compile-receipt.v1")
"$exporter" bind "$first" "$second" "$stage/measurements/compile-receipt.v1" \
  "$receipt_sha" "$driver_identity" "$compiler_version" >"$stage/measurements/bind.stdout" \
  2>"$stage/measurements/bind.stderr"
[[ ! -s $stage/measurements/bind.stderr ]] || fail 'compiled binding emitted stderr'
grep -F 'outcome=paged-attention-readonly-compiled-candidate-bound ' \
  "$stage/measurements/bind.stdout" >/dev/null || fail 'compiled binding result is absent'
binding_sha=$(sed -n 's/.* binding_sha256=\([^ ]*\).*/\1/p' "$stage/measurements/bind.stdout")
[[ $binding_sha =~ ^[0-9a-f]{64}$ ]] || fail 'binding digest is malformed'

"$nvcc" --version >"$stage/measurements/nvcc-version-after.stdout" \
  2>"$stage/measurements/nvcc-version-after.stderr"
"$ptxas" --version >"$stage/measurements/ptxas-version-after.stdout" \
  2>"$stage/measurements/ptxas-version-after.stderr"
scripts/inspect-luna-cuda-aot-driver.sh "$nvcc" \
  >"$stage/measurements/driver-after.v1" 2>"$stage/measurements/driver-after.stderr"
[[ ! -s $stage/measurements/nvcc-version-after.stderr &&
   ! -s $stage/measurements/ptxas-version-after.stderr &&
   ! -s $stage/measurements/driver-after.stderr ]] || fail 'tool reinspection emitted stderr'
cmp -s "$stage/measurements/nvcc-version-before.stdout" "$stage/measurements/nvcc-version-after.stdout" || fail 'NVCC identity drifted'
cmp -s "$stage/measurements/ptxas-version-before.stdout" "$stage/measurements/ptxas-version-after.stdout" || fail 'ptxas identity drifted'
cmp -s "$stage/measurements/driver-before.v1" "$stage/measurements/driver-after.v1" || fail 'driver identity drifted'
[[ $(sha256_file "$nvcc") == "$nvcc_sha" && $(sha256_file "$ptxas") == "$ptxas_sha" ]] || fail 'tool bytes drifted'

. "$root/scripts/immutable-evidence-directory.sh"
lunaflux_prepare_evidence_manifest "$stage" || fail 'evidence manifest failed'
inner_sha=$lunaflux_evidence_manifest_sha256
printf '%s\n' \
  'schema=lunaflux-paged-attention-readonly-source-campaign.v1' \
  'outcome=paged-attention-readonly-source-campaign-pass' \
  "candidate_sha256=$candidate_sha" "source_sha256=$source_sha" \
  "recipe_sha256=$recipe_sha" "fallback_source_sha256=$fallback_source_sha" \
  "fallback_recipe_sha256=$fallback_recipe_sha" "artifact_sha256=$module_sha" \
  "compile_receipt_sha256=$receipt_sha" "compiled_binding_sha256=$binding_sha" \
  'target=sm_120' "compiler_version=$compiler_version" "nvcc_sha256=$nvcc_sha" \
  "ptxas_sha256=$ptxas_sha" "driver_identity_sha256=$driver_identity" \
  "function_symbol=$symbol" "raw_pointer_abi=$abi" \
  "input_row_width=$input_row_width" "dispatch_canary_per_token=$dispatch_canary" \
  "input_activation_ref=$input_activation" \
  "dispatch_canary_cell_count=$canary_cells" \
  'selection_precondition=standalone-positioned-rope-paged-kvwrite-complete-v1' \
  'kv_cache_mutation=none' 'source_only=true' 'physical_cuda_observed=false' \
  'qualification_only=true' 'manifest_bindable=false' 'promotion_authority=absent' \
  "evidence_files_manifest_sha256=$inner_sha" >"$stage/RESULT.txt"
{
  printf '%s  FILES.sha256\n' "$(sha256_file "$stage/FILES.sha256")"
  printf '%s  RESULT.txt\n' "$(sha256_file "$stage/RESULT.txt")"
} >"$stage/OUTER_SEAL.sha256"
outer_sha=$(sha256_file "$stage/OUTER_SEAL.sha256")
lunaflux_seal_evidence_directory "$stage" || fail 'evidence seal failed'
chmod 0755 "$stage"
mv -- "$stage" "$output"
chmod 0555 "$output"
stage=
trap - EXIT HUP INT TERM
printf '%s\n' \
  "outcome=paged-attention-readonly-source-campaign-published inner_seal_sha256=$inner_sha outer_seal_sha256=$outer_sha authority=qualification-only physical_cuda_observed=false"
