#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
TZ=UTC
export LC_ALL TZ
umask 077

fail() {
  printf 'LunaFlux FP8 v2 physical campaign rejected: %s\n' "$1" >&2
  exit 1
}

if [[ $# -lt 3 || $# -gt 4 ]]; then
  printf '%s\n' \
    "usage: $0 ABSOLUTE_NVCC_13_1 ABSOLUTE_COMPUTE_SANITIZER ABSOLUTE_NEW_EVIDENCE_DIR [CYCLES]" >&2
  exit 2
fi
nvcc=$1
sanitizer=$2
output=$3
cycles=${4:-8}

require_tool() {
  local path=$1 name=$2
  [[ $path == /* && -f $path && -x $path && ! -L $path ]] ||
    fail "$name is not an absolute executable regular file"
  [[ $(realpath -- "$path") == "$path" ]] || fail "$name path is not canonical"
}
require_tool "$nvcc" nvcc
require_tool "$sanitizer" compute-sanitizer
ptxas=$(dirname -- "$nvcc")/ptxas
require_tool "$ptxas" ptxas
[[ $cycles =~ ^[1-9][0-9]?$ ]] && (( cycles <= 32 )) ||
  fail 'cycles must be canonical decimal in 1..32'
[[ $output == /* ]] || fail 'evidence path is not absolute'
parent=${output%/*}
name=${output##*/}
[[ $name =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] ||
  fail 'evidence basename is not bounded'
[[ -d $parent && ! -L $parent && $(realpath -- "$parent") == "$parent" ]] ||
  fail 'evidence parent is not canonical'
[[ $output == "$parent/$name" && ! -e $output && ! -L $output ]] ||
  fail 'evidence path must be canonical and new'

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"
scripts/validate-fp8-v2-sm120-physical-probe.sh --static-only >/dev/null 2>&1

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

stage=$(mktemp -d "$parent/.${name}.partial.XXXXXX")
stage=$(CDPATH= cd -- "$stage" && pwd -P)
cleanup() {
  if [[ -n ${stage:-} && -d $stage ]]; then
    chmod -R u+rwX "$stage" 2>/dev/null || true
    rm -rf -- "$stage"
  fi
}
trap cleanup EXIT HUP INT TERM
mkdir -p "$stage/fixture" "$stage/builds" "$stage/measurements"

scripts/inspect-luna-cuda-aot-driver.sh "$nvcc" \
  >"$stage/measurements/driver-before.v1" \
  2>"$stage/measurements/driver-before.stderr"
[[ ! -s $stage/measurements/driver-before.stderr ]] || fail 'driver inspection emitted stderr'
compiler_version=$(sed -n 's/^compiler_version=//p' "$stage/measurements/driver-before.v1")
[[ $compiler_version == 13.1.115 ]] || fail 'compiler must be exact CUDA 13.1.115'
toolchain_sha=$(sha256_file "$nvcc")
sanitizer_sha=$(sha256_file "$sanitizer")

moon build tests/fp8_v2_cuda_probe --target native --deny-warn --warn-list +73 \
  >"$stage/measurements/moon-build.stdout" \
  2>"$stage/measurements/moon-build.stderr" || fail 'MoonBit probe build failed'
probe=$repo_root/_build/native/debug/build/tests/fp8_v2_cuda_probe/fp8_v2_cuda_probe.exe
[[ -f $probe && -x $probe && ! -L $probe ]] || fail 'probe executable is absent'
"$probe" export "$toolchain_sha" "$stage/fixture" \
  >"$stage/measurements/export.stdout" 2>"$stage/measurements/export.stderr"
[[ ! -s $stage/measurements/export.stderr ]] || fail 'fixture export emitted stderr'

fixture=$stage/fixture/fixture.v1
simple_source=$stage/fixture/simple.cu
gated_source=$stage/fixture/gated.cu
simple_source_sha=$(sed -n 's/^simple_source_sha256=//p' "$fixture")
gated_source_sha=$(sed -n 's/^gated_source_sha256=//p' "$fixture")
simple_symbol=$(sed -n 's/^simple_symbol=//p' "$fixture")
gated_symbol=$(sed -n 's/^gated_symbol=//p' "$fixture")
[[ $(sha256_file "$simple_source") == "$simple_source_sha" &&
   $(sha256_file "$gated_source") == "$gated_source_sha" ]] ||
  fail 'exported production source digest mismatch'
grep -F "void $simple_symbol(" "$simple_source" >/dev/null || fail 'simple ABI symbol drifted'
grep -F "void $gated_symbol(" "$gated_source" >/dev/null || fail 'gated ABI symbol drifted'
grep -Fx 'target=sm_120' "$fixture" >/dev/null
grep -Fx 'abi=production-paged-v4-operands-v2' "$fixture" >/dev/null
grep -Fx 'manifest_bindable=false' "$fixture" >/dev/null

compile_family() {
  local family=$1 source=$2 pass
  for pass in first second; do
    mkdir -p "$stage/builds/$family/$pass"
    cp "$source" "$stage/builds/$family/$pass/kernel.cu"
    (
      cd "$stage/builds/$family/$pass"
      SOURCE_DATE_EPOCH=0 CUDA_CACHE_DISABLE=1 "$nvcc" \
        --cubin --std=c++17 --generate-code=arch=compute_120,code=sm_120 \
        -O3 --fmad=false --ftz=false --prec-div=true --prec-sqrt=true \
        --maxrregcount=128 --Werror all-warnings kernel.cu -o kernel.cubin \
        >compiler.stdout 2>compiler.stderr
    ) || fail "$family $pass compilation failed"
    [[ -s $stage/builds/$family/$pass/kernel.cubin ]] || fail "$family CUBIN is empty"
  done
  cmp -s "$stage/builds/$family/first/kernel.cubin" \
    "$stage/builds/$family/second/kernel.cubin" || fail "$family CUBIN is nondeterministic"
}
compile_family simple "$simple_source"
compile_family gated "$gated_source"
simple_cubin=$stage/builds/simple/first/kernel.cubin
gated_cubin=$stage/builds/gated/first/kernel.cubin

run_sanitizer() {
  local tool=$1
  if [[ $tool == memcheck ]]; then
    "$sanitizer" --tool "$tool" --leak-check full --error-exitcode 99 \
      --log-file "$stage/measurements/$tool.log" \
      "$probe" run "$simple_cubin" "$gated_cubin" "$toolchain_sha" "$cycles" sm_120 \
      >"$stage/measurements/$tool-runtime.stdout" \
      2>"$stage/measurements/$tool-runtime.stderr"
  else
    "$sanitizer" --tool "$tool" --error-exitcode 99 \
      --log-file "$stage/measurements/$tool.log" \
      "$probe" run "$simple_cubin" "$gated_cubin" "$toolchain_sha" "$cycles" sm_120 \
      >"$stage/measurements/$tool-runtime.stdout" \
      2>"$stage/measurements/$tool-runtime.stderr"
  fi
  [[ ! -s $stage/measurements/$tool-runtime.stderr ]] || fail "$tool runtime emitted stderr"
  [[ $(grep -Fxc '========= ERROR SUMMARY: 0 errors' "$stage/measurements/$tool.log") == 1 ]] ||
    fail "$tool did not report exactly zero errors"
}
run_sanitizer memcheck
run_sanitizer racecheck
run_sanitizer initcheck
cmp -s "$stage/measurements/memcheck-runtime.stdout" \
  "$stage/measurements/racecheck-runtime.stdout" || fail 'sanitized numeric outcomes differ'
cmp -s "$stage/measurements/memcheck-runtime.stdout" \
  "$stage/measurements/initcheck-runtime.stdout" || fail 'sanitized numeric outcomes differ'
runtime=$stage/measurements/memcheck-runtime.stdout
[[ $(wc -l <"$runtime" | tr -d ' ') == 1 ]] || fail 'runtime outcome is ambiguous'
grep -Eq "^outcome=fp8-v2-sm120-physical-pass cycles=$cycles families=qkv,gated-mlp launches=$((cycles * 2)) numeric_values=$((cycles * 48)) qkv_max_abs_error=[^ ]+ gated_max_abs_error=[^ ]+ absolute_tolerance=0.03125 relative_tolerance=0.015625 cpu_oracle=independent-ordered-f32-v1 abi=production-paged-v4-operands-v2 diagnostic=scale-cells-finite-positive target=sm_120 device_ordinal=0 device_name_sha256=[0-9a-f]{64} device_total_memory_bytes=[1-9][0-9]* resources=context0,stream0,executor0,allocation0,module0,function0,device_bytes0,pending0,cleanup0 manifest_bindable=false readiness=false$" "$runtime" ||
  fail 'typed runtime outcome drifted'

scripts/inspect-luna-cuda-aot-driver.sh "$nvcc" \
  >"$stage/measurements/driver-after.v1" \
  2>"$stage/measurements/driver-after.stderr"
cmp -s "$stage/measurements/driver-before.v1" "$stage/measurements/driver-after.v1" ||
  fail 'driver/tool identity drifted'
[[ ! -s $stage/measurements/driver-after.stderr &&
   $(sha256_file "$nvcc") == "$toolchain_sha" &&
   $(sha256_file "$sanitizer") == "$sanitizer_sha" ]] || fail 'tool bytes drifted'

. "$repo_root/scripts/immutable-evidence-directory.sh"
lunaflux_prepare_evidence_manifest "$stage" || fail 'inner evidence manifest failed'
inner_sha=$lunaflux_evidence_manifest_sha256
printf '%s\n' \
  'schema=lunaflux-fp8-v2-sm120-physical-campaign.v1' \
  'outcome=fp8-v2-sm120-physical-campaign-pass' \
  'target=sm_120' 'compiler_version=13.1.115' \
  "nvcc_sha256=$toolchain_sha" "compute_sanitizer_sha256=$sanitizer_sha" \
  "driver_record_sha256=$(sha256_file "$stage/measurements/driver-before.v1")" \
  "fixture_sha256=$(sha256_file "$fixture")" \
  "simple_source_sha256=$simple_source_sha" \
  "gated_source_sha256=$gated_source_sha" \
  "simple_cubin_sha256=$(sha256_file "$simple_cubin")" \
  "gated_cubin_sha256=$(sha256_file "$gated_cubin")" \
  "cycles=$cycles" "launches=$((cycles * 2))" \
  'families=qkv,gated-mlp' 'cpu_oracle=independent-ordered-f32-v1' \
  'absolute_tolerance=0.03125' 'relative_tolerance=0.015625' \
  'memcheck_errors=0' 'racecheck_errors=0' 'initcheck_errors=0' \
  'cleanup_balance=context0,stream0,executor0,allocation0,module0,function0,device_bytes0,pending0,cleanup0' \
  'physical_cuda_observed=true' 'qualification_only=true' \
  'manifest_bindable=false' 'readiness=false' \
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
printf 'outcome=fp8-v2-sm120-physical-campaign-published inner_seal_sha256=%s outer_seal_sha256=%s authority=qualification-only\n' "$inner_sha" "$outer_sha"
