#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
export LC_ALL

if [[ $# -lt 3 || $# -gt 4 ]]; then
  printf '%s\n' \
    "usage: $0 ABSOLUTE_NVCC_13_1 ABSOLUTE_COMPUTE_SANITIZER ABSOLUTE_NEW_OUTPUT [CYCLES]" >&2
  exit 2
fi

nvcc=$1
sanitizer=$2
output=$3
cycles=${4:-8}

fail() {
  printf 'LunaTile parallel physical campaign rejected: %s\n' "$1" >&2
  exit 1
}

require_file() {
  local path=$1 description=$2
  [[ $path == /* && -f $path && ! -L $path && -x $path ]] ||
    fail "$description is not an absolute executable regular file"
  [[ $(realpath -- "$path") == "$path" ]] || fail "$description is not canonical"
}

require_file "$nvcc" NVCC
require_file "$sanitizer" compute-sanitizer
[[ $cycles =~ ^[1-9][0-9]?$ ]] || fail 'CYCLES is not canonical decimal'
(( cycles <= 32 )) || fail 'CYCLES is outside 1..32'
[[ $output == /* ]] || fail 'output path is not absolute'
parent=${output%/*}
name=${output##*/}
[[ $name =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] ||
  fail 'output basename is not a bounded evidence name'
[[ -d $parent && ! -L $parent && $(realpath -- "$parent") == "$parent" ]] ||
  fail 'output parent is not canonical'
[[ $output == "$parent/$name" && ! -e $output && ! -L $output ]] ||
  fail 'output path is not canonical and new'

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"
scripts/validate-luna-tile-parallel-cuda-probe.sh >/dev/null 2>&1

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

nvcc_sha=$(sha256_file "$nvcc")
sanitizer_sha=$(sha256_file "$sanitizer")
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

"$nvcc" --version >"$stage/measurements/nvcc-version-before.stdout" \
  2>"$stage/measurements/nvcc-version-before.stderr"
[[ ! -s $stage/measurements/nvcc-version-before.stderr ]] ||
  fail 'NVCC version emitted stderr'
grep -F 'release 13.1,' "$stage/measurements/nvcc-version-before.stdout" >/dev/null ||
  fail 'NVCC release is not 13.1'
scripts/inspect-luna-cuda-aot-driver.sh "$nvcc" \
  >"$stage/measurements/driver-before.stdout" \
  2>"$stage/measurements/driver-before.stderr"
[[ ! -s $stage/measurements/driver-before.stderr ]] ||
  fail 'CUDA driver identity emitted stderr'
"$sanitizer" --version >"$stage/measurements/sanitizer-version.stdout" \
  2>"$stage/measurements/sanitizer-version.stderr"
[[ ! -s $stage/measurements/sanitizer-version.stderr ]] ||
  fail 'compute-sanitizer version emitted stderr'

moon build --target native --deny-warn --warn-list +73 \
  tests/luna_tile_parallel_cuda_probe \
  >"$stage/measurements/moon-build.stdout" \
  2>"$stage/measurements/moon-build.stderr"
probe=$repo_root/_build/native/debug/build/tests/luna_tile_parallel_cuda_probe/luna_tile_parallel_cuda_probe.exe
[[ -x $probe && -f $probe && ! -L $probe ]] || fail 'probe executable is missing'
"$probe" export "$stage/fixture" >"$stage/measurements/export.stdout" \
  2>"$stage/measurements/export.stderr"
[[ ! -s $stage/measurements/export.stderr ]] || fail 'fixture export emitted stderr'

fixture=$stage/fixture/fixture.v1
serial_source=$stage/fixture/serial.cu
parallel_source=$stage/fixture/parallel.cu
[[ $(wc -l <"$fixture" | tr -d ' ') == 19 ]] || fail 'fixture field count drifted'
serial_source_sha=$(sed -n 's/^serial_source_sha256=//p' "$fixture")
parallel_source_sha=$(sed -n 's/^parallel_source_sha256=//p' "$fixture")
parallel_candidate_sha=$(sed -n 's/^parallel_candidate_sha256=//p' "$fixture")
serial_translation_sha=$(sed -n 's/^serial_translation_sha256=//p' "$fixture")
serial_symbol=$(sed -n 's/^serial_entry_point=//p' "$fixture")
parallel_symbol=$(sed -n 's/^parallel_entry_point=//p' "$fixture")
[[ $(sha256_file "$serial_source") == "$serial_source_sha" &&
   $(sha256_file "$parallel_source") == "$parallel_source_sha" ]] ||
  fail 'exported source digest does not match fixture'
[[ $(sha256_file "$stage/fixture/serial.canonical") == "$serial_translation_sha" &&
   $(sha256_file "$stage/fixture/parallel.canonical") == "$parallel_candidate_sha" ]] ||
  fail 'exported canonical digest does not match typed identity'
grep -F "void $serial_symbol(" "$serial_source" >/dev/null ||
  fail 'serial source lost its exact entry point'
grep -F "void $parallel_symbol(" "$parallel_source" >/dev/null ||
  fail 'parallel source lost its exact entry point'
grep -Fx 'target=sm_120' "$fixture" >/dev/null
grep -Fx 'manifest_bindable=false' "$fixture" >/dev/null
grep -Fx 'promotion_authority=absent' "$fixture" >/dev/null
grep -F "source_sha256=$parallel_source_sha" \
  "$stage/fixture/parallel.canonical" >/dev/null ||
  fail 'parallel canonical identity lost source binding'

compile_pair() {
  local family=$1 source=$2 first second
  mkdir -p "$stage/builds/$family/first" "$stage/builds/$family/second"
  for build in first second; do
    cp "$source" "$stage/builds/$family/$build/kernel.cu"
    (
      cd "$stage/builds/$family/$build"
      TZ=UTC SOURCE_DATE_EPOCH=0 CUDA_CACHE_DISABLE=1 \
        "$nvcc" --cubin --std=c++14 \
        --generate-code=arch=compute_120,code=sm_120 -O2 --fmad=false \
        --ftz=false --prec-div=true --prec-sqrt=true --maxrregcount=128 \
        --Werror all-warnings kernel.cu -o kernel.cubin \
        >compiler.stdout 2>compiler.stderr
    ) || fail "$family $build compilation failed"
    [[ -s $stage/builds/$family/$build/kernel.cubin ]] ||
      fail "$family $build CUBIN is empty"
  done
  first=$stage/builds/$family/first/kernel.cubin
  second=$stage/builds/$family/second/kernel.cubin
  cmp -s "$first" "$second" || fail "$family independent CUBIN bytes differ"
}

compile_pair serial "$serial_source"
compile_pair parallel "$parallel_source"
serial_cubin=$stage/builds/serial/first/kernel.cubin
parallel_cubin=$stage/builds/parallel/first/kernel.cubin
serial_cubin_sha=$(sha256_file "$serial_cubin")
parallel_cubin_sha=$(sha256_file "$parallel_cubin")

runtime=$stage/measurements/runtime.stdout
runtime_err=$stage/measurements/runtime.stderr
memcheck=$stage/measurements/memcheck.log
"$sanitizer" --tool memcheck --leak-check full --error-exitcode 99 \
  --log-file "$memcheck" "$probe" run "$serial_cubin" "$parallel_cubin" \
  "$cycles" >"$runtime" 2>"$runtime_err"
[[ ! -s $runtime_err ]] || fail 'memcheck runtime emitted stderr'
[[ $(grep -Fc 'ERROR SUMMARY:' "$memcheck") -eq 1 &&
   $(grep -Fxc '========= ERROR SUMMARY: 0 errors' "$memcheck") -eq 1 ]] ||
  fail 'memcheck summary is not uniquely clean'

race_runtime=$stage/measurements/racecheck.stdout
race_err=$stage/measurements/racecheck.stderr
race_log=$stage/measurements/racecheck.log
"$sanitizer" --tool racecheck --error-exitcode 99 --log-file "$race_log" \
  "$probe" run "$serial_cubin" "$parallel_cubin" "$cycles" \
  >"$race_runtime" 2>"$race_err"
[[ ! -s $race_err ]] || fail 'racecheck runtime emitted stderr'
[[ $(grep -Fc 'ERROR SUMMARY:' "$race_log") -eq 1 &&
   $(grep -Fxc '========= ERROR SUMMARY: 0 errors' "$race_log") -eq 1 ]] ||
  fail 'racecheck summary is not uniquely clean'
cmp -s "$runtime" "$race_runtime" || fail 'sanitizer observations differ'
expected="outcome=lunatile-parallel-sm120-qualification-pass cycles=$cycles launches=$((cycles * 2)) bytes_per_launch=256 serial_source_sha256=$serial_source_sha parallel_source_sha256=$parallel_source_sha parallel_candidate_sha256=$parallel_candidate_sha target=sm_120 resources=context0,stream0,allocation0,module0,function0,device_bytes0,pending0,cleanup0 manifest_bindable=false promotion_authority=absent"
grep -Fx "$expected" "$runtime" >/dev/null || fail 'typed runtime outcome drifted'
[[ $(wc -l <"$runtime" | tr -d ' ') == 1 ]] || fail 'runtime outcome is ambiguous'

"$nvcc" --version >"$stage/measurements/nvcc-version-after.stdout" \
  2>"$stage/measurements/nvcc-version-after.stderr"
scripts/inspect-luna-cuda-aot-driver.sh "$nvcc" \
  >"$stage/measurements/driver-after.stdout" \
  2>"$stage/measurements/driver-after.stderr"
cmp -s "$stage/measurements/nvcc-version-before.stdout" \
  "$stage/measurements/nvcc-version-after.stdout" || fail 'NVCC output drifted'
cmp -s "$stage/measurements/driver-before.stdout" \
  "$stage/measurements/driver-after.stdout" || fail 'CUDA driver identity drifted'
[[ ! -s $stage/measurements/nvcc-version-after.stderr &&
   ! -s $stage/measurements/driver-after.stderr &&
   $(sha256_file "$nvcc") == "$nvcc_sha" &&
   $(sha256_file "$sanitizer") == "$sanitizer_sha" ]] ||
  fail 'tool identity drifted'

printf '%s\n' \
  'outcome=lunatile-parallel-physical-campaign-pass' \
  "fixture_sha256=$(sha256_file "$fixture")" \
  "serial_source_sha256=$serial_source_sha" \
  "parallel_source_sha256=$parallel_source_sha" \
  "parallel_candidate_sha256=$parallel_candidate_sha" \
  "serial_cubin_sha256=$serial_cubin_sha" \
  "parallel_cubin_sha256=$parallel_cubin_sha" \
  'physical_cuda_observed=true' \
  'manifest_bindable=false' \
  'promotion_authority=absent' >"$stage/CAMPAIGN_RESULT.txt"
. "$repo_root/scripts/immutable-evidence-directory.sh"
lunaflux_prepare_evidence_manifest "$stage" || fail 'evidence manifest failed'
seal_sha=$lunaflux_evidence_manifest_sha256
lunaflux_seal_evidence_directory "$stage" || fail 'evidence seal failed'
chmod 0755 "$stage"
mv -- "$stage" "$output"
chmod 0555 "$output"
stage=
trap - EXIT HUP INT TERM
printf '%s\n' \
  "outcome=lunatile-parallel-physical-campaign-published evidence_seal_sha256=$seal_sha authority=qualification-only"
