#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
export LC_ALL

if [[ $# -lt 5 || $# -gt 6 ]]; then
  printf '%s\n' \
    "usage: $0 ABSOLUTE_NVCC_13_1 ABSOLUTE_COMPUTE_SANITIZER ABSOLUTE_CUOBJDUMP ABSOLUTE_NVDISASM ABSOLUTE_NEW_OUTPUT [CYCLES]" >&2
  exit 2
fi

nvcc=$1
sanitizer=$2
cuobjdump=$3
nvdisasm=$4
output=$5
cycles=${6:-8}

fail() {
  printf 'LunaTile tensor-core physical campaign rejected: %s\n' "$1" >&2
  exit 1
}

require_tool() {
  local path=$1 description=$2
  [[ $path == /* && -f $path && ! -L $path && -x $path ]] ||
    fail "$description is not an absolute executable regular file"
  [[ $(realpath -- "$path") == "$path" ]] || fail "$description is not canonical"
}

require_tool "$nvcc" NVCC
require_tool "$sanitizer" compute-sanitizer
require_tool "$cuobjdump" cuobjdump
require_tool "$nvdisasm" nvdisasm
ptxas=$(dirname -- "$nvcc")/ptxas
require_tool "$ptxas" ptxas
[[ $cycles =~ ^[1-9][0-9]?$ ]] || fail 'CYCLES is not canonical decimal'
(( cycles <= 32 )) || fail 'CYCLES is outside 1..32'
[[ $output == /* ]] || fail 'output path is not absolute'
parent=${output%/*}
name=${output##*/}
[[ $name =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail 'output basename is not bounded'
[[ -d $parent && ! -L $parent && $(realpath -- "$parent") == "$parent" ]] ||
  fail 'output parent is not canonical'
[[ $output == "$parent/$name" && ! -e $output && ! -L $output ]] ||
  fail 'output path is not canonical and new'

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"
scripts/validate-luna-tensor-core-cuda-probe.sh >/dev/null 2>&1

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

for tool_pair in \
  "nvcc:$nvcc" "ptxas:$ptxas" "sanitizer:$sanitizer" \
  "cuobjdump:$cuobjdump" "nvdisasm:$nvdisasm"; do
  tool_name=${tool_pair%%:*}
  tool_path=${tool_pair#*:}
  eval "${tool_name}_sha=\$(sha256_file \"\$tool_path\")"
done

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

capture_version() {
  local tool=$1 stem=$2
  "$tool" --version >"$stage/measurements/$stem-version-before.stdout" \
    2>"$stage/measurements/$stem-version-before.stderr"
  [[ ! -s $stage/measurements/$stem-version-before.stderr ]] ||
    fail "$stem version emitted stderr"
}
capture_version "$nvcc" nvcc
grep -F 'release 13.1,' "$stage/measurements/nvcc-version-before.stdout" >/dev/null ||
  fail 'NVCC release is not 13.1'
capture_version "$ptxas" ptxas
capture_version "$sanitizer" compute-sanitizer
capture_version "$cuobjdump" cuobjdump
capture_version "$nvdisasm" nvdisasm
scripts/inspect-luna-cuda-aot-driver.sh "$nvcc" \
  >"$stage/measurements/driver-before.v1" \
  2>"$stage/measurements/driver-before.stderr"
[[ ! -s $stage/measurements/driver-before.stderr ]] || fail 'driver identity emitted stderr'

moon build --target native --deny-warn --warn-list +73 \
  tests/luna_tensor_core_cuda_probe \
  >"$stage/measurements/moon-build.stdout" \
  2>"$stage/measurements/moon-build.stderr"
probe=$repo_root/_build/native/debug/build/tests/luna_tensor_core_cuda_probe/luna_tensor_core_cuda_probe.exe
[[ -x $probe && -f $probe && ! -L $probe ]] || fail 'probe executable is missing'
"$probe" export "$stage/fixture" >"$stage/measurements/export.stdout" \
  2>"$stage/measurements/export.stderr"
[[ ! -s $stage/measurements/export.stderr ]] || fail 'fixture export emitted stderr'

fixture=$stage/fixture/fixture.v1
serial_source=$stage/fixture/serial.cu
tensor_source=$stage/fixture/tensor_core.cu
program_sha=$(sed -n 's/^program_sha256=//p' "$fixture")
serial_translation_sha=$(sed -n 's/^serial_translation_sha256=//p' "$fixture")
parallel_plan_sha=$(sed -n 's/^parallel_plan_sha256=//p' "$fixture")
serial_source_sha=$(sed -n 's/^serial_source_sha256=//p' "$fixture")
tensor_candidate_sha=$(sed -n 's/^tensor_core_candidate_sha256=//p' "$fixture")
tensor_source_sha=$(sed -n 's/^tensor_core_source_sha256=//p' "$fixture")
serial_symbol=$(sed -n 's/^serial_entry_point=//p' "$fixture")
tensor_symbol=$(sed -n 's/^tensor_core_entry_point=//p' "$fixture")
for digest in "$program_sha" "$serial_translation_sha" "$parallel_plan_sha" \
  "$serial_source_sha" "$tensor_candidate_sha" "$tensor_source_sha"; do
  [[ $digest =~ ^[0-9a-f]{64}$ ]] || fail 'fixture identity digest is malformed'
done
[[ $(sha256_file "$serial_source") == "$serial_source_sha" &&
   $(sha256_file "$tensor_source") == "$tensor_source_sha" ]] ||
  fail 'exported source digest mismatch'
[[ $(sha256_file "$stage/fixture/serial.canonical") == "$serial_translation_sha" &&
   $(sha256_file "$stage/fixture/tensor_core.canonical") == "$tensor_candidate_sha" ]] ||
  fail 'exported canonical digest mismatch'
grep -F "void $serial_symbol(" "$serial_source" >/dev/null || fail 'serial symbol drifted'
grep -F "void $tensor_symbol(" "$tensor_source" >/dev/null || fail 'tensor symbol drifted'
grep -Fx 'target=sm_120' "$fixture" >/dev/null
grep -Fx 'manifest_bindable=false' "$fixture" >/dev/null
grep -Fx 'promotion_authority=absent' "$fixture" >/dev/null

compile_pair() {
  local family=$1 source=$2 build
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
    [[ -s $stage/builds/$family/$build/kernel.cubin ]] || fail "$family CUBIN is empty"
  done
  cmp -s "$stage/builds/$family/first/kernel.cubin" \
    "$stage/builds/$family/second/kernel.cubin" || fail "$family CUBIN bytes differ"
}
compile_pair serial "$serial_source"
compile_pair tensor_core "$tensor_source"
serial_cubin=$stage/builds/serial/first/kernel.cubin
tensor_cubin=$stage/builds/tensor_core/first/kernel.cubin
serial_cubin_sha=$(sha256_file "$serial_cubin")
tensor_cubin_sha=$(sha256_file "$tensor_cubin")

"$cuobjdump" --dump-sass "$tensor_cubin" \
  >"$stage/measurements/cuobjdump-sass.stdout" \
  2>"$stage/measurements/cuobjdump-sass.stderr"
"$nvdisasm" "$tensor_cubin" >"$stage/measurements/nvdisasm-sass.stdout" \
  2>"$stage/measurements/nvdisasm-sass.stderr"
[[ ! -s $stage/measurements/cuobjdump-sass.stderr &&
   ! -s $stage/measurements/nvdisasm-sass.stderr ]] || fail 'SASS tools emitted stderr'
cu_sass=$stage/measurements/cuobjdump-sass.stdout
nv_sass=$stage/measurements/nvdisasm-sass.stdout
extract_sass_families() {
  sed -nE \
    's@^[[:space:]]*/\*(0x)?[0-9A-Fa-f]+\*/[[:space:]]+(TCGEN05\.MMA|HMMA|MMA)(\.|[[:space:]]).*@\2@p' \
    "$1"
}
extract_sass_families "$cu_sass" >"$stage/measurements/cuobjdump-sass-families.txt"
extract_sass_families "$nv_sass" >"$stage/measurements/nvdisasm-sass-families.txt"
cu_sass_count=$(wc -l <"$stage/measurements/cuobjdump-sass-families.txt" | tr -d ' ')
nv_sass_count=$(wc -l <"$stage/measurements/nvdisasm-sass-families.txt" | tr -d ' ')
[[ $cu_sass_count =~ ^[1-9][0-9]*$ && $cu_sass_count == "$nv_sass_count" ]] ||
  fail 'SASS instruction counts are absent or disagree'
cu_family_count=$(sort -u "$stage/measurements/cuobjdump-sass-families.txt" | wc -l | tr -d ' ')
nv_family_count=$(sort -u "$stage/measurements/nvdisasm-sass-families.txt" | wc -l | tr -d ' ')
[[ $cu_family_count == 1 && $nv_family_count == 1 ]] ||
  fail 'SASS instruction family is ambiguous'
cu_family=$(sort -u "$stage/measurements/cuobjdump-sass-families.txt")
nv_family=$(sort -u "$stage/measurements/nvdisasm-sass-families.txt")
[[ $cu_family == "$nv_family" ]] || fail 'SASS tools disagree on instruction family'
case $cu_family in
  TCGEN05.MMA) sass_family=TCGEN05_MMA ;;
  HMMA) sass_family=HMMA ;;
  MMA) sass_family=MMA ;;
  *) fail 'SASS instruction family is not allowed' ;;
esac

"$cuobjdump" --dump-resource-usage "$tensor_cubin" \
  >"$stage/measurements/cuobjdump-resources.stdout" \
  2>"$stage/measurements/cuobjdump-resources.stderr"
[[ ! -s $stage/measurements/cuobjdump-resources.stderr ]] || fail 'resource dump emitted stderr'
mkdir -p "$stage/builds/tensor_core/resource"
cp "$tensor_source" "$stage/builds/tensor_core/resource/kernel.cu"
(
  cd "$stage/builds/tensor_core/resource"
  TZ=UTC SOURCE_DATE_EPOCH=0 CUDA_CACHE_DISABLE=1 \
    "$nvcc" --cubin --std=c++14 --generate-code=arch=compute_120,code=sm_120 \
    -O2 --fmad=false --ftz=false --prec-div=true --prec-sqrt=true \
    --maxrregcount=128 -Xptxas=-v kernel.cu -o kernel.cubin \
    >compiler.stdout 2>ptxas-resource.stderr
) || fail 'resource-audit compilation failed'
cmp -s "$tensor_cubin" "$stage/builds/tensor_core/resource/kernel.cubin" ||
  fail 'resource-audit CUBIN differs'
resource=$stage/builds/tensor_core/resource/ptxas-resource.stderr
used_count=$(grep -Ec '^ptxas info[[:space:]]*: Used [0-9]+ registers, [0-9]+ bytes smem(,.*)?$' "$resource")
stack_count=$(grep -Ec '^[[:space:]]*[0-9]+ bytes stack frame, [0-9]+ bytes spill stores, [0-9]+ bytes spill loads$' "$resource")
[[ $used_count == 1 && $stack_count == 1 ]] || fail 'ptxas resource facts are ambiguous'
read -r registers static_shared < <(
  sed -nE 's/^ptxas info[[:space:]]*: Used ([0-9]+) registers, ([0-9]+) bytes smem(,.*)?$/\1 \2/p' "$resource"
)
read -r stack spill_store spill_load < <(
  sed -nE 's/^[[:space:]]*([0-9]+) bytes stack frame, ([0-9]+) bytes spill stores, ([0-9]+) bytes spill loads$/\1 \2 \3/p' "$resource"
)
[[ $registers =~ ^[0-9]+$ && $static_shared =~ ^[0-9]+$ &&
   $stack =~ ^[0-9]+$ && $spill_store =~ ^[0-9]+$ && $spill_load =~ ^[0-9]+$ ]] ||
  fail 'ptxas resource facts are incomplete'
(( registers <= 128 && static_shared == 0 && stack == 0 &&
   spill_store == 0 && spill_load == 0 )) || fail 'resource bound exceeded'
cu_resource=$stage/measurements/cuobjdump-resources.stdout
[[ $(grep -Ec '(^|[[:space:]])REG:[0-9]+([[:space:]]|$)' "$cu_resource") == 1 ]] ||
  fail 'cuobjdump resource facts are ambiguous'
cu_resource_line=$(grep -E '(^|[[:space:]])REG:[0-9]+([[:space:]]|$)' "$cu_resource")
resource_token() {
  local prefix=$1 token value= count=0
  for token in $cu_resource_line; do
    case $token in
      "$prefix":[0-9]*)
        value=${token#*:}
        [[ $value =~ ^[0-9]+$ ]] || fail "cuobjdump $prefix token is malformed"
        count=$((count + 1))
        ;;
    esac
  done
  [[ $count == 1 ]] || fail "cuobjdump $prefix token is absent or duplicated"
  printf '%s\n' "$value"
}
cu_registers=$(resource_token REG)
cu_shared=$(resource_token SHARED)
cu_stack=$(resource_token STACK)
local_bytes=$(resource_token LOCAL)
[[ $cu_registers == "$registers" && $cu_shared == "$static_shared" &&
   $cu_stack == "$stack" ]] || fail 'cuobjdump and ptxas resource facts disagree'
(( local_bytes == 0 )) || fail 'local-memory bound exceeded'

run_sanitizer() {
  local tool=$1 stem=$2
  if [[ $tool == memcheck ]]; then
    "$sanitizer" --tool "$tool" --leak-check full --error-exitcode 99 \
      --log-file "$stage/measurements/$stem.log" \
      "$probe" run "$serial_cubin" "$tensor_cubin" "$cycles" \
      >"$stage/measurements/$stem-runtime.stdout" \
      2>"$stage/measurements/$stem-runtime.stderr"
  else
    "$sanitizer" --tool "$tool" --error-exitcode 99 \
      --log-file "$stage/measurements/$stem.log" \
      "$probe" run "$serial_cubin" "$tensor_cubin" "$cycles" \
      >"$stage/measurements/$stem-runtime.stdout" \
      2>"$stage/measurements/$stem-runtime.stderr"
  fi
  [[ ! -s $stage/measurements/$stem-runtime.stderr ]] || fail "$stem runtime emitted stderr"
  [[ $(grep -Fc 'ERROR SUMMARY:' "$stage/measurements/$stem.log") -eq 1 &&
     $(grep -Fxc '========= ERROR SUMMARY: 0 errors' "$stage/measurements/$stem.log") -eq 1 ]] ||
    fail "$stem summary is not uniquely clean"
}
run_sanitizer memcheck memcheck
run_sanitizer racecheck racecheck
run_sanitizer initcheck initcheck
cmp -s "$stage/measurements/memcheck-runtime.stdout" \
  "$stage/measurements/racecheck-runtime.stdout" || fail 'numeric observations differ'
cmp -s "$stage/measurements/memcheck-runtime.stdout" \
  "$stage/measurements/initcheck-runtime.stdout" || fail 'numeric observations differ'
runtime=$stage/measurements/memcheck-runtime.stdout
[[ $(wc -l <"$runtime" | tr -d ' ') == 1 ]] || fail 'runtime outcome is ambiguous'
grep -F 'outcome=lunatile-tensor-core-sm120-qualification-pass ' "$runtime" >/dev/null ||
  fail 'typed runtime outcome is absent'
grep -F 'resources=context0,stream0,allocation0,module0,function0,device_bytes0,pending0,cleanup0 ' "$runtime" >/dev/null ||
  fail 'cleanup balance is absent'

device_name_sha=$(sed -n 's/.* device_name_sha256=\([^ ]*\).*/\1/p' "$runtime")
device_memory=$(sed -n 's/.* device_total_memory_bytes=\([^ ]*\).*/\1/p' "$runtime")
cpu_serial=$(sed -n 's/.* cpu_vs_serial_max_abs_error=\([^ ]*\).*/\1/p' "$runtime")
cpu_tensor=$(sed -n 's/.* cpu_vs_tensor_core_max_abs_error=\([^ ]*\).*/\1/p' "$runtime")
serial_tensor=$(sed -n 's/.* serial_vs_tensor_core_max_abs_error=\([^ ]*\).*/\1/p' "$runtime")
[[ $device_name_sha =~ ^[0-9a-f]{64}$ && $device_memory =~ ^[1-9][0-9]*$ ]] ||
  fail 'device identity is malformed'
for error_value in "$cpu_serial" "$cpu_tensor" "$serial_tensor"; do
  [[ $error_value =~ ^(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$ ]] ||
    fail 'numeric error observation is not canonical and nonnegative'
done
printf '%s\n' 'schema=lunaflux-lunatile-tensor-core-device.v1' \
  'device_ordinal=0' "device_name_sha256=$device_name_sha" \
  "device_total_memory_bytes=$device_memory" 'device_compute=12.0' 'device_bf16=true' \
  >"$stage/measurements/device-identity.v1"

for stem_tool in \
  "nvcc:$nvcc" "ptxas:$ptxas" "compute-sanitizer:$sanitizer" \
  "cuobjdump:$cuobjdump" "nvdisasm:$nvdisasm"; do
  stem=${stem_tool%%:*}
  tool=${stem_tool#*:}
  "$tool" --version >"$stage/measurements/$stem-version-after.stdout" \
    2>"$stage/measurements/$stem-version-after.stderr"
  [[ ! -s $stage/measurements/$stem-version-after.stderr ]] || fail "$stem identity emitted stderr"
  cmp -s "$stage/measurements/$stem-version-before.stdout" \
    "$stage/measurements/$stem-version-after.stdout" || fail "$stem identity drifted"
done
scripts/inspect-luna-cuda-aot-driver.sh "$nvcc" \
  >"$stage/measurements/driver-after.v1" \
  2>"$stage/measurements/driver-after.stderr"
cmp -s "$stage/measurements/driver-before.v1" "$stage/measurements/driver-after.v1" ||
  fail 'driver identity drifted'
[[ ! -s $stage/measurements/driver-after.stderr &&
   $(sha256_file "$nvcc") == "$nvcc_sha" && $(sha256_file "$ptxas") == "$ptxas_sha" &&
   $(sha256_file "$sanitizer") == "$sanitizer_sha" &&
   $(sha256_file "$cuobjdump") == "$cuobjdump_sha" &&
   $(sha256_file "$nvdisasm") == "$nvdisasm_sha" ]] || fail 'tool identity drifted'

driver_identity_sha=$(sed -n 's/^driver_identity_sha256=//p' "$stage/measurements/driver-before.v1")
driver_record_sha=$(sha256_file "$stage/measurements/driver-before.v1")
compiler_version=$(sed -n 's/^compiler_version=//p' "$stage/measurements/driver-before.v1")
[[ $compiler_version == 13.1.115 ]] || fail 'compiler version is not exact 13.1.115'
nvcc_version_sha=$(sha256_file "$stage/measurements/nvcc-version-before.stdout")
ptxas_version_sha=$(sha256_file "$stage/measurements/ptxas-version-before.stdout")
cuobjdump_version_sha=$(sha256_file "$stage/measurements/cuobjdump-version-before.stdout")
nvdisasm_version_sha=$(sha256_file "$stage/measurements/nvdisasm-version-before.stdout")
sanitizer_version_sha=$(sha256_file "$stage/measurements/compute-sanitizer-version-before.stdout")
cu_sass_sha=$(sha256_file "$cu_sass")
nv_sass_sha=$(sha256_file "$nv_sass")

. "$repo_root/scripts/immutable-evidence-directory.sh"
lunaflux_prepare_evidence_manifest "$stage" || fail 'inner evidence manifest failed'
inner_sha=$lunaflux_evidence_manifest_sha256
printf '%s\n' \
  'schema=lunaflux-lunatile-tensor-core-physical-campaign.v1' \
  'outcome=lunatile-tensor-core-physical-campaign-pass' \
  "program_sha256=$program_sha" "serial_translation_sha256=$serial_translation_sha" \
  "parallel_plan_sha256=$parallel_plan_sha" "tensor_core_candidate_sha256=$tensor_candidate_sha" \
  "serial_source_sha256=$serial_source_sha" "tensor_core_source_sha256=$tensor_source_sha" \
  "serial_cubin_sha256=$serial_cubin_sha" "tensor_core_cubin_sha256=$tensor_cubin_sha" \
  'target=sm_120' 'device_ordinal=0' "device_name_sha256=$device_name_sha" \
  "device_total_memory_bytes=$device_memory" 'device_compute=12.0' 'device_bf16=true' \
  "driver_identity_sha256=$driver_identity_sha" "driver_record_sha256=$driver_record_sha" \
  "compiler_version=$compiler_version" \
  "nvcc_sha256=$nvcc_sha" "ptxas_sha256=$ptxas_sha" \
  "cuobjdump_sha256=$cuobjdump_sha" "nvdisasm_sha256=$nvdisasm_sha" \
  "compute_sanitizer_sha256=$sanitizer_sha" "nvcc_version_sha256=$nvcc_version_sha" \
  "ptxas_version_sha256=$ptxas_version_sha" "cuobjdump_version_sha256=$cuobjdump_version_sha" \
  "nvdisasm_version_sha256=$nvdisasm_version_sha" \
  "compute_sanitizer_version_sha256=$sanitizer_version_sha" \
  "sass_instruction_family=$sass_family" "cuobjdump_sass_sha256=$cu_sass_sha" \
  "cuobjdump_sass_instruction_count=$cu_sass_count" \
  "nvdisasm_sass_sha256=$nv_sass_sha" \
  "nvdisasm_sass_instruction_count=$nv_sass_count" 'sass_instruction_observed=true' \
  "registers_per_thread=$registers" 'register_bound=128' \
  "static_shared_bytes=$static_shared" 'static_shared_bound=0' \
  "stack_bytes=$stack" "local_bytes=$local_bytes" \
  "spill_store_bytes=$spill_store" "spill_load_bytes=$spill_load" \
  'resource_bounds_passed=true' "numeric_case_count=$((cycles * 16384))" \
  'absolute_tolerance=0.001' 'relative_tolerance=0.0001' \
  "cpu_vs_serial_max_abs_error=$cpu_serial" \
  "cpu_vs_tensor_core_max_abs_error=$cpu_tensor" \
  "serial_vs_tensor_core_max_abs_error=$serial_tensor" \
  'cpu_oracle=independent-ordered-f32-v1' 'serial_cuda_oracle=passed' \
  "memcheck_log_sha256=$(sha256_file "$stage/measurements/memcheck.log")" \
  'memcheck_error_summary_count=1' 'memcheck_errors=0' \
  "racecheck_log_sha256=$(sha256_file "$stage/measurements/racecheck.log")" \
  'racecheck_error_summary_count=1' 'racecheck_errors=0' \
  "initcheck_log_sha256=$(sha256_file "$stage/measurements/initcheck.log")" \
  'initcheck_error_summary_count=1' 'initcheck_errors=0' \
  'cleanup_balance=context0,stream0,allocation0,module0,function0,device_bytes0,pending0,cleanup0' \
  'physical_cuda_observed=true' 'qualification_only=true' \
  'manifest_bindable=false' 'promotion_authority=absent' \
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
  "outcome=lunatile-tensor-core-physical-campaign-published inner_seal_sha256=$inner_sha outer_seal_sha256=$outer_sha authority=qualification-only"
