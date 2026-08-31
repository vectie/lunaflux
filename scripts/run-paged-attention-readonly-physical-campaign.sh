#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
export LC_ALL

if [[ $# -lt 6 || $# -gt 7 ]]; then
  printf '%s\n' "usage: $0 ABSOLUTE_NVCC_13_1_115 ABSOLUTE_COMPUTE_SANITIZER ABSOLUTE_CUOBJDUMP ABSOLUTE_NVDISASM ABSOLUTE_APPROVED_POLICY ABSOLUTE_NEW_OUTPUT [CYCLES]" >&2
  exit 2
fi
nvcc=$1 sanitizer=$2 cuobjdump=$3 nvdisasm=$4 approved_policy=$5 output=$6 cycles=${7:-4}
synthetic_test_only=${LUNAFLUX_SYNTHETIC_TEST_ONLY:-false}
[[ $synthetic_test_only == true || $synthetic_test_only == false ]] || { printf '%s\n' 'invalid synthetic-test mode' >&2; exit 2; }
if [[ $synthetic_test_only == true ]]; then
  campaign_outcome=paged-attention-readonly-synthetic-campaign-pass
  physical_observed=false
else
  campaign_outcome=paged-attention-readonly-physical-campaign-pass
  physical_observed=true
fi
fail() { printf 'paged-attention read-only physical campaign rejected: %s\n' "$1" >&2; exit 1; }
require_tool() {
  [[ $1 == /* && -f $1 && ! -L $1 && -x $1 && $(realpath -- "$1") == "$1" ]] ||
    fail "$2 is not an absolute canonical executable file"
}
require_tool "$nvcc" NVCC
require_tool "$sanitizer" compute-sanitizer
require_tool "$cuobjdump" cuobjdump
require_tool "$nvdisasm" nvdisasm
ptxas=$(dirname -- "$nvcc")/ptxas
require_tool "$ptxas" ptxas
[[ $approved_policy == /* && -f $approved_policy && ! -L $approved_policy &&
   $(realpath -- "$approved_policy") == "$approved_policy" ]] || fail 'approved policy is not an absolute canonical regular file'
[[ $cycles =~ ^[1-9][0-9]?$ ]] && (( cycles <= 32 )) || fail 'cycles are outside 1..32'
[[ $output == /* ]] || fail 'output is not absolute'
parent=${output%/*}; name=${output##*/}
[[ $name =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail 'output name is not bounded'
[[ -d $parent && ! -L $parent && $(realpath -- "$parent") == "$parent" ]] || fail 'output parent is not canonical'
[[ $output == "$parent/$name" && ! -e $output && ! -L $output ]] || fail 'output is not canonical and new'

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
for pair in "nvcc:$nvcc" "ptxas:$ptxas" "compute_sanitizer:$sanitizer" \
  "cuobjdump:$cuobjdump" "nvdisasm:$nvdisasm"; do
  stem=${pair%%:*}; tool=${pair#*:}; eval "${stem}_sha=\$(sha256_file \"\$tool\")"
done

stage=$(mktemp -d "$parent/.${name}.partial.XXXXXX")
stage=$(CDPATH= cd -- "$stage" && pwd -P)
cleanup() { if [[ -n ${stage:-} && -d $stage ]]; then chmod -R u+rwX "$stage" 2>/dev/null || true; rm -rf -- "$stage"; fi; }
trap cleanup EXIT HUP INT TERM
mkdir -p "$stage/export" "$stage/builds/first" "$stage/builds/second" \
  "$stage/builds/resource" "$stage/probe" "$stage/measurements" "$stage/policy"
cp "$approved_policy" "$stage/policy/approved.v1"

capture_version() {
  "$1" --version >"$stage/measurements/$2-version-before.stdout" \
    2>"$stage/measurements/$2-version-before.stderr"
  [[ ! -s $stage/measurements/$2-version-before.stderr ]] || fail "$2 version emitted stderr"
}
capture_version "$nvcc" nvcc; capture_version "$ptxas" ptxas
capture_version "$sanitizer" compute-sanitizer; capture_version "$cuobjdump" cuobjdump
capture_version "$nvdisasm" nvdisasm
grep -F 'release 13.1,' "$stage/measurements/nvcc-version-before.stdout" >/dev/null || fail 'NVCC release mismatch'
scripts/inspect-luna-cuda-aot-driver.sh "$nvcc" >"$stage/measurements/driver-before.v1" \
  2>"$stage/measurements/driver-before.stderr"
[[ ! -s $stage/measurements/driver-before.stderr ]] || fail 'driver inspection emitted stderr'
compiler_version=$(sed -n 's/^compiler_version=//p' "$stage/measurements/driver-before.v1")
driver_identity=$(sed -n 's/^driver_identity_sha256=//p' "$stage/measurements/driver-before.v1")
[[ $compiler_version == 13.1.115 && $driver_identity =~ ^[0-9a-f]{64}$ ]] || fail 'compiler/driver identity mismatch'
driver_record_sha=$(sha256_file "$stage/measurements/driver-before.v1")
policy_field() { local count; count=$(grep -c "^$1=" "$stage/policy/approved.v1"); [[ $count == 1 ]] || fail "approved policy $1 absent/duplicated"; sed -n "s/^$1=//p" "$stage/policy/approved.v1"; }
[[ $(wc -l <"$stage/policy/approved.v1" | tr -d ' ') == 16 ]] || fail 'approved policy is not canonical 16-line v1'
expected_policy_keys='schema
target
compiler_version
nvcc_sha256
ptxas_sha256
cuobjdump_sha256
nvdisasm_sha256
compute_sanitizer_sha256
driver_identity_sha256
driver_record_sha256
device_uuid
device_pci_bus_id
device_name_hex
device_total_memory_bytes
device_compute
policy_authority'
actual_policy_keys=$(sed -n 's/=.*//p' "$stage/policy/approved.v1")
[[ $actual_policy_keys == "$expected_policy_keys" ]] || fail 'approved policy field order is not canonical'
[[ $(policy_field schema) == lunaflux-paged-attention-readonly-approved-physical-policy.v1 &&
   $(policy_field target) == sm_120 && $(policy_field compiler_version) == 13.1.115 &&
   $(policy_field nvcc_sha256) == "$nvcc_sha" && $(policy_field ptxas_sha256) == "$ptxas_sha" &&
   $(policy_field cuobjdump_sha256) == "$cuobjdump_sha" && $(policy_field nvdisasm_sha256) == "$nvdisasm_sha" &&
   $(policy_field compute_sanitizer_sha256) == "$compute_sanitizer_sha" &&
   $(policy_field driver_identity_sha256) == "$driver_identity" &&
   $(policy_field driver_record_sha256) == "$driver_record_sha" &&
   $(policy_field device_compute) == 12.0 &&
   $(policy_field policy_authority) == deployment-approved ]] || fail 'tool/driver does not match approved policy'
approved_policy_sha=$(sha256_file "$stage/policy/approved.v1")

moon build tests/paged_attention_readonly_source_export --target native --deny-warn --warn-list +73 \
  >"$stage/measurements/moon-build.stdout" 2>"$stage/measurements/moon-build.stderr"
exporter=$root/_build/native/debug/build/tests/paged_attention_readonly_source_export/paged_attention_readonly_source_export.exe
[[ -x $exporter && -f $exporter && ! -L $exporter ]] || fail 'exporter is missing'
"$exporter" export "$stage/export" "$driver_identity" "$compiler_version" \
  >"$stage/measurements/export.stdout" 2>"$stage/measurements/export.stderr"
[[ ! -s $stage/measurements/export.stderr ]] || fail 'exporter emitted stderr'
record=$stage/export/export.v1; source=$stage/export/readonly.cu
read_field() { local count; count=$(grep -c "^$2=" "$1"); [[ $count == 1 ]] || fail "$2 is absent or duplicated"; sed -n "s/^$2=//p" "$1"; }
candidate_sha=$(read_field "$record" candidate_sha256); source_sha=$(read_field "$record" source_sha256)
recipe_sha=$(read_field "$record" recipe_sha256); fallback_source_sha=$(read_field "$record" fallback_source_sha256)
fallback_recipe_sha=$(read_field "$record" fallback_recipe_sha256); symbol=$(read_field "$record" function_symbol)
abi=$(read_field "$record" raw_pointer_abi); input_row_width=$(read_field "$record" input_row_width)
input_activation=$(read_field "$record" input_activation_ref)
dispatch_canary=$(read_field "$record" dispatch_canary_per_token); canary_cells=$(read_field "$record" dispatch_canary_cell_count)
profile=$(read_field "$record" profile); operation_id=$(read_field "$record" operation_id)
for digest in "$candidate_sha" "$source_sha" "$recipe_sha" "$fallback_source_sha" "$fallback_recipe_sha"; do
  [[ $digest =~ ^[0-9a-f]{64}$ ]] || fail 'export digest malformed'
done
[[ $input_row_width == 8 && $dispatch_canary == 65624 && $canary_cells == 260 && $profile == 4,260,132 ]] ||
  fail 'row/profile/canary identity drifted'
[[ $(read_field "$stage/export/readonly.recipe" toolchain_sha256) == "$driver_identity" ]] || fail 'candidate toolchain is not authenticated driver identity'
[[ $(sha256_file "$source") == "$source_sha" && $(sha256_file "$stage/export/readonly.recipe") == "$recipe_sha" &&
   $(sha256_file "$stage/export/fallback.cu") == "$fallback_source_sha" &&
   $(sha256_file "$stage/export/fallback.recipe") == "$fallback_recipe_sha" ]] || fail 'export digest mismatch'
grep -F "void $symbol(" "$source" >/dev/null || fail 'symbol absent'
grep -F 'const __nv_bfloat16 *key_cache' "$source" >/dev/null
grep -F 'dispatch_canary[token] = 65624U' "$source" >/dev/null
if grep -E '(key_cache|value_cache)\[[^]]+\][[:space:]]*=' "$source"; then fail 'source contains KV mutation'; fi

compile_cubin() {
  local dir=$1 resource=${2:-false} extra=
  cp "$source" "$dir/kernel.cu"
  $resource && extra=-Xptxas=-v
  (cd "$dir"; TZ=UTC SOURCE_DATE_EPOCH=0 CUDA_CACHE_DISABLE=1 "$nvcc" --cubin --std=c++17 \
    --generate-code=arch=compute_120,code=sm_120 -O3 --fmad=false --ftz=false \
    --prec-div=true --prec-sqrt=true --maxrregcount=128 --Werror all-warnings \
    $extra kernel.cu -o kernel.cubin >compiler.stdout 2>compiler.stderr) || fail 'CUBIN compilation failed'
  [[ -s $dir/kernel.cubin ]] || fail 'CUBIN is empty'
}
compile_cubin "$stage/builds/first"; compile_cubin "$stage/builds/second"
cmp -s "$stage/builds/first/kernel.cubin" "$stage/builds/second/kernel.cubin" || fail 'independent CUBIN bytes differ'
cubin=$stage/builds/first/kernel.cubin; module_sha=$(sha256_file "$cubin")
printf '%s\n' 'schema=lunaflux-paged-attention-readonly-aot-evidence.v1' \
  "candidate_sha256=$candidate_sha" "source_sha256=$source_sha" "recipe_sha256=$recipe_sha" \
  "fallback_source_sha256=$fallback_source_sha" "fallback_recipe_sha256=$fallback_recipe_sha" \
  "toolchain_sha256=$driver_identity" "driver_identity_sha256=$driver_identity" 'target=sm_120' \
  "function_symbol=$symbol" "raw_pointer_abi=$abi" "input_row_width=$input_row_width" \
  "input_activation_ref=$input_activation" \
  "dispatch_canary_per_token=$dispatch_canary" "dispatch_canary_cell_count=$canary_cells" \
  "artifact_sha256=$module_sha" "first_build_sha256=$module_sha" "second_build_sha256=$module_sha" \
  'deterministic=1' 'kv_cache_mutation=none' >"$stage/measurements/compile-receipt.v1"
receipt_sha=$(sha256_file "$stage/measurements/compile-receipt.v1")
"$exporter" bind "$stage/builds/first/kernel.cubin" "$stage/builds/second/kernel.cubin" \
  "$stage/measurements/compile-receipt.v1" "$receipt_sha" "$driver_identity" "$compiler_version" \
  >"$stage/measurements/bind.stdout" 2>"$stage/measurements/bind.stderr"
[[ ! -s $stage/measurements/bind.stderr ]] || fail 'compiled binder emitted stderr'
binding_sha=$(sed -n 's/.* binding_sha256=\([^ ]*\).*/\1/p' "$stage/measurements/bind.stdout")
[[ $binding_sha =~ ^[0-9a-f]{64}$ ]] || fail 'compiled binding missing'

"$cuobjdump" --dump-sass "$cubin" >"$stage/measurements/cuobjdump-sass.stdout" 2>"$stage/measurements/cuobjdump-sass.stderr"
"$nvdisasm" "$cubin" >"$stage/measurements/nvdisasm-sass.stdout" 2>"$stage/measurements/nvdisasm-sass.stderr"
[[ ! -s $stage/measurements/cuobjdump-sass.stderr && ! -s $stage/measurements/nvdisasm-sass.stderr ]] || fail 'SASS tool emitted stderr'
extract_instructions() { sed -nE 's@^[[:space:]]*/\*(0x)?[0-9A-Fa-f]+\*/[[:space:]]+([A-Z][A-Z0-9_.]*)([[:space:]].*)?$@\2@p' "$1"; }
extract_instructions "$stage/measurements/cuobjdump-sass.stdout" >"$stage/measurements/cuobjdump-instructions.txt"
extract_instructions "$stage/measurements/nvdisasm-sass.stdout" >"$stage/measurements/nvdisasm-instructions.txt"
cu_count=$(wc -l <"$stage/measurements/cuobjdump-instructions.txt" | tr -d ' ')
nv_count=$(wc -l <"$stage/measurements/nvdisasm-instructions.txt" | tr -d ' ')
[[ $cu_count =~ ^[1-9][0-9]*$ && $nv_count =~ ^[1-9][0-9]*$ ]] || fail 'SASS instruction evidence absent'
for file in "$stage/measurements/cuobjdump-instructions.txt" "$stage/measurements/nvdisasm-instructions.txt"; do
  grep -E '^LDG([.]|$)' "$file" >/dev/null || fail 'SASS global load absent'
  grep -E '^STG([.]|$)' "$file" >/dev/null || fail 'SASS global store absent'
done

compile_cubin "$stage/builds/resource" true
cmp -s "$cubin" "$stage/builds/resource/kernel.cubin" || fail 'resource CUBIN differs'
resource=$stage/builds/resource/compiler.stderr
[[ $(grep -Ec '^ptxas info[[:space:]]*: Used [0-9]+ registers, [0-9]+ bytes smem(,.*)?$' "$resource") == 1 &&
   $(grep -Ec '^[[:space:]]*[0-9]+ bytes stack frame, [0-9]+ bytes spill stores, [0-9]+ bytes spill loads$' "$resource") == 1 ]] || fail 'ptxas resources ambiguous'
read -r registers shared < <(sed -nE 's/^ptxas info[[:space:]]*: Used ([0-9]+) registers, ([0-9]+) bytes smem(,.*)?$/\1 \2/p' "$resource")
read -r stack spill_store spill_load < <(sed -nE 's/^[[:space:]]*([0-9]+) bytes stack frame, ([0-9]+) bytes spill stores, ([0-9]+) bytes spill loads$/\1 \2 \3/p' "$resource")
[[ $registers =~ ^[0-9]+$ && $shared =~ ^[0-9]+$ && $stack =~ ^[0-9]+$ && $spill_store =~ ^[0-9]+$ && $spill_load =~ ^[0-9]+$ ]] || fail 'resource facts malformed'
(( registers <= 128 && shared == 0 && stack == 0 && spill_store == 0 && spill_load == 0 )) || fail 'resource bound exceeded'
"$cuobjdump" --dump-resource-usage "$cubin" >"$stage/measurements/cuobjdump-resources.stdout" 2>"$stage/measurements/cuobjdump-resources.stderr"
[[ ! -s $stage/measurements/cuobjdump-resources.stderr ]] || fail 'resource dump emitted stderr'
line_count=$(grep -Ec '(^|[[:space:]])REG:[0-9]+([[:space:]]|$)' "$stage/measurements/cuobjdump-resources.stdout")
[[ $line_count == 1 ]] || fail 'cuobjdump resource line ambiguous'
resource_line=$(grep -E '(^|[[:space:]])REG:[0-9]+([[:space:]]|$)' "$stage/measurements/cuobjdump-resources.stdout")
resource_token() { local token value= count=0; for token in $resource_line; do case $token in "$1":[0-9]*) value=${token#*:}; [[ $value =~ ^[0-9]+$ ]] || fail "$1 resource malformed"; count=$((count+1));; esac; done; [[ $count == 1 ]] || fail "$1 resource absent/duplicated"; printf '%s\n' "$value"; }
cu_registers=$(resource_token REG); cu_shared=$(resource_token SHARED); cu_stack=$(resource_token STACK); local_bytes=$(resource_token LOCAL)
[[ $cu_registers == "$registers" && $cu_shared == "$shared" && $cu_stack == "$stack" && $local_bytes == 0 ]] || fail 'resource tools disagree'

cp tests/paged_attention_readonly_cuda_probe/probe.cu "$stage/probe/probe.cu"
(cd "$stage/probe"; "$nvcc" -std=c++17 -O2 --generate-code=arch=compute_120,code=sm_120 \
  probe.cu -lcuda -o physical_probe >compiler.stdout 2>compiler.stderr) || fail 'physical probe build failed'
probe=$stage/probe/physical_probe
[[ -x $probe && -f $probe && ! -L $probe ]] || fail 'physical probe missing'
run_sanitizer() {
  local tool=$1
  if [[ $tool == memcheck ]]; then
    "$sanitizer" --tool "$tool" --leak-check full --error-exitcode 99 \
      --log-file "$stage/measurements/$tool.log" "$probe" "$cubin" "$symbol" "$cycles" \
      >"$stage/measurements/$tool-runtime.stdout" 2>"$stage/measurements/$tool-runtime.stderr"
  else
    "$sanitizer" --tool "$tool" --error-exitcode 99 \
      --log-file "$stage/measurements/$tool.log" "$probe" "$cubin" "$symbol" "$cycles" \
      >"$stage/measurements/$tool-runtime.stdout" 2>"$stage/measurements/$tool-runtime.stderr"
  fi
  [[ ! -s $stage/measurements/$tool-runtime.stderr ]] || fail "$tool runtime emitted stderr"
  [[ $(grep -Fc 'ERROR SUMMARY:' "$stage/measurements/$tool.log") == 1 &&
     $(grep -Fxc '========= ERROR SUMMARY: 0 errors' "$stage/measurements/$tool.log") == 1 ]] || fail "$tool summary not uniquely clean"
}
run_sanitizer memcheck; run_sanitizer racecheck; run_sanitizer initcheck
cmp -s "$stage/measurements/memcheck-runtime.stdout" "$stage/measurements/racecheck-runtime.stdout" || fail 'runtime observations differ'
cmp -s "$stage/measurements/memcheck-runtime.stdout" "$stage/measurements/initcheck-runtime.stdout" || fail 'runtime observations differ'
runtime=$stage/measurements/memcheck-runtime.stdout
[[ $(wc -l <"$runtime" | tr -d ' ') == 1 ]] || fail 'runtime result ambiguous'
for exact in 'outcome=paged-attention-readonly-sm120-qualification-pass ' \
  'case_families=origin,page-tail,cross-page,multirow,long-context ' 'output_dtype=bf16 ' \
  'scheduler_modes=prefill-only,decode-only,mixed-prefill-decode ' \
  'cache_snapshot_unchanged=true ' 'input_guards_unchanged=true ' 'output_guards_unchanged=true ' \
  'dispatch_symbol_resolved=true ' "dispatch_canary_per_token=$dispatch_canary " \
  'dispatch_grid_x_max=32 ' 'dispatch_grid_y=2 ' 'dispatch_block_x=32 ' \
  "dispatch_canary_cell_count=$canary_cells " 'dispatch_canary_exact=true ' \
  'dispatch_canary_tail_zero=true ' 'dispatch_canary_sum_checked=true ' \
  "input_row_width=$input_row_width " 'target=sm_120 ' \
  'resources=module0,allocation0,device_bytes0,pending0,cleanup0 '; do
  grep -F "$exact" "$runtime" >/dev/null || fail "runtime claim absent: $exact"
done
runtime_value() { local value; value=$(sed -n "s/.* $1=\([^ ]*\).*/\1/p" "$runtime"); [[ -n $value ]] || fail "$1 runtime value absent"; printf '%s\n' "$value"; }
numeric_cases=$(runtime_value numeric_case_count); cpu_error=$(runtime_value cpu_vs_candidate_max_abs_error)
serial_error=$(runtime_value serial_vs_candidate_max_abs_error); cache_bytes=$(runtime_value cache_snapshot_bytes)
device_uuid=$(runtime_value device_uuid); device_pci_bus_id=$(runtime_value device_pci_bus_id)
device_name_hex=$(runtime_value device_name_hex); device_memory=$(runtime_value device_total_memory_bytes)
[[ $numeric_cases =~ ^[1-9][0-9]*$ && $cache_bytes == 81920 &&
   $device_uuid =~ ^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ &&
   $device_pci_bus_id =~ ^[0-9a-f]{8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$ &&
   $device_name_hex =~ ^[0-9a-f]+$ && $device_memory =~ ^[1-9][0-9]*$ ]] || fail 'runtime facts malformed'
[[ $(policy_field device_uuid) == "$device_uuid" &&
   $(policy_field device_pci_bus_id) == "$device_pci_bus_id" &&
   $(policy_field device_name_hex) == "$device_name_hex" &&
   $(policy_field device_total_memory_bytes) == "$device_memory" ]] || fail 'physical device does not match approved policy'
for value in "$cpu_error" "$serial_error"; do [[ $value =~ ^(0|[1-9][0-9]*)([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || fail 'numeric error malformed'; done
printf '%s\n' 'schema=lunaflux-paged-attention-readonly-device.v1' 'device_ordinal=0' \
  "device_uuid=$device_uuid" "device_pci_bus_id=$device_pci_bus_id" \
  "device_name_hex=$device_name_hex" "device_total_memory_bytes=$device_memory" \
  'device_compute=12.0' 'device_bf16=true' >"$stage/measurements/device-identity.v1"

for pair in "nvcc:$nvcc" "ptxas:$ptxas" "compute-sanitizer:$sanitizer" "cuobjdump:$cuobjdump" "nvdisasm:$nvdisasm"; do
  stem=${pair%%:*}; tool=${pair#*:}; "$tool" --version >"$stage/measurements/$stem-version-after.stdout" 2>"$stage/measurements/$stem-version-after.stderr"
  [[ ! -s $stage/measurements/$stem-version-after.stderr ]] || fail "$stem after-version emitted stderr"
  cmp -s "$stage/measurements/$stem-version-before.stdout" "$stage/measurements/$stem-version-after.stdout" || fail "$stem version drifted"
done
scripts/inspect-luna-cuda-aot-driver.sh "$nvcc" >"$stage/measurements/driver-after.v1" 2>"$stage/measurements/driver-after.stderr"
[[ ! -s $stage/measurements/driver-after.stderr ]] && cmp -s "$stage/measurements/driver-before.v1" "$stage/measurements/driver-after.v1" || fail 'driver drifted'
[[ $(sha256_file "$nvcc") == "$nvcc_sha" && $(sha256_file "$ptxas") == "$ptxas_sha" &&
   $(sha256_file "$sanitizer") == "$compute_sanitizer_sha" && $(sha256_file "$cuobjdump") == "$cuobjdump_sha" &&
   $(sha256_file "$nvdisasm") == "$nvdisasm_sha" ]] || fail 'tool bytes drifted'

device_record_sha=$(sha256_file "$stage/measurements/device-identity.v1")
nvcc_version_sha=$(sha256_file "$stage/measurements/nvcc-version-before.stdout")
ptxas_version_sha=$(sha256_file "$stage/measurements/ptxas-version-before.stdout")
cuobjdump_version_sha=$(sha256_file "$stage/measurements/cuobjdump-version-before.stdout")
nvdisasm_version_sha=$(sha256_file "$stage/measurements/nvdisasm-version-before.stdout")
sanitizer_version_sha=$(sha256_file "$stage/measurements/compute-sanitizer-version-before.stdout")
. "$root/scripts/immutable-evidence-directory.sh"
lunaflux_prepare_evidence_manifest "$stage" || fail 'inner manifest failed'
inner_sha=$lunaflux_evidence_manifest_sha256
printf '%s\n' 'schema=lunaflux-paged-attention-readonly-physical-campaign.v1' \
  "outcome=$campaign_outcome" "candidate_sha256=$candidate_sha" \
  "approved_policy_sha256=$approved_policy_sha" \
  "source_sha256=$source_sha" "recipe_sha256=$recipe_sha" "fallback_source_sha256=$fallback_source_sha" \
  "fallback_recipe_sha256=$fallback_recipe_sha" "compiled_binding_sha256=$binding_sha" "cubin_sha256=$module_sha" \
  "compile_receipt_sha256=$receipt_sha" "operation_id=$operation_id" 'target=sm_120' 'profile=4,260,132' \
  'kv_layout=layer_major_split_key_value_v1' 'kv_layout_stable_version=1' 'tokens_per_page=2' \
  'page_stride_bytes=256' 'physical_page_capacity=160' "input_row_width=$input_row_width" \
  "input_activation_ref=$input_activation" \
  "function_symbol=$symbol" "raw_pointer_abi=$abi" 'selection_precondition=standalone-positioned-rope-paged-kvwrite-complete-v1' \
  "dispatch_canary_per_token=$dispatch_canary" "dispatch_canary_cell_count=$canary_cells" \
  'dispatch_canary_publication=exactly-once-after-output' 'dispatch_canary_exact=true' \
  'dispatch_canary_tail_zero=true' 'dispatch_canary_sum_checked=true' 'kv_cache_mutation=none' \
  "driver_identity_sha256=$driver_identity" "driver_record_sha256=$driver_record_sha" \
  "device_record_sha256=$device_record_sha" "compiler_version=$compiler_version" \
  "device_uuid=$device_uuid" "device_pci_bus_id=$device_pci_bus_id" \
  "device_name_hex=$device_name_hex" "device_total_memory_bytes=$device_memory" \
  "nvcc_sha256=$nvcc_sha" "ptxas_sha256=$ptxas_sha" "cuobjdump_sha256=$cuobjdump_sha" \
  "nvdisasm_sha256=$nvdisasm_sha" "compute_sanitizer_sha256=$compute_sanitizer_sha" \
  "nvcc_version_sha256=$nvcc_version_sha" "ptxas_version_sha256=$ptxas_version_sha" \
  "cuobjdump_version_sha256=$cuobjdump_version_sha" "nvdisasm_version_sha256=$nvdisasm_version_sha" \
  "compute_sanitizer_version_sha256=$sanitizer_version_sha" \
  "probe_source_sha256=$(sha256_file "$stage/probe/probe.cu")" \
  "probe_binary_sha256=$(sha256_file "$stage/probe/physical_probe")" \
  "cuobjdump_sass_sha256=$(sha256_file "$stage/measurements/cuobjdump-sass.stdout")" \
  "cuobjdump_sass_instruction_count=$cu_count" "nvdisasm_sass_sha256=$(sha256_file "$stage/measurements/nvdisasm-sass.stdout")" \
  "nvdisasm_sass_instruction_count=$nv_count" 'sass_global_load_observed=true' 'sass_global_store_observed=true' \
  "registers_per_thread=$registers" 'register_bound=128' "static_shared_bytes=$shared" 'static_shared_bound=0' \
  "stack_bytes=$stack" "local_bytes=$local_bytes" "spill_store_bytes=$spill_store" "spill_load_bytes=$spill_load" \
  'resource_bounds_passed=true' 'case_families=origin,page-tail,cross-page,multirow,long-context' \
  'scheduler_modes=prefill-only,decode-only,mixed-prefill-decode' \
  'dispatch_grid_x_max=32' 'dispatch_grid_y=2' 'dispatch_block_x=32' \
  "numeric_case_count=$numeric_cases" 'output_dtype=bf16' 'absolute_tolerance=0.0078125' 'relative_tolerance=0.01' \
  "cpu_vs_candidate_max_abs_error=$cpu_error" "serial_vs_candidate_max_abs_error=$serial_error" \
  'cpu_oracle=independent-ordered-f32-v1' 'serial_cuda_oracle=independent-ordered-f32-kernel-v1' \
  "cache_snapshot_bytes=$cache_bytes" 'cache_snapshot_unchanged=true' 'input_guards_unchanged=true' \
  'output_guards_unchanged=true' 'dispatch_symbol_resolved=true' \
  "memcheck_log_sha256=$(sha256_file "$stage/measurements/memcheck.log")" 'memcheck_error_summary_count=1' 'memcheck_errors=0' \
  "racecheck_log_sha256=$(sha256_file "$stage/measurements/racecheck.log")" 'racecheck_error_summary_count=1' 'racecheck_errors=0' \
  "initcheck_log_sha256=$(sha256_file "$stage/measurements/initcheck.log")" 'initcheck_error_summary_count=1' 'initcheck_errors=0' \
  'cleanup_balance=module0,allocation0,device_bytes0,pending0,cleanup0' "physical_cuda_observed=$physical_observed" \
  "synthetic_test_only=$synthetic_test_only" \
  'qualification_only=true' 'manifest_bindable=false' 'promotion_authority=absent' \
  "evidence_files_manifest_sha256=$inner_sha" >"$stage/RESULT.txt"
{ printf '%s  FILES.sha256\n' "$(sha256_file "$stage/FILES.sha256")"; printf '%s  RESULT.txt\n' "$(sha256_file "$stage/RESULT.txt")"; } >"$stage/OUTER_SEAL.sha256"
outer_sha=$(sha256_file "$stage/OUTER_SEAL.sha256")
lunaflux_seal_evidence_directory "$stage" || fail 'evidence seal failed'
chmod 0755 "$stage"; mv -- "$stage" "$output"; chmod 0555 "$output"; stage=
trap - EXIT HUP INT TERM
printf 'outcome=paged-attention-readonly-physical-campaign-published inner_seal_sha256=%s outer_seal_sha256=%s authority=qualification-only\n' "$inner_sha" "$outer_sha"
