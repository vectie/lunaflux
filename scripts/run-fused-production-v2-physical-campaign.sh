#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
TZ=UTC
export LC_ALL TZ
umask 077

fail() { printf 'fused production V2 campaign rejected: %s\n' "$1" >&2; exit 1; }
if [[ $# -lt 6 || $# -gt 7 ]]; then
  printf '%s\n' \
    "usage: $0 ABSOLUTE_NVCC_13_1 ABSOLUTE_APPROVED_POLICY EXPECTED_POLICY_SHA256 ABSOLUTE_COMPUTE_SANITIZER ABSOLUTE_NVIDIA_SMI ABSOLUTE_NEW_OUTPUT [CYCLES]" >&2
  exit 2
fi
nvcc=$1
policy=$2
expected_policy_sha=$3
sanitizer=$4
nvidia_smi=$5
output=$6
cycles=${7:-8}

require_file() {
  local path=$1 name=$2 executable=${3:-false}
  [[ $path == /* && -f $path && ! -L $path && $(realpath -- "$path") == "$path" ]] ||
    fail "$name is not an absolute canonical regular file"
  if [[ $executable == true ]]; then [[ -x $path ]] || fail "$name is not executable"; fi
}
require_file "$nvcc" nvcc true
require_file "$policy" approved-policy
require_file "$sanitizer" compute-sanitizer true
require_file "$nvidia_smi" nvidia-smi true
ptxas=$(dirname -- "$nvcc")/ptxas
cuobjdump=$(dirname -- "$nvcc")/cuobjdump
nvdisasm=$(dirname -- "$nvcc")/nvdisasm
require_file "$ptxas" ptxas true
require_file "$cuobjdump" cuobjdump true
require_file "$nvdisasm" nvdisasm true
[[ $expected_policy_sha =~ ^[0-9a-f]{64}$ ]] || fail 'policy pin is not SHA-256'
[[ $cycles =~ ^[1-9][0-9]?$ ]] && (( cycles <= 32 )) || fail 'cycles are outside 1..32'
[[ $output == /* ]] || fail 'output is not absolute'
parent=${output%/*}
name=${output##*/}
[[ $name =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail 'output name is not bounded'
[[ -d $parent && ! -L $parent && $(realpath -- "$parent") == "$parent" ]] || fail 'output parent is not canonical'
[[ $output == "$parent/$name" && ! -e $output && ! -L $output ]] || fail 'output is not canonical and new'

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"
scripts/validate-fused-production-v2-physical-campaign.sh --static-only >/dev/null 2>&1 ||
  fail 'local production V2 boundary validation failed'
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
policy_sha=$(sha256_file "$policy")
[[ $policy_sha == "$expected_policy_sha" ]] || fail 'approved policy digest does not match its pin'
nvcc_sha=$(sha256_file "$nvcc")
sanitizer_sha=$(sha256_file "$sanitizer")
ptxas_sha=$(sha256_file "$ptxas")
cuobjdump_sha=$(sha256_file "$cuobjdump")
nvdisasm_sha=$(sha256_file "$nvdisasm")
nvidia_smi_sha=$(sha256_file "$nvidia_smi")

stage=$(mktemp -d "$parent/.${name}.partial.XXXXXX")
stage=$(CDPATH= cd -- "$stage" && pwd -P)
cleanup() { if [[ -n ${stage:-} && -d $stage ]]; then chmod -R u+rwX "$stage" 2>/dev/null || true; rm -rf -- "$stage"; fi; }
trap cleanup EXIT HUP INT TERM
artifacts=$stage/artifacts
measurements=$stage/measurements
mkdir -p "$artifacts/builds" "$measurements"
cp "$policy" "$artifacts/approved-policy.v1"

"$nvcc" --version >"$measurements/nvcc-before.stdout" 2>"$measurements/nvcc-before.stderr"
[[ ! -s $measurements/nvcc-before.stderr ]] || fail 'nvcc version emitted stderr'
grep -F 'release 13.1,' "$measurements/nvcc-before.stdout" >/dev/null || fail 'NVCC release is not 13.1'
grep -F 'V13.1.115' "$measurements/nvcc-before.stdout" >/dev/null || fail 'NVCC is not exact 13.1.115'
scripts/inspect-luna-cuda-aot-driver.sh "$nvcc" >"$measurements/driver-before.v1" 2>"$measurements/driver-before.stderr"
[[ ! -s $measurements/driver-before.stderr ]] || fail 'driver inspection emitted stderr'
scripts/validate-fused-physical-approved-policy.sh "$policy" "$expected_policy_sha" \
  "$nvcc" "$ptxas" "$cuobjdump" "$nvdisasm" "$sanitizer" "$nvidia_smi" \
  "$measurements/driver-before.v1" >"$measurements/device-policy.v1" \
  2>"$measurements/device-policy.stderr" || fail 'approved device/tool policy rejected'
[[ ! -s $measurements/device-policy.stderr ]] || fail 'policy validation emitted stderr'
toolchain_sha=$(sha256_file "$policy")

moon build tests/fused_parallel_cuda_probe --target native --deny-warn --warn-list +73 \
  >"$measurements/moon-build.stdout" 2>"$measurements/moon-build.stderr" || fail 'MoonBit probe build failed'
probe=$root/_build/native/debug/build/tests/fused_parallel_cuda_probe/fused_parallel_cuda_probe.exe
[[ -x $probe && -f $probe && ! -L $probe ]] || fail 'probe executable is absent'
export_root=$artifacts/production-inputs
"$probe" export-production "$toolchain_sha" "$export_root" \
  >"$measurements/export.stdout" 2>"$measurements/export.stderr"
[[ ! -s $measurements/export.stderr ]] || fail 'production input export emitted stderr'
grep -Fx 'outcome=fused-production-v2-inputs-exported authority=qualification-only manifest_bindable=0 promotion_authority=0' \
  "$measurements/export.stdout" >/dev/null || fail 'production export outcome drifted'
identities=$export_root/IDENTITIES.v1
[[ -f $identities && ! -L $identities ]] || fail 'production identities are absent'
for exact in 'target=sm_120' 'executor_identity=ordered-kernel-executor.v1' \
  'bootstrap_join=production-compiled-candidate-source-and-symbol.v1' \
  'graph_policy=capture-required' 'qkv_abi=QkvPositionedRopePagedKvWriteProductionAbiV2' \
  'readonly_abi=RotatedQueryPagedKvReadOnlyProductionRawPointerV2' \
  'residual_abi=ResidualRmsNormProductionAbiV2' 'manifest_bindable=0' 'promotion_authority=0'; do
  grep -Fx "$exact" "$identities" >/dev/null || fail "identity drift: $exact"
done

. "$root/scripts/fused-physical-campaign-functions.sh"
build_pair() {
  local family=$1 stem=$2
  lunaflux_fused_build_family "$family-first" "$export_root/sources/$stem.cu" "$export_root/recipes/$stem.recipe"
  lunaflux_fused_build_family "$family-second" "$export_root/sources/$stem.cu" "$export_root/recipes/$stem.recipe"
  local first_sha second_sha
  first_sha=$(sed -n 's/^artifact_sha256=//p' "$artifacts/builds/$family-first/build.stdout")
  second_sha=$(sed -n 's/^artifact_sha256=//p' "$artifacts/builds/$family-second/build.stdout")
  [[ $first_sha =~ ^[0-9a-f]{64}$ && $first_sha == "$second_sha" ]] || fail "$family CUBIN is nondeterministic"
  cmp -s "$artifacts/builds/$family-first/evidence/luna-cuda-aot-evidence-v1.txt" \
    "$artifacts/builds/$family-second/evidence/luna-cuda-aot-evidence-v1.txt" || fail "$family receipts differ"
  printf -v "${family}_sha" '%s' "$first_sha"
}
build_pair qkv production_qkv_v2
build_pair readonly production_readonly_attention_v2
build_pair residual production_residual_rmsnorm_v2
build_pair oracle_qkv oracle_qkv
build_pair oracle_rope oracle_rope
build_pair oracle_attention oracle_attention
artifact() { printf '%s/builds/%s-first/kernels/sha256/%s.cubin\n' "$artifacts" "$1" "$2"; }
receipt() { printf '%s/builds/%s-first/evidence/luna-cuda-aot-evidence-v1.txt\n' "$artifacts" "$1"; }
qkv_first=$(artifact qkv "$qkv_sha"); qkv_second=$(artifact qkv "$qkv_sha")
readonly_first=$(artifact readonly "$readonly_sha"); readonly_second=$(artifact readonly "$readonly_sha")
residual_first=$(artifact residual "$residual_sha"); residual_second=$(artifact residual "$residual_sha")
oracle_qkv=$(artifact oracle_qkv "$oracle_qkv_sha")
oracle_rope=$(artifact oracle_rope "$oracle_rope_sha")
oracle_attention=$(artifact oracle_attention "$oracle_attention_sha")
qkv_receipt=$(receipt qkv); residual_receipt=$(receipt residual)
qkv_second=$artifacts/builds/qkv-second/kernels/sha256/$qkv_sha.cubin
readonly_second=$artifacts/builds/readonly-second/kernels/sha256/$readonly_sha.cubin
residual_second=$artifacts/builds/residual-second/kernels/sha256/$residual_sha.cubin

lunaflux_fused_audit_resources qkvv2 "$qkv_first" "$export_root/sources/production_qkv_v2.cu" 0
lunaflux_fused_audit_resources readonlyv2 "$readonly_first" "$export_root/sources/production_readonly_attention_v2.cu" 0
lunaflux_fused_audit_resources residualv2 "$residual_first" "$export_root/sources/production_residual_rmsnorm_v2.cu" 512

run_sanitizer() {
  local tool=$1 leak=()
  if [[ $tool == memcheck ]]; then leak=(--leak-check full); fi
  "$sanitizer" --tool "$tool" "${leak[@]}" --error-exitcode 99 \
    --log-file "$measurements/$tool.log" "$probe" run-production "$toolchain_sha" \
    "$qkv_first" "$qkv_second" "$qkv_receipt" \
    "$readonly_first" "$readonly_second" \
    "$residual_first" "$residual_second" "$residual_receipt" \
    "$oracle_qkv" "$oracle_rope" "$oracle_attention" "$cycles" \
    >"$measurements/$tool-runtime.stdout" 2>"$measurements/$tool-runtime.stderr"
  [[ ! -s $measurements/$tool-runtime.stderr ]] || fail "$tool runtime emitted stderr"
  [[ $(grep -Fxc '========= ERROR SUMMARY: 0 errors' "$measurements/$tool.log") == 1 ]] || fail "$tool did not report exactly zero errors"
}
run_sanitizer memcheck
run_sanitizer racecheck
run_sanitizer initcheck
cmp -s "$measurements/memcheck-runtime.stdout" "$measurements/racecheck-runtime.stdout" || fail 'sanitizer observations differ'
cmp -s "$measurements/memcheck-runtime.stdout" "$measurements/initcheck-runtime.stdout" || fail 'sanitizer observations differ'
runtime=$measurements/memcheck-runtime.stdout
[[ $(wc -l <"$runtime" | tr -d ' ') == 1 ]] || fail 'runtime outcome is ambiguous'
grep -Eq "^outcome=fused-production-v2-sm120-qualification-pass cycles=$cycles shapes=4 launches=$((cycles * 24)) maximum_absolute_error_ppb=[0-9]+ maximum_relative_error_ppb=[0-9]+ qkv_abi=QkvPositionedRopePagedKvWriteProductionAbiV2 readonly_abi=RotatedQueryPagedKvReadOnlyProductionRawPointerV2 residual_abi=ResidualRmsNormProductionAbiV2 executor=ordered-kernel-executor graph_policy=capture-required graph_mode=captured oracle=standalone-cuda device_uuid=[^ ]+ device_pci=[^ ]+ device_name_sha256=[0-9a-f]{64} device_total_memory_bytes=[1-9][0-9]* cuda_driver_version=[1-9][0-9]* resources=context0,stream0,event0,graph0,graph_exec0,allocation0,module0,function0,executor0,device_bytes0,pending0,cleanup0 authority=qualification-only manifest_bindable=0 promotion_authority=0$" "$runtime" || fail 'runtime result schema drifted'
runtime_field() { awk -v key="$1" '{ for (i=1;i<=NF;i++) if ($i ~ ("^" key "=")) { sub("^[^=]*=", "", $i); print $i } }' "$runtime"; }
policy_field() { sed -n "s/^$1=//p" "$measurements/device-policy.v1"; }
[[ $(runtime_field device_uuid) == "$(policy_field device_uuid)" && $(runtime_field device_pci) == "$(policy_field device_pci)" ]] || fail 'runtime device differs from approved device'

"$nvcc" --version >"$measurements/nvcc-after.stdout" 2>"$measurements/nvcc-after.stderr"
scripts/inspect-luna-cuda-aot-driver.sh "$nvcc" >"$measurements/driver-after.v1" 2>"$measurements/driver-after.stderr"
cmp -s "$measurements/nvcc-before.stdout" "$measurements/nvcc-after.stdout" || fail 'NVCC identity drifted'
cmp -s "$measurements/driver-before.v1" "$measurements/driver-after.v1" || fail 'driver identity drifted'
[[ ! -s $measurements/nvcc-after.stderr && ! -s $measurements/driver-after.stderr && \
   $(sha256_file "$nvcc") == "$nvcc_sha" && $(sha256_file "$sanitizer") == "$sanitizer_sha" && \
   $(sha256_file "$ptxas") == "$ptxas_sha" && $(sha256_file "$cuobjdump") == "$cuobjdump_sha" && \
   $(sha256_file "$nvdisasm") == "$nvdisasm_sha" && $(sha256_file "$nvidia_smi") == "$nvidia_smi_sha" ]] || fail 'tool identity drifted'

identity_field() {
  local value
  value=$(sed -n "s/^$1=//p" "$identities")
  [[ -n $value && $(grep -c "^$1=" "$identities") == 1 ]] ||
    fail "production identity is absent or duplicated: $1"
  printf '%s\n' "$value"
}
spawn_join=$artifacts/SPAWN_JOIN.v1
printf '%s\n' 'schema=lunaflux-fused-production-v2-spawn-join.v1' \
  "qkv_artifact_relative=artifacts/builds/qkv-first/kernels/sha256/$qkv_sha.cubin" \
  "qkv_module_sha256=$qkv_sha" \
  "qkv_source_sha256=$(identity_field production_qkv_source_sha256)" \
  "qkv_symbol=$(identity_field production_qkv_symbol)" \
  "readonly_artifact_relative=artifacts/builds/readonly-first/kernels/sha256/$readonly_sha.cubin" \
  "readonly_module_sha256=$readonly_sha" \
  "readonly_source_sha256=$(identity_field production_readonly_source_sha256)" \
  "readonly_symbol=$(identity_field production_readonly_symbol)" \
  "residual_artifact_relative=artifacts/builds/residual-first/kernels/sha256/$residual_sha.cubin" \
  "residual_module_sha256=$residual_sha" \
  "residual_source_sha256=$(identity_field production_residual_source_sha256)" \
  "residual_symbol=$(identity_field production_residual_symbol)" \
  "identities_sha256=$(sha256_file "$identities")" \
  'aggregate_runtime=externally-prepared-required' \
  'normal_spawn_admission=not-claimed' \
  'qualification_only=true' 'promotion_authority=absent' >"$spawn_join"
spawn_join_sha=$(sha256_file "$spawn_join")

. "$root/scripts/immutable-evidence-directory.sh"
lunaflux_prepare_evidence_manifest "$stage" || fail 'FILES manifest failed'
inner_sha=$lunaflux_evidence_manifest_sha256
printf '%s\n' 'schema=lunaflux-fused-production-v2-physical-campaign.v1' \
  'outcome=fused-production-v2-physical-campaign-pass' 'target=sm_120' \
  'compiler_version=13.1.115' "approved_policy_sha256=$policy_sha" \
  "nvcc_sha256=$nvcc_sha" "compute_sanitizer_sha256=$sanitizer_sha" \
  "identities_sha256=$(sha256_file "$identities")" \
  "spawn_join_sha256=$spawn_join_sha" "cycles=$cycles" \
  "launches=$((cycles * 24))" 'production_abis=qkv-v2,readonly-attention-v2,residual-rmsnorm-v2' \
  'executor=ordered-kernel-executor' 'graph_policy=capture-required' 'graph_mode=captured' \
  'oracle=standalone-cuda' 'memcheck_errors=0' 'racecheck_errors=0' 'initcheck_errors=0' \
  'physical_cuda_observed=true' 'qualification_only=true' 'manifest_bindable=false' \
  'promotion_authority=absent' "evidence_files_manifest_sha256=$inner_sha" >"$stage/RESULT.txt"
{
  printf '%s  FILES.sha256\n' "$(sha256_file "$stage/FILES.sha256")"
  printf '%s  RESULT.txt\n' "$(sha256_file "$stage/RESULT.txt")"
} >"$stage/OUTER_SEAL.sha256"
outer_sha=$(sha256_file "$stage/OUTER_SEAL.sha256")
lunaflux_seal_evidence_directory "$stage" || fail 'evidence seal failed'
[[ ! -e $output && ! -L $output ]] || fail 'output appeared before publication'
chmod 0755 "$stage"
mv -- "$stage" "$output"
chmod 0555 "$output"
stage=
trap - EXIT HUP INT TERM
scripts/verify-fused-production-v2-physical-campaign.sh \
  "$output" "$outer_sha" "$expected_policy_sha" >/dev/null || fail 'published evidence verification failed'
printf 'outcome=fused-production-v2-physical-campaign-published inner_seal_sha256=%s outer_seal_sha256=%s authority=qualification-only\n' "$inner_sha" "$outer_sha"
