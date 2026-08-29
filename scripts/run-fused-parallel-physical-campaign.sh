#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
export LC_ALL
if [[ $# -lt 6 || $# -gt 7 ]]; then
  printf '%s\n' \
    "usage: $0 ABSOLUTE_NVCC_13_1 ABSOLUTE_APPROVED_POLICY EXPECTED_POLICY_SHA256 ABSOLUTE_COMPUTE_SANITIZER ABSOLUTE_NVIDIA_SMI ABSOLUTE_NEW_OUTPUT [CYCLES]" >&2
  exit 2
fi
nvcc=$1
toolchain_manifest=$2
expected_toolchain_sha256=$3
compute_sanitizer=$4
nvidia_smi=$5
output=$6
cycles=${7:-8}
synthetic_campaign=${LUNAFLUX_SYNTHETIC_PHYSICAL_CAMPAIGN:-0}
fail() {
  printf 'fused physical campaign rejected: %s\n' "$1" >&2
  exit 1
}
require_canonical_file() {
  local path=$1
  local description=$2
  [[ $path == /* ]] || fail "$description path is not absolute"
  [[ $(realpath -- "$path") == "$path" ]] ||
    fail "$description path is not canonical"
  [[ -f $path && ! -L $path ]] ||
    fail "$description is not a regular non-symlink file"
}
require_canonical_file "$nvcc" NVCC
require_canonical_file "$toolchain_manifest" 'toolchain manifest'
require_canonical_file "$compute_sanitizer" compute-sanitizer
require_canonical_file "$nvidia_smi" nvidia-smi
ptxas=$(dirname -- "$nvcc")/ptxas
cuobjdump=$(dirname -- "$nvcc")/cuobjdump
nvdisasm=$(dirname -- "$nvcc")/nvdisasm
require_canonical_file "$ptxas" ptxas
require_canonical_file "$cuobjdump" cuobjdump
require_canonical_file "$nvdisasm" nvdisasm
[[ -x $nvcc ]] || fail 'NVCC is not executable'
[[ -x $compute_sanitizer && -x $nvidia_smi ]] || fail 'physical tool is not executable'
[[ $expected_toolchain_sha256 =~ ^[0-9a-f]{64}$ ]] ||
  fail 'approved physical policy lacks an out-of-band positional SHA-256 pin'
[[ $cycles =~ ^[1-9][0-9]?$ ]] || fail 'CYCLES is not canonical decimal'
(( cycles <= 32 )) || fail 'CYCLES is outside 1..32'
[[ $synthetic_campaign == 0 || $synthetic_campaign == 1 ]] ||
  fail 'synthetic campaign mode must be exactly 0 or 1'
[[ $output == /* ]] || fail 'output path is not absolute'
output_parent=${output%/*}
output_name=${output##*/}
[[ -n $output_parent && -n $output_name && $output_name != . && $output_name != .. ]] ||
  fail 'output path has no exact parent and basename'
[[ $output_name =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] ||
  fail 'output basename is not a bounded canonical evidence name'
[[ -d $output_parent && ! -L $output_parent ]] ||
  fail 'output parent is not a regular directory'
[[ $(realpath -- "$output_parent") == "$output_parent" ]] ||
  fail 'output parent is not canonical'
[[ $output == "$output_parent/$output_name" ]] || fail 'output path is not canonical'
[[ ! -e $output && ! -L $output ]] || fail 'output path already exists'
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"
env -u FAKE_LUNA_CUDA_VERSION scripts/validate-fused-parallel-cuda-probe.sh >/dev/null 2>&1 ||
  fail 'CUDA probe boundary validation failed'
scripts/validate-luna-fused-physical-evidence.sh >/dev/null 2>&1 ||
  fail 'physical evidence boundary validation failed'
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}
. "$repo_root/scripts/fused-physical-campaign-functions.sh"
nvcc_sha256=$(sha256_file "$nvcc")
toolchain_sha256=$(sha256_file "$toolchain_manifest")
[[ $toolchain_sha256 == "$expected_toolchain_sha256" ]] ||
  fail 'approved physical policy digest does not match its pin'
sanitizer_sha256=$(sha256_file "$compute_sanitizer")
ptxas_sha256=$(sha256_file "$ptxas")
cuobjdump_sha256=$(sha256_file "$cuobjdump")
nvdisasm_sha256=$(sha256_file "$nvdisasm")
stage=$(mktemp -d "$output_parent/.${output_name}.partial.XXXXXX")
stage=$(CDPATH= cd -- "$stage" && pwd -P)
cleanup() {
  if [[ -n ${stage:-} && -d $stage ]]; then
    chmod -R u+rwX "$stage" 2>/dev/null || true
    rm -rf -- "$stage"
  fi
}
trap cleanup EXIT HUP INT TERM
artifacts=$stage/artifacts
measurements=$stage/measurements
admission=$stage/admission
mkdir -p "$artifacts" "$measurements" "$admission"
"$nvcc" --version >"$measurements/nvcc-version-before.stdout" \
  2>"$measurements/nvcc-version-before.stderr"
[[ ! -s $measurements/nvcc-version-before.stderr ]] ||
  fail 'NVCC version probe emitted stderr'
grep -F 'release 13.1,' "$measurements/nvcc-version-before.stdout" >/dev/null ||
  fail 'fused qualification requires NVCC release 13.1'
grep -F 'V13.1.115' "$measurements/nvcc-version-before.stdout" >/dev/null ||
  fail 'fused qualification requires exact NVCC 13.1.115'
scripts/inspect-luna-cuda-aot-driver.sh "$nvcc" \
  >"$measurements/driver-identity-before.stdout" \
  2>"$measurements/driver-identity-before.stderr"
[[ ! -s $measurements/driver-identity-before.stderr ]] ||
  fail 'initial CUDA driver identity probe emitted stderr'
policy_observation=$measurements/approved-device-observation.v1
scripts/validate-fused-physical-approved-policy.sh "$toolchain_manifest" \
  "$expected_toolchain_sha256" \
  "$nvcc" "$ptxas" "$cuobjdump" "$nvdisasm" "$compute_sanitizer" \
  "$nvidia_smi" "$measurements/driver-identity-before.stdout" \
  >"$policy_observation" 2>"$measurements/approved-policy.stderr" ||
  fail 'approved tool/device policy validation failed'
[[ ! -s $measurements/approved-policy.stderr ]] || fail 'approved policy emitted stderr'
"$compute_sanitizer" --version \
  >"$measurements/compute-sanitizer-version.stdout" \
  2>"$measurements/compute-sanitizer-version.stderr"
[[ ! -s $measurements/compute-sanitizer-version.stderr ]] ||
  fail 'compute-sanitizer version probe emitted stderr'
for tool_pair in "ptxas:$ptxas" "cuobjdump:$cuobjdump" "nvdisasm:$nvdisasm"; do
  stem=${tool_pair%%:*}
  tool=${tool_pair#*:}
  "$tool" --version >"$measurements/$stem-version.stdout" \
    2>"$measurements/$stem-version.stderr"
  [[ ! -s $measurements/$stem-version.stderr ]] || fail "$stem version emitted stderr"
done
cp "$toolchain_manifest" "$artifacts/toolchain.manifest"
moon build --target native tests/fused_parallel_cuda_probe --deny-warn \
  --warn-list +73 >"$measurements/moon-build.stdout" \
  2>"$measurements/moon-build.stderr"
probe=$repo_root/_build/native/debug/build/tests/fused_parallel_cuda_probe/fused_parallel_cuda_probe.exe
[[ -x $probe && -f $probe && ! -L $probe ]] || fail 'probe executable is missing'
candidate_root=$artifacts/candidates
"$probe" export "$toolchain_sha256" "$candidate_root" \
  >"$measurements/export.stdout" 2>"$measurements/export.stderr"
[[ ! -s $measurements/export.stderr ]] || fail 'candidate export emitted stderr'
  grep -Eq '^outcome=fused-candidate-inputs-exported .* authority=qualification-only$' \
  "$measurements/export.stdout" || fail 'candidate export outcome drifted'
shopt -s nullglob
qkv_sources=("$candidate_root"/candidate-root/sources/*_fused_qkv_positioned_rope_paged_kv_write.cu)
qkv_recipes=("$candidate_root"/candidate-root/recipes/*_fused_qkv_positioned_rope_paged_kv_write.recipe)
residual_sources=("$candidate_root"/candidate-root/sources/*_fused_residual_rmsnorm.cu)
residual_recipes=("$candidate_root"/candidate-root/recipes/*_fused_residual_rmsnorm.recipe)
(( ${#qkv_sources[@]} == 1 && ${#qkv_recipes[@]} == 1 &&
   ${#residual_sources[@]} == 1 && ${#residual_recipes[@]} == 1 )) ||
  fail 'candidate export did not contain exactly two source/recipe pairs'
qkv_source=${qkv_sources[0]}
qkv_recipe=${qkv_recipes[0]}
residual_source=${residual_sources[0]}
residual_recipe=${residual_recipes[0]}
oracle_qkv_source=$candidate_root/candidate-root/sources/oracle_qkv_projection.cu
oracle_qkv_recipe=$candidate_root/candidate-root/recipes/oracle_qkv_projection.recipe
oracle_rope_source=$candidate_root/candidate-root/sources/oracle_positioned_rope.cu
oracle_rope_recipe=$candidate_root/candidate-root/recipes/oracle_positioned_rope.recipe
oracle_kv_source=$candidate_root/candidate-root/sources/oracle_paged_kv_write.cu
oracle_kv_recipe=$candidate_root/candidate-root/recipes/oracle_paged_kv_write.recipe
for oracle_input in "$oracle_qkv_source" "$oracle_qkv_recipe" \
  "$oracle_rope_source" "$oracle_rope_recipe" "$oracle_kv_source" "$oracle_kv_recipe"; do
  [[ -f $oracle_input && ! -L $oracle_input ]] || fail 'standalone oracle export is missing'
done
lunaflux_fused_build_family qkv-first "$qkv_source" "$qkv_recipe"
lunaflux_fused_build_family qkv-second "$qkv_source" "$qkv_recipe"
lunaflux_fused_build_family residual-first "$residual_source" "$residual_recipe"
lunaflux_fused_build_family residual-second "$residual_source" "$residual_recipe"
lunaflux_fused_build_family oracle-qkv-first "$oracle_qkv_source" "$oracle_qkv_recipe"
lunaflux_fused_build_family oracle-qkv-second "$oracle_qkv_source" "$oracle_qkv_recipe"
lunaflux_fused_build_family oracle-rope-first "$oracle_rope_source" "$oracle_rope_recipe"
lunaflux_fused_build_family oracle-rope-second "$oracle_rope_source" "$oracle_rope_recipe"
lunaflux_fused_build_family oracle-kv-first "$oracle_kv_source" "$oracle_kv_recipe"
lunaflux_fused_build_family oracle-kv-second "$oracle_kv_source" "$oracle_kv_recipe"
artifact_sha() {
  sed -n 's/^artifact_sha256=//p' "$artifacts/builds/$1/build.stdout"
}
qkv_sha=$(artifact_sha qkv-first)
qkv_second_sha=$(artifact_sha qkv-second)
residual_sha=$(artifact_sha residual-first)
residual_second_sha=$(artifact_sha residual-second)
oracle_qkv_sha=$(artifact_sha oracle-qkv-first)
oracle_qkv_second_sha=$(artifact_sha oracle-qkv-second)
oracle_rope_sha=$(artifact_sha oracle-rope-first)
oracle_rope_second_sha=$(artifact_sha oracle-rope-second)
oracle_kv_sha=$(artifact_sha oracle-kv-first)
oracle_kv_second_sha=$(artifact_sha oracle-kv-second)
[[ $qkv_sha =~ ^[0-9a-f]{64}$ && $residual_sha =~ ^[0-9a-f]{64}$ ]] ||
  fail 'builder did not publish canonical CUBIN digests'
[[ $qkv_sha == "$qkv_second_sha" && $residual_sha == "$residual_second_sha" ]] ||
  fail 'independent fused CUBIN publications differ'
[[ $oracle_qkv_sha == "$oracle_qkv_second_sha" &&
   $oracle_rope_sha == "$oracle_rope_second_sha" &&
   $oracle_kv_sha == "$oracle_kv_second_sha" ]] ||
  fail 'independent standalone-oracle CUBIN publications differ'
qkv_first=$artifacts/builds/qkv-first/kernels/sha256/$qkv_sha.cubin
qkv_second=$artifacts/builds/qkv-second/kernels/sha256/$qkv_second_sha.cubin
residual_first=$artifacts/builds/residual-first/kernels/sha256/$residual_sha.cubin
residual_second=$artifacts/builds/residual-second/kernels/sha256/$residual_second_sha.cubin
oracle_qkv=$artifacts/builds/oracle-qkv-first/kernels/sha256/$oracle_qkv_sha.cubin
oracle_rope=$artifacts/builds/oracle-rope-first/kernels/sha256/$oracle_rope_sha.cubin
oracle_kv=$artifacts/builds/oracle-kv-first/kernels/sha256/$oracle_kv_sha.cubin
oracle_qkv_receipt=$artifacts/builds/oracle-qkv-first/evidence/luna-cuda-aot-evidence-v1.txt
oracle_rope_receipt=$artifacts/builds/oracle-rope-first/evidence/luna-cuda-aot-evidence-v1.txt
oracle_kv_receipt=$artifacts/builds/oracle-kv-first/evidence/luna-cuda-aot-evidence-v1.txt
qkv_receipt=$artifacts/builds/qkv-first/evidence/luna-cuda-aot-evidence-v1.txt
qkv_second_receipt=$artifacts/builds/qkv-second/evidence/luna-cuda-aot-evidence-v1.txt
residual_receipt=$artifacts/builds/residual-first/evidence/luna-cuda-aot-evidence-v1.txt
residual_second_receipt=$artifacts/builds/residual-second/evidence/luna-cuda-aot-evidence-v1.txt
for required in "$qkv_first" "$qkv_second" "$residual_first" "$residual_second" \
  "$qkv_receipt" "$qkv_second_receipt" "$residual_receipt" "$residual_second_receipt" \
  "$oracle_qkv" "$oracle_rope" "$oracle_kv" "$oracle_qkv_receipt" \
  "$oracle_rope_receipt" "$oracle_kv_receipt"; do
  [[ -f $required && ! -L $required ]] || fail 'compiled artifact is missing or aliased'
done
cmp -s "$qkv_first" "$qkv_second" || fail 'QKV CUBIN bytes differ'
cmp -s "$residual_first" "$residual_second" || fail 'residual CUBIN bytes differ'
cmp -s "$qkv_receipt" "$qkv_second_receipt" || fail 'QKV receipts differ'
cmp -s "$residual_receipt" "$residual_second_receipt" ||
  fail 'residual receipts differ'
for oracle_family in oracle-qkv oracle-rope oracle-kv; do
  cmp -s "$artifacts/builds/$oracle_family-first/evidence/luna-cuda-aot-evidence-v1.txt" \
    "$artifacts/builds/$oracle_family-second/evidence/luna-cuda-aot-evidence-v1.txt" ||
    fail "$oracle_family receipts differ"
  cmp -s "$artifacts/builds/$oracle_family-first/input.source.cu" \
    "$artifacts/builds/$oracle_family-second/input.source.cu" ||
    fail "$oracle_family sources differ"
  cmp -s "$artifacts/builds/$oracle_family-first/input.recipe" \
    "$artifacts/builds/$oracle_family-second/input.recipe" ||
    fail "$oracle_family recipes differ"
done
cmp -s "$artifacts/builds/qkv-first/input.source.cu" \
  "$artifacts/builds/qkv-second/input.source.cu" || fail 'QKV build sources differ'
cmp -s "$artifacts/builds/qkv-first/input.recipe" \
  "$artifacts/builds/qkv-second/input.recipe" || fail 'QKV build recipes differ'
cmp -s "$artifacts/builds/residual-first/input.source.cu" \
  "$artifacts/builds/residual-second/input.source.cu" ||
  fail 'residual build sources differ'
cmp -s "$artifacts/builds/residual-first/input.recipe" \
  "$artifacts/builds/residual-second/input.recipe" ||
  fail 'residual build recipes differ'
lunaflux_fused_audit_resources qkv "$qkv_first" "$qkv_source" 0
lunaflux_fused_audit_resources residual "$residual_first" "$residual_source" 512
. "$repo_root/scripts/immutable-evidence-directory.sh"
lunaflux_prepare_evidence_manifest "$artifacts" || fail 'artifact manifest failed'
artifact_seal_sha256=$lunaflux_evidence_manifest_sha256
lunaflux_seal_evidence_directory "$artifacts" || fail 'artifact seal failed'
runtime_stdout=$measurements/runtime.stdout
runtime_stderr=$measurements/runtime.stderr
memcheck_log=$measurements/compute-sanitizer-memcheck.log
"$compute_sanitizer" --tool memcheck --leak-check full --error-exitcode 99 \
  --log-file "$memcheck_log" "$probe" run "$toolchain_sha256" \
  "$qkv_first" "$qkv_second" "$qkv_receipt" \
  "$residual_first" "$residual_second" "$residual_receipt" \
  "$oracle_qkv" "$oracle_rope" "$oracle_kv" "$cycles" \
  >"$runtime_stdout" 2>"$runtime_stderr"
[[ ! -s $runtime_stderr ]] || fail 'memcheck runtime emitted stderr'
[[ $(grep -Fc 'ERROR SUMMARY:' "$memcheck_log") -eq 1 &&
   $(grep -Fxc '========= ERROR SUMMARY: 0 errors' "$memcheck_log") -eq 1 ]] ||
  fail 'memcheck did not produce exactly one clean error summary'
runtime_field() {
  local key=$1 value
  value=$(awk -v wanted="$key" '
    {
      for (field_index = 1; field_index <= NF; field_index += 1) {
        if ($field_index ~ ("^" wanted "=")) {
          count += 1
          value = substr($field_index, length(wanted) + 2)
        }
      }
    }
    END { if (count == 1) print value }
  ' "$runtime_stdout")
  [[ -n $value && $value != *$'\n'* ]] ||
    fail "runtime $key observation is absent or ambiguous"
  printf '%s\n' "$value"
}
policy_field() {
  local key=$1 value
  value=$(sed -n "s/^${key}=//p" "$policy_observation")
  [[ -n $value && $value != *$'\n'* ]] ||
    fail "approved policy $key observation is absent or ambiguous"
  printf '%s\n' "$value"
}
[[ $(runtime_field device_uuid) == "$(policy_field device_uuid)" &&
   $(runtime_field device_pci) == "$(policy_field device_pci)" ]] ||
  fail 'CUDA context device UUID/PCI does not match approved nvidia-smi policy'
race_stdout=$measurements/racecheck.stdout
race_stderr=$measurements/racecheck.stderr
race_log=$measurements/compute-sanitizer-racecheck.log
"$compute_sanitizer" --tool racecheck --error-exitcode 99 \
  --log-file "$race_log" "$probe" run "$toolchain_sha256" \
  "$qkv_first" "$qkv_second" "$qkv_receipt" \
  "$residual_first" "$residual_second" "$residual_receipt" \
  "$oracle_qkv" "$oracle_rope" "$oracle_kv" "$cycles" \
  >"$race_stdout" 2>"$race_stderr"
[[ ! -s $race_stderr ]] || fail 'racecheck runtime emitted stderr'
[[ $(grep -Fc 'ERROR SUMMARY:' "$race_log") -eq 1 &&
   $(grep -Fxc '========= ERROR SUMMARY: 0 errors' "$race_log") -eq 1 ]] ||
  fail 'racecheck did not produce exactly one clean error summary'
cmp -s "$runtime_stdout" "$race_stdout" ||
  fail 'memcheck and racecheck observations differ'
init_stdout=$measurements/initcheck.stdout
init_stderr=$measurements/initcheck.stderr
init_log=$measurements/compute-sanitizer-initcheck.log
"$compute_sanitizer" --tool initcheck --error-exitcode 99 \
  --log-file "$init_log" "$probe" run "$toolchain_sha256" \
  "$qkv_first" "$qkv_second" "$qkv_receipt" \
  "$residual_first" "$residual_second" "$residual_receipt" \
  "$oracle_qkv" "$oracle_rope" "$oracle_kv" "$cycles" \
  >"$init_stdout" 2>"$init_stderr"
[[ ! -s $init_stderr ]] || fail 'initcheck runtime emitted stderr'
[[ $(grep -Fc 'ERROR SUMMARY:' "$init_log") -eq 1 &&
   $(grep -Fxc '========= ERROR SUMMARY: 0 errors' "$init_log") -eq 1 ]] ||
  fail 'initcheck did not produce exactly one clean error summary'
cmp -s "$runtime_stdout" "$init_stdout" ||
  fail 'memcheck and initcheck observations differ'
"$nvcc" --version >"$measurements/nvcc-version-after.stdout" \
  2>"$measurements/nvcc-version-after.stderr"
[[ ! -s $measurements/nvcc-version-after.stderr ]] ||
  fail 'final NVCC version probe emitted stderr'
cmp -s "$measurements/nvcc-version-before.stdout" \
  "$measurements/nvcc-version-after.stdout" || fail 'NVCC identity output drifted'
scripts/inspect-luna-cuda-aot-driver.sh "$nvcc" \
  >"$measurements/driver-identity-after.stdout" \
  2>"$measurements/driver-identity-after.stderr"
[[ ! -s $measurements/driver-identity-after.stderr ]] ||
  fail 'final CUDA driver identity probe emitted stderr'
cmp -s "$measurements/driver-identity-before.stdout" \
  "$measurements/driver-identity-after.stdout" ||
  fail 'CUDA driver/toolkit identity report drifted'
[[ $(sha256_file "$nvcc") == "$nvcc_sha256" &&
   $(sha256_file "$toolchain_manifest") == "$toolchain_sha256" &&
   $(sha256_file "$compute_sanitizer") == "$sanitizer_sha256" &&
   $(sha256_file "$ptxas") == "$ptxas_sha256" &&
   $(sha256_file "$cuobjdump") == "$cuobjdump_sha256" &&
   $(sha256_file "$nvdisasm") == "$nvdisasm_sha256" &&
   $(sha256_file "$nvidia_smi") == "$(sed -n 's/^nvidia_smi_sha256=//p' "$toolchain_manifest")" ]] ||
  fail 'toolchain identity bytes drifted during campaign'
printf '%s\n' \
  'schema=lunaflux-fused-physical-tool-identities.v1' \
  "nvcc_sha256=$nvcc_sha256" \
  "toolchain_manifest_sha256=$toolchain_sha256" \
  "compute_sanitizer_sha256=$sanitizer_sha256" \
  >"$measurements/tool-identities.txt"
audit_record=$measurements/physical-audit-record.v1
cp "$policy_observation" "$audit_record"
printf '%s\n' \
  "nvcc_sha256=$nvcc_sha256" \
  "compute_sanitizer_sha256=$sanitizer_sha256" \
  "referee_source_sha256=$(sha256_file "$repo_root/tests/fused_parallel_qualification/qkv_referee.mbt")" \
  "probe_executable_sha256=$(sha256_file "$probe")" \
  "standalone_qkv_source_sha256=$(sha256_file "$oracle_qkv_source")" \
  "standalone_qkv_recipe_sha256=$(sha256_file "$oracle_qkv_recipe")" \
  "standalone_qkv_cubin_sha256=$(sha256_file "$oracle_qkv")" \
  "standalone_qkv_receipt_sha256=$(sha256_file "$oracle_qkv_receipt")" \
  "standalone_rope_source_sha256=$(sha256_file "$oracle_rope_source")" \
  "standalone_rope_recipe_sha256=$(sha256_file "$oracle_rope_recipe")" \
  "standalone_rope_cubin_sha256=$(sha256_file "$oracle_rope")" \
  "standalone_rope_receipt_sha256=$(sha256_file "$oracle_rope_receipt")" \
  "standalone_kv_write_source_sha256=$(sha256_file "$oracle_kv_source")" \
  "standalone_kv_write_recipe_sha256=$(sha256_file "$oracle_kv_recipe")" \
  "standalone_kv_write_cubin_sha256=$(sha256_file "$oracle_kv")" \
  "standalone_kv_write_receipt_sha256=$(sha256_file "$oracle_kv_receipt")" \
  "standalone_oracle_output_sha256=$(sha256_file "$runtime_stdout")" \
  'standalone_oracle_executed=true' \
  "ptxas_sha256=$ptxas_sha256" \
  "cuobjdump_sha256=$cuobjdump_sha256" \
  "nvdisasm_sha256=$nvdisasm_sha256" \
  "ptxas_version_sha256=$(sha256_file "$measurements/ptxas-version.stdout")" \
  "cuobjdump_version_sha256=$(sha256_file "$measurements/cuobjdump-version.stdout")" \
  "nvdisasm_version_sha256=$(sha256_file "$measurements/nvdisasm-version.stdout")" \
  "qkv_cuobjdump_sass_sha256=$qkv_cu_sass_sha" \
  "qkv_cuobjdump_sass_instruction_count=$qkv_sass_count" \
  "qkv_nvdisasm_sass_sha256=$qkv_nv_sass_sha" \
  "qkv_nvdisasm_sass_instruction_count=$qkv_sass_count" \
  "qkv_global_load_instruction_count=$qkv_global_load_count" \
  "qkv_global_store_instruction_count=$qkv_global_store_count" \
  "qkv_registers_per_thread=$qkv_registers" \
  "qkv_static_shared_bytes=$qkv_shared" \
  "qkv_local_bytes=$qkv_local" \
  "qkv_stack_bytes=$qkv_stack" \
  "qkv_spill_store_bytes=$qkv_spill_store" \
  "qkv_spill_load_bytes=$qkv_spill_load" \
  "residual_cuobjdump_sass_sha256=$residual_cu_sass_sha" \
  "residual_cuobjdump_sass_instruction_count=$residual_sass_count" \
  "residual_nvdisasm_sass_sha256=$residual_nv_sass_sha" \
  "residual_nvdisasm_sass_instruction_count=$residual_sass_count" \
  "residual_global_load_instruction_count=$residual_global_load_count" \
  "residual_global_store_instruction_count=$residual_global_store_count" \
  "residual_registers_per_thread=$residual_registers" \
  "residual_static_shared_bytes=$residual_shared" \
  "residual_local_bytes=$residual_local" \
  "residual_stack_bytes=$residual_stack" \
  "residual_spill_store_bytes=$residual_spill_store" \
  "residual_spill_load_bytes=$residual_spill_load" >>"$audit_record"
lunaflux_prepare_evidence_manifest "$measurements" ||
  fail 'measurement manifest failed'
evidence_seal_sha256=$lunaflux_evidence_manifest_sha256
lunaflux_seal_evidence_directory "$measurements" ||
  fail 'measurement seal failed'
canonical=$admission/fused-physical-evidence-v1.txt
render_stderr=$admission/render.stderr
"$probe" render "$toolchain_sha256" \
  "$qkv_first" "$qkv_second" "$qkv_receipt" \
  "$residual_first" "$residual_second" "$residual_receipt" \
  "$runtime_stdout" "$runtime_stderr" "$memcheck_log" \
  "$race_stdout" "$race_stderr" "$race_log" \
  "$init_stdout" "$init_stderr" "$init_log" \
  "$audit_record" \
  "$artifact_seal_sha256" "$evidence_seal_sha256" \
  >"$canonical" 2>"$render_stderr"
[[ ! -s $render_stderr ]] || fail 'canonical renderer emitted stderr'
[[ $(wc -l <"$canonical" | tr -d ' ') == 171 ]] ||
  fail 'canonical evidence field-line count drifted'
canonical_sha256=$(sha256_file "$canonical")
printf '%s  %s\n' "$canonical_sha256" fused-physical-evidence-v1.txt \
  >"$admission/fused-physical-evidence-v1.sha256"
if [[ $synthetic_campaign == 1 ]]; then
  printf '%s\n' \
    'outcome=fused-synthetic-evidence-not-admitted authority=absent' \
    >"$admission/admit.not-run"
else
  "$probe" admit "$toolchain_sha256" \
    "$qkv_first" "$qkv_second" "$qkv_receipt" \
    "$residual_first" "$residual_second" "$residual_receipt" \
    "$canonical" "$canonical_sha256" "$expected_toolchain_sha256" \
    >"$admission/admit.stdout" 2>"$admission/admit.stderr"
  [[ ! -s $admission/admit.stderr ]] || fail 'evidence admission emitted stderr'
  grep -Fx \
    "outcome=fused-physical-evidence-admitted evidence_sha256=$canonical_sha256 artifact_seal_sha256=$artifact_seal_sha256 evidence_seal_sha256=$evidence_seal_sha256 authority=observation-only" \
    "$admission/admit.stdout" >/dev/null || fail 'evidence admission outcome drifted'
fi

printf '%s\n' \
  'The canonical record authenticates the FILES.sha256 digests of the sealed' \
  'artifacts/ and measurements/ payloads. It is intentionally outside both' \
  'payloads, avoiding a self-referential digest. The admission package validates' \
  'the canonical bytes and typed identities; it does not inspect filesystem modes.' \
  'The campaign-root FILES.sha256 and read-only modes are a later external seal.' \
  >"$admission/SEAL_SCOPE.txt"
if [[ $synthetic_campaign == 1 ]]; then
  campaign_outcome=fused-synthetic-campaign-pass
  physical_observed=false
  admission_authority=absent
  publication_authority=synthetic-only
else
  campaign_outcome=fused-physical-campaign-pass
  physical_observed=true
  admission_authority=observation-only
  publication_authority=qualification-only
fi
printf '%s\n' \
  "outcome=$campaign_outcome" \
  "evidence_sha256=$canonical_sha256" \
  "artifact_seal_sha256=$artifact_seal_sha256" \
  "evidence_seal_sha256=$evidence_seal_sha256" \
  "physical_cuda_observed=$physical_observed" \
  "admission_authority=$admission_authority" \
  'manifest_bindable=false' \
  'promotion_authority=absent' >"$stage/CAMPAIGN_RESULT.txt"
lunaflux_prepare_evidence_manifest "$stage" || fail 'campaign manifest failed'
campaign_seal_sha256=$lunaflux_evidence_manifest_sha256
lunaflux_seal_evidence_directory "$stage" || fail 'campaign seal failed'
[[ $(sha256_file "$stage/FILES.sha256") == "$campaign_seal_sha256" ]] ||
  fail 'campaign seal digest changed after sealing'
[[ ! -e $output && ! -L $output ]] || fail 'output appeared before publication'
# macOS refuses to rename the read-only staging root. Only that root is made
# writable for the atomic rename; both authenticated inner payloads and every
# file remain read-only. The published root is sealed before success is emitted.
chmod 0755 "$stage"
mv -- "$stage" "$output"
chmod 0555 "$output"
[[ $(sha256_file "$output/FILES.sha256") == "$campaign_seal_sha256" ]] ||
  fail 'published campaign seal digest drifted'
stage=
trap - EXIT HUP INT TERM
printf '%s\n' \
  "outcome=fused-physical-campaign-published evidence_sha256=$canonical_sha256 artifact_seal_sha256=$artifact_seal_sha256 evidence_seal_sha256=$evidence_seal_sha256 campaign_seal_sha256=$campaign_seal_sha256 authority=$publication_authority"
