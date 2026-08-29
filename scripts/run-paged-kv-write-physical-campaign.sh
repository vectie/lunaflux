#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
export LC_ALL
if [[ $# -lt 5 || $# -gt 6 ]]; then
  printf '%s\n' "usage: $0 ABSOLUTE_NVCC_13_1_115 APPROVED_POLICY#sha256=DIGEST ABSOLUTE_COMPUTE_SANITIZER ABSOLUTE_NVIDIA_SMI ABSOLUTE_NEW_OUTPUT [CYCLES]" >&2
  exit 2
fi
nvcc=$1 policy_argument=$2 sanitizer=$3 nvidia_smi=$4 output=$5 cycles=${6:-4}
policy=${policy_argument%#sha256=*}; expected_policy_sha=${policy_argument##*#sha256=}
synthetic_test_only=${LUNAFLUX_SYNTHETIC_TEST_ONLY:-false}
fail() { printf 'paged KV-write physical campaign rejected: %s\n' "$1" >&2; exit 1; }
require_file() { [[ $1 == /* && $(realpath -- "$1") == "$1" && -f $1 && ! -L $1 ]] || fail "$2 is not a canonical file"; }
require_file "$nvcc" nvcc; require_file "$policy" policy; require_file "$sanitizer" sanitizer; require_file "$nvidia_smi" nvidia-smi
ptxas=$(dirname -- "$nvcc")/ptxas; cuobjdump=$(dirname -- "$nvcc")/cuobjdump; nvdisasm=$(dirname -- "$nvcc")/nvdisasm
require_file "$ptxas" ptxas; require_file "$cuobjdump" cuobjdump; require_file "$nvdisasm" nvdisasm
[[ -x $nvcc && -x $sanitizer && -x $nvidia_smi && -x $ptxas && -x $cuobjdump && -x $nvdisasm ]] || fail 'tool is not executable'
[[ $policy != "$policy_argument" && $expected_policy_sha =~ ^[0-9a-f]{64}$ ]] || fail 'policy lacks independent digest pin'
[[ $cycles =~ ^[1-9][0-9]?$ ]] && (( cycles <= 32 )) || fail 'cycles outside 1..32'
[[ $synthetic_test_only == false || $synthetic_test_only == true ]] || fail 'synthetic test marker is invalid'
[[ $output == /* ]] || fail 'output is not absolute'; parent=${output%/*}; name=${output##*/}
[[ $name =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ && -d $parent && ! -L $parent && $(realpath -- "$parent") == "$parent" && ! -e $output && ! -L $output ]] || fail 'output is not canonical and new'
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P); cd "$root"
scripts/validate-luna-positioned-paged-kv-write-aot.sh >/dev/null 2>&1 || fail 'candidate boundary failed'
sha256_file() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
[[ $(sha256_file "$policy") == "$expected_policy_sha" ]] || fail 'policy pin mismatch'
stage=$(mktemp -d "$parent/.${name}.partial.XXXXXX"); stage=$(CDPATH= cd -- "$stage" && pwd -P)
cleanup() { if [[ -n ${stage:-} && -d $stage ]]; then chmod -R u+rwX "$stage" 2>/dev/null || true; rm -rf -- "$stage"; fi; }
trap cleanup EXIT HUP INT TERM
artifacts=$stage/artifacts; measurements=$stage/measurements; admission=$stage/admission
mkdir -p "$artifacts" "$measurements" "$admission" "$artifacts/build-candidate-first/kernels" "$artifacts/build-candidate-first/evidence" "$artifacts/build-candidate-second/kernels" "$artifacts/build-candidate-second/evidence" "$artifacts/oracle-first" "$artifacts/oracle-second"
cp "$policy" "$artifacts/approved-policy.v1"
for pair in "nvcc:$nvcc" "ptxas:$ptxas" "cuobjdump:$cuobjdump" "nvdisasm:$nvdisasm" "compute-sanitizer:$sanitizer"; do
  stem=${pair%%:*}; tool=${pair#*:}; "$tool" --version >"$measurements/$stem-version.stdout" 2>"$measurements/$stem-version.stderr"
  [[ ! -s $measurements/$stem-version.stderr ]] || fail "$stem version emitted stderr"
done
grep -F 'V13.1.115' "$measurements/nvcc-version.stdout" >/dev/null || fail 'NVCC is not exact 13.1.115'
scripts/inspect-luna-cuda-aot-driver.sh "$nvcc" >"$measurements/driver-record.v1" 2>"$measurements/driver.stderr"
[[ ! -s $measurements/driver.stderr ]] || fail 'driver inspection emitted stderr'
scripts/validate-paged-kv-write-approved-policy.sh "$policy_argument" "$nvcc" "$ptxas" "$cuobjdump" "$nvdisasm" "$sanitizer" "$nvidia_smi" "$measurements/driver-record.v1" >"$measurements/policy-observation.v1" 2>"$measurements/policy.stderr" || fail 'approved policy rejected tools/device'
[[ ! -s $measurements/policy.stderr ]] || fail 'approved policy emitted stderr'
moon build tests/paged_kv_write_cuda_probe --target native --deny-warn --warn-list +73 >"$measurements/moon-build.stdout" 2>"$measurements/moon-build.stderr"
probe=$root/_build/native/debug/build/tests/paged_kv_write_cuda_probe/paged_kv_write_cuda_probe.exe
[[ -x $probe && -f $probe && ! -L $probe ]] || fail 'probe executable missing'
source=$artifacts/candidate.cu; recipe=$artifacts/candidate.recipe; oracle_source=$artifacts/serial-oracle.cu
"$probe" export "$expected_policy_sha" "$source" "$recipe" "$oracle_source" >"$measurements/export.stdout" 2>"$measurements/export.stderr"
[[ ! -s $measurements/export.stderr ]] || fail 'export emitted stderr'; export_record=$recipe.export.v1
[[ -f $source && -f $recipe && -f $oracle_source && -f $export_record ]] || fail 'export payload incomplete'
for build in first second; do
  build_root=$artifacts/build-candidate-$build
  scripts/build-luna-cuda-aot.sh "$nvcc" "$policy" "$source" "$recipe" "$build_root/kernels" "$build_root/evidence" >"$build_root/build.stdout" 2>"$build_root/build.stderr"
  [[ ! -s $build_root/build.stderr ]] || fail "candidate $build build emitted stderr"
  (
    cd "$artifacts/oracle-$build"
    TZ=UTC SOURCE_DATE_EPOCH=0 CUDA_CACHE_DISABLE=1 "$nvcc" --cubin --std=c++14 --generate-code=arch=compute_120,code=sm_120 -O3 --fmad=false --ftz=false --prec-div=true --prec-sqrt=true --maxrregcount=128 --Werror all-warnings "$oracle_source" -o oracle.cubin >compiler.stdout 2>compiler.stderr
  ) || fail "serial oracle $build compile failed"
  [[ ! -s $artifacts/oracle-$build/compiler.stderr ]] || fail "serial oracle $build emitted stderr"
done
artifact_sha() { sed -n 's/^artifact_sha256=//p' "$artifacts/build-candidate-$1/build.stdout"; }
candidate_sha=$(artifact_sha first); [[ $candidate_sha =~ ^[0-9a-f]{64}$ && $candidate_sha == "$(artifact_sha second)" ]] || fail 'candidate CUBIN nondeterministic'
candidate_first=$artifacts/build-candidate-first/kernels/sha256/$candidate_sha.cubin; candidate_second=$artifacts/build-candidate-second/kernels/sha256/$candidate_sha.cubin
builder_receipt=$artifacts/build-candidate-first/evidence/luna-cuda-aot-evidence-v1.txt
receipt=$artifacts/positioned-paged-kv-write-compile-receipt.v1
cmp -s "$candidate_first" "$candidate_second" || fail 'candidate bytes differ'; cmp -s "$artifacts/oracle-first/oracle.cubin" "$artifacts/oracle-second/oracle.cubin" || fail 'oracle CUBIN nondeterministic'
oracle_first=$artifacts/oracle-first/oracle.cubin; oracle_second=$artifacts/oracle-second/oracle.cubin
"$probe" receipt "$expected_policy_sha" "$candidate_first" "$builder_receipt" "$receipt" >"$measurements/receipt.stdout" 2>"$measurements/receipt.stderr"
[[ ! -s $measurements/receipt.stderr ]] || fail 'typed receipt materialization emitted stderr'
"$probe" bind "$expected_policy_sha" "$candidate_first" "$candidate_second" "$receipt" >"$measurements/binding.stdout" 2>"$measurements/binding.stderr"
[[ ! -s $measurements/binding.stderr ]] || fail 'binding inspection emitted stderr'
binding_token() { tr ' ' '\n' <"$measurements/binding.stdout" | sed -n "s/^$1=//p"; }
resource=$measurements/resources; mkdir -p "$resource/rebuild"
"$cuobjdump" --dump-sass "$candidate_first" >"$resource/cuobjdump-sass.stdout" 2>"$resource/cuobjdump-sass.stderr"
"$nvdisasm" "$candidate_first" >"$resource/nvdisasm-sass.stdout" 2>"$resource/nvdisasm-sass.stderr"
[[ ! -s $resource/cuobjdump-sass.stderr && ! -s $resource/nvdisasm-sass.stderr ]] || fail 'SASS tools emitted stderr'
sed -nE 's@^[[:space:]]*/\*(0x)?[0-9A-Fa-f]+\*/[[:space:]]+([A-Z][A-Z0-9_.]*).*@\2@p' "$resource/cuobjdump-sass.stdout" >"$resource/cu-opcodes.txt"
sed -nE 's@^[[:space:]]*/\*(0x)?[0-9A-Fa-f]+\*/[[:space:]]+([A-Z][A-Z0-9_.]*).*@\2@p' "$resource/nvdisasm-sass.stdout" >"$resource/nv-opcodes.txt"
cu_count=$(wc -l <"$resource/cu-opcodes.txt" | tr -d ' '); nv_count=$(wc -l <"$resource/nv-opcodes.txt" | tr -d ' ')
[[ $cu_count =~ ^[1-9][0-9]*$ && $cu_count == "$nv_count" ]] || fail 'SASS counts absent or disagree'
for opcodes in "$resource/cu-opcodes.txt" "$resource/nv-opcodes.txt"; do
  grep -Eq '^LDG([.]|$)' "$opcodes" || fail 'SASS global load absent'
  grep -Eq '^STG([.]|$)' "$opcodes" || fail 'SASS global store absent'
done
"$cuobjdump" --dump-resource-usage "$candidate_first" >"$resource/cu-resources.stdout" 2>"$resource/cu-resources.stderr"
[[ ! -s $resource/cu-resources.stderr ]] || fail 'resource dump emitted stderr'
cp "$source" "$resource/rebuild/kernel.cu"
(
  cd "$resource/rebuild"
  TZ=UTC SOURCE_DATE_EPOCH=0 CUDA_CACHE_DISABLE=1 "$nvcc" --cubin --std=c++14 --generate-code=arch=compute_120,code=sm_120 -O3 --fmad=false --ftz=false --prec-div=true --prec-sqrt=true --maxrregcount=128 --Werror all-warnings -Xptxas=-v kernel.cu -o kernel.cubin >compiler.stdout 2>ptxas.stderr
) || fail 'resource rebuild failed'
cmp -s "$candidate_first" "$resource/rebuild/kernel.cubin" || fail 'resource rebuild CUBIN drifted'
[[ $(grep -Ec '^ptxas info[[:space:]]*: Used [0-9]+ registers, [0-9]+ bytes smem(,.*)?$' "$resource/rebuild/ptxas.stderr") == 1 && $(grep -Ec '^[[:space:]]*[0-9]+ bytes stack frame, [0-9]+ bytes spill stores, [0-9]+ bytes spill loads$' "$resource/rebuild/ptxas.stderr") == 1 ]] || fail 'ptxas resources ambiguous'
read -r registers shared < <(sed -nE 's/^ptxas info[[:space:]]*: Used ([0-9]+) registers, ([0-9]+) bytes smem(,.*)?$/\1 \2/p' "$resource/rebuild/ptxas.stderr")
read -r stack spill_store spill_load < <(sed -nE 's/^[[:space:]]*([0-9]+) bytes stack frame, ([0-9]+) bytes spill stores, ([0-9]+) bytes spill loads$/\1 \2 \3/p' "$resource/rebuild/ptxas.stderr")
resource_line=$(grep -E '(^|[[:space:]])REG:[0-9]+([[:space:]]|$)' "$resource/cu-resources.stdout")
token() { local prefix=$1 item value= count=0; for item in $resource_line; do case $item in "$prefix":[0-9]*) value=${item#*:}; count=$((count+1));; esac; done; [[ $count == 1 && $value =~ ^[0-9]+$ ]] || fail "$prefix resource absent/duplicate"; printf '%s\n' "$value"; }
cu_registers=$(token REG); cu_shared=$(token SHARED); cu_stack=$(token STACK); local_bytes=$(token LOCAL)
[[ $registers -gt 0 && $registers -le 128 && $registers == "$cu_registers" && $shared == 0 && $shared == "$cu_shared" && $stack == 0 && $stack == "$cu_stack" && $spill_store == 0 && $spill_load == 0 && $local_bytes == 0 ]] || fail 'resource bounds failed'
. "$root/scripts/immutable-evidence-directory.sh"
lunaflux_prepare_evidence_manifest "$artifacts" || fail 'artifact manifest failed'; artifact_seal=$lunaflux_evidence_manifest_sha256
lunaflux_seal_evidence_directory "$artifacts" || fail 'artifact seal failed'
run_tool() {
  local tool=$1
  local stdout=$measurements/$tool.stdout stderr=$measurements/$tool.stderr log=$measurements/$tool.log
  "$sanitizer" --tool "$tool" --error-exitcode 99 --log-file "$log" "$probe" run "$expected_policy_sha" "$candidate_first" "$candidate_second" "$receipt" "$oracle_first" "$oracle_second" "$cycles" >"$stdout" 2>"$stderr"
  [[ ! -s $stderr && $(grep -Fc 'ERROR SUMMARY:' "$log") == 1 && $(grep -Fxc '========= ERROR SUMMARY: 0 errors' "$log") == 1 ]] || fail "$tool failed"
}
run_tool memcheck; run_tool racecheck; run_tool initcheck
cmp -s "$measurements/memcheck.stdout" "$measurements/racecheck.stdout" && cmp -s "$measurements/memcheck.stdout" "$measurements/initcheck.stdout" || fail 'sanitizer observations differ'
runtime=$measurements/memcheck.stdout
grep -Eq "^outcome=paged-kv-write-sm120-qualification-pass cycles=$cycles cases=6 launches=$((cycles*12)) qualified_token_count=17 key_mutations=1088 value_mutations=1088 observed_dispatch_canary=379457 prefill=pass decode=pass mixed=pass origin=pass page_tail=pass cross_page=pass multirow=pass full_grid=pass guards=pass non_target=pass scalar_oracle=pass serial_cuda_oracle=pass device_uuid=GPU-[^[:space:]]+ device_pci=[^[:space:]]+ device_name_sha256=[0-9a-f]{64} device_total_memory_bytes=[1-9][0-9]* cuda_driver_version=[1-9][0-9]* resources=context0,stream0,allocation0,module0,function0,device_bytes0,pending0,cleanup0 authority=qualification-only$" "$runtime" || fail 'runtime observation drifted'
field() { local file=$1 key=$2 count; count=$(grep -c "^$key=" "$file"); [[ $count == 1 ]] || fail "$key absent/duplicate"; sed -n "s/^$key=//p" "$file"; }
runtime_token() { tr ' ' '\n' <"$runtime" | sed -n "s/^$1=//p"; }
[[ $(runtime_token device_uuid) == "$(field "$policy" device_uuid)" &&
   $(runtime_token device_pci) == "$(field "$policy" device_pci)" &&
   $(runtime_token device_name_sha256) == "$(field "$policy" device_name_sha256)" &&
   $(runtime_token device_total_memory_bytes) == "$(field "$policy" device_total_memory_bytes)" ]] || fail 'CUDA context device differs from approved policy'
scripts/validate-paged-kv-write-approved-policy.sh "$policy_argument" "$nvcc" "$ptxas" "$cuobjdump" "$nvdisasm" "$sanitizer" "$nvidia_smi" "$measurements/driver-record.v1" >"$measurements/policy-recheck.v1" 2>"$measurements/policy-recheck.stderr" || fail 'approved policy drifted during campaign'
[[ ! -s $measurements/policy-recheck.stderr ]] || fail 'approved policy recheck emitted stderr'
cmp -s "$measurements/policy-observation.v1" "$measurements/policy-recheck.v1" || fail 'approved policy observation drifted during campaign'
lunaflux_prepare_evidence_manifest "$measurements" || fail 'measurement manifest failed'; measurement_seal=$lunaflux_evidence_manifest_sha256
lunaflux_seal_evidence_directory "$measurements" || fail 'measurement seal failed'
canonical=$admission/paged-kv-write-physical-evidence.v1
if [[ $synthetic_test_only == true ]]; then canonical_physical=false; canonical_synthetic=true; else canonical_physical=true; canonical_synthetic=false; fi
{
  printf '%s\n' 'schema=lunaflux-paged-kv-write-physical-evidence.v1'; cat "$measurements/policy-observation.v1"
  printf 'cuda_driver_version=%s\n' "$(runtime_token cuda_driver_version)"
  printf 'context_device_uuid=%s\ncontext_device_pci=%s\n' "$(runtime_token device_uuid)" "$(runtime_token device_pci)"
  for key in nvcc ptxas cuobjdump nvdisasm; do printf '%s_sha256=%s\n' "$key" "$(sha256_file "$(dirname -- "$nvcc")/$key")"; done
  printf 'compute_sanitizer_sha256=%s\nnvidia_smi_sha256=%s\ndriver_record_sha256=%s\n' "$(sha256_file "$sanitizer")" "$(sha256_file "$nvidia_smi")" "$(sha256_file "$measurements/driver-record.v1")"
  for key in nvcc ptxas cuobjdump nvdisasm compute-sanitizer; do printf '%s_version_sha256=%s\n' "${key//-/_}" "$(sha256_file "$measurements/$key-version.stdout")"; done
  printf 'driver_identity_sha256=%s\n' "$(sed -n 's/^driver_identity_sha256=//p' "$measurements/driver-record.v1")"
  for key in candidate_sha256 source_sha256 recipe_sha256; do printf '%s=%s\n' "$key" "$(field "$export_record" "$key")"; done
  printf 'compiled_binding_sha256=%s\nmodule_sha256=%s\ncompile_receipt_sha256=%s\n' "$(binding_token binding_sha256)" "$(binding_token module_sha256)" "$(binding_token receipt_sha256)"
  printf 'serial_oracle_source_sha256=%s\n' "$(field "$export_record" serial_oracle_source_sha256)"
  printf 'scalar_referee_source_sha256=%s\nprobe_executable_sha256=%s\n' "$(sha256_file "$root/tests/paged_kv_write_cuda_probe/device_execute.mbt")" "$(sha256_file "$probe")"
  for key in model_content_sha256 model_plan_sha256 attention_operation positioned_qkv_activation query_heads kv_heads head_dimension positioned_qkv_row_width target max_query_rows max_query_tokens max_page_table_entries tokens_per_page total_page_count page_stride_bytes toolchain_sha256 compiler_version function_symbol raw_pointer_abi grid_x grid_y grid_z block_x block_y block_z shared_memory_bytes dispatch_canary_per_token fallback_source_sha256 fallback_recipe_sha256; do printf '%s=%s\n' "$key" "$(field "$export_record" "$key")"; done
  printf '%s\n' 'case_count=6' 'prefill_case_pass=pass' 'decode_case_pass=pass' 'mixed_case_pass=pass' 'origin_case_pass=pass' 'page_tail_case_pass=pass' 'cross_page_case_pass=pass' 'multirow_case_pass=pass' 'full_grid_launch_pass=pass' 'expected_cells_pass=pass' 'byte_oracle_pass=pass' 'scalar_oracle_pass=pass' 'serial_cuda_oracle_pass=pass' 'guards_unchanged=pass' 'non_target_cache_unchanged=pass' 'dispatch_canary_exact=pass' 'deterministic_cubins=pass' 'memcheck_status=pass' 'racecheck_status=pass' 'initcheck_status=pass' 'qualified_token_count=17' 'kv_width=64' 'expected_key_mutations=1088' 'observed_key_mutations=1088' 'expected_value_mutations=1088' 'observed_value_mutations=1088' 'dispatch_canary_cell_count=17' 'expected_dispatch_canary=379457' 'observed_dispatch_canary=379457'
  printf 'cuobjdump_sass_sha256=%s\ncuobjdump_sass_instruction_count=%s\nnvdisasm_sass_sha256=%s\nnvdisasm_sass_instruction_count=%s\n' "$(sha256_file "$resource/cuobjdump-sass.stdout")" "$cu_count" "$(sha256_file "$resource/nvdisasm-sass.stdout")" "$nv_count"
  printf '%s\n' 'sass_global_load_observed=true' 'sass_global_store_observed=true'
  printf 'registers_per_thread=%s\nstatic_shared_bytes=%s\nlocal_bytes=%s\nstack_bytes=%s\nspill_store_bytes=%s\nspill_load_bytes=%s\n' "$registers" "$shared" "$local_bytes" "$stack" "$spill_store" "$spill_load"
  printf '%s\n' 'contexts_live_after=0' 'streams_live_after=0' 'allocations_live_after=0' 'modules_live_after=0' 'functions_live_after=0' 'device_bytes_live_after=0' 'pending_work_after=0' 'stderr_bytes=0'
  printf 'physical_cuda_observed=%s\nsynthetic_test_only=%s\n' "$canonical_physical" "$canonical_synthetic"
  printf 'files_manifest_sha256=%s\nouter_seal_sha256=%s\n' "$artifact_seal" "$measurement_seal"
  printf '%s\n' 'manifest_bindable=false' 'promotion_authority=absent' 'runtime_dispatch_authority=absent'
} >"$canonical"
evidence_sha=$(sha256_file "$canonical")
if [[ $synthetic_test_only == true ]]; then
  printf '%s\n' 'outcome=paged-kv-write-synthetic-transaction-pass' 'physical_cuda_observed=false' 'admission_exercised=false' >"$admission/synthetic-test-only.txt"
  result_outcome=paged-kv-write-synthetic-campaign-pass
  physical_observed=false
else
  "$probe" admit "$expected_policy_sha" "$candidate_first" "$candidate_second" "$receipt" "$canonical" "$evidence_sha" "$expected_policy_sha" "$artifact_seal" "$measurement_seal" >"$admission/admit.stdout" 2>"$admission/admit.stderr"
  [[ ! -s $admission/admit.stderr ]] || fail 'evidence admission emitted stderr'; grep -F 'outcome=paged-kv-write-physical-evidence-admitted' "$admission/admit.stdout" >/dev/null || fail 'admission outcome drifted'
  result_outcome=paged-kv-write-physical-campaign-pass
  physical_observed=true
fi
printf '%s\n' "outcome=$result_outcome" "evidence_sha256=$evidence_sha" "files_manifest_sha256=$artifact_seal" "outer_seal_sha256=$measurement_seal" "physical_cuda_observed=$physical_observed" "context_device_uuid=$(runtime_token device_uuid)" "context_device_pci=$(runtime_token device_pci)" 'manifest_bindable=false' 'promotion_authority=absent' >"$stage/RESULT.txt"
lunaflux_prepare_evidence_manifest "$stage" || fail 'campaign manifest failed'; campaign_seal=$lunaflux_evidence_manifest_sha256
lunaflux_seal_evidence_directory "$stage" || fail 'campaign seal failed'; chmod 0755 "$stage"; mv -- "$stage" "$output"; chmod 0555 "$output"; stage=; trap - EXIT HUP INT TERM
[[ $(sha256_file "$output/FILES.sha256") == "$campaign_seal" ]] || fail 'published campaign seal drifted'
printf 'outcome=paged-kv-write-physical-campaign-published evidence_sha256=%s campaign_seal_sha256=%s authority=qualification-only\n' "$evidence_sha" "$campaign_seal"
