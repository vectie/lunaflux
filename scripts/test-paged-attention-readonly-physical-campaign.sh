#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
export LC_ALL

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
runner=$root/scripts/run-paged-attention-readonly-physical-campaign.sh
scratch=$(mktemp -d /private/tmp/lunaflux-readonly-physical-test.XXXXXX)
cleanup() {
  [[ ${KEEP_EVIDENCE:-false} == true ]] && { printf 'kept=%s\n' "$scratch"; return; }
  chmod -R u+rwX "$scratch" 2>/dev/null || true
  rm -rf -- "$scratch"
}
trap cleanup EXIT HUP INT TERM
tools=$scratch/tools
mkdir -p "$tools"

cat >"$tools/nvcc" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --version ]]; then
  printf '%s\n' 'Cuda compilation tools, release 13.1, V13.1.115'
  exit 0
fi
output= cubin=false resource=false
while [[ $# -gt 0 ]]; do
  case $1 in
    -o) output=$2; shift 2 ;;
    --cubin) cubin=true; shift ;;
    -Xptxas=-v) resource=true; shift ;;
    *) shift ;;
  esac
done
[[ -n $output ]]
if $cubin; then
  cp kernel.cu "$output"
  if [[ ${FAKE_CASE:-success} == nondeterministic ]]; then
    counter=$(dirname -- "$0")/nvcc.counter; value=0; [[ ! -f $counter ]] || value=$(<"$counter")
    value=$((value+1)); printf '%s' "$value" >"$counter"; printf '\nnonce=%s\n' "$value" >>"$output"
  fi
  if $resource; then
    registers=64; [[ ${FAKE_CASE:-success} != resource_overflow ]] || registers=129
    printf '    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n' >&2
    printf 'ptxas info    : Used %s registers, 0 bytes smem, 0 bytes cmem[0]\n' "$registers" >&2
  fi
else
  cp "$FAKE_PROBE" "$output"; chmod +x "$output"
fi
EOF

cat >"$tools/ptxas" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' 'ptxas release 13.1, V13.1.115'
EOF

cat >"$tools/cuobjdump" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
case ${1:-} in
  --version)
    if [[ ${FAKE_CASE:-success} == tool_drift ]]; then
      counter=$(dirname -- "$0")/cuobjdump.counter; value=0; [[ ! -f $counter ]] || value=$(<"$counter")
      value=$((value+1)); printf '%s' "$value" >"$counter"; printf 'cuobjdump 13.1.%s\n' "$value"
    else printf '%s\n' 'cuobjdump 13.1.115'; fi ;;
  --dump-sass)
    if [[ ${FAKE_CASE:-success} == comment_only_sass ]]; then
      printf '%s\n' '// LDG STG mentioned but no instruction'
    else
      printf '%s\n' '/*0000*/ LDG.E R2, [R4];' '/*0010*/ FFMA R6, R2, R3, R6;' '/*0020*/ STG.E [R8], R6;'
    fi ;;
  --dump-resource-usage)
    registers=64; [[ ${FAKE_CASE:-success} != resource_overflow ]] || registers=129
    printf 'Function readonly: REG:%s STACK:0 SHARED:0 LOCAL:0\n' "$registers" ;;
  *) exit 2 ;;
esac
EOF

cat >"$tools/nvdisasm" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --version ]]; then printf '%s\n' 'nvdisasm 13.1.115'
elif [[ ${FAKE_CASE:-success} == comment_only_sass ]]; then printf '%s\n' '// LDG STG'
else printf '%s\n' '/*0000*/ LDG.E R2, [R4];' '/*0010*/ FFMA R6, R2, R3, R6;' '/*0020*/ STG.E [R8], R6;'; fi
EOF

cat >"$tools/compute-sanitizer" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --version ]]; then printf '%s\n' 'NVIDIA Compute Sanitizer 2026.1.0'; exit 0; fi
log=
while [[ $# -gt 0 ]]; do if [[ $1 == --log-file ]]; then log=$2; shift 2; else shift; fi; done
[[ -n $log ]]
if [[ ${FAKE_CASE:-success} == dirty_sanitizer ]]; then printf '%s\n' '========= ERROR SUMMARY: 1 error' >"$log"
else printf '%s\n' '========= ERROR SUMMARY: 0 errors' >"$log"; fi
runtime='outcome=paged-attention-readonly-sm120-qualification-pass cycles=2 case_families=origin,page-tail,cross-page,multirow,long-context scheduler_modes=prefill-only,decode-only,mixed-prefill-decode numeric_case_count=72 candidate_launches=10 serial_launches=10 cpu_vs_candidate_max_abs_error=0 serial_vs_candidate_max_abs_error=0 absolute_tolerance=0.0078125 relative_tolerance=0.01 output_dtype=bf16 cpu_oracle=independent-ordered-f32-v1 serial_cuda_oracle=independent-ordered-f32-kernel-v1 cache_snapshot_bytes=81920 cache_snapshot_unchanged=true input_guards_unchanged=true output_guards_unchanged=true dispatch_symbol_resolved=true dispatch_grid_x=260 dispatch_canary_per_token=65624 dispatch_canary_cell_count=260 dispatch_canary_exact=true dispatch_canary_tail_zero=true dispatch_canary_sum_checked=true input_row_width=8 target=sm_120 device_ordinal=0 device_uuid=GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6 device_pci_bus_id=00000000:17:00.0 device_name_hex=66616b652d736d313230 device_total_memory_bytes=25769803776 resources=module0,allocation0,device_bytes0,pending0,cleanup0 manifest_bindable=false promotion_authority=absent'
case ${FAKE_CASE:-success} in
  partial) printf '%s\n' 'outcome=paged-attention-readonly-sm120-qualification-pass' ;;
  cache_mutation) printf '%s\n' "${runtime/cache_snapshot_unchanged=true/cache_snapshot_unchanged=false}" ;;
  input_guard) printf '%s\n' "${runtime/input_guards_unchanged=true/input_guards_unchanged=false}" ;;
  output_guard) printf '%s\n' "${runtime/output_guards_unchanged=true/output_guards_unchanged=false}" ;;
  canary_zero) printf '%s\n' "${runtime/dispatch_canary_per_token=65624/dispatch_canary_per_token=0}" ;;
  canary_wrong) printf '%s\n' "${runtime/dispatch_canary_exact=true/dispatch_canary_exact=false}" ;;
  canary_tail) printf '%s\n' "${runtime/dispatch_canary_tail_zero=true/dispatch_canary_tail_zero=false}" ;;
  canary_overflow) printf '%s\n' "${runtime/dispatch_canary_sum_checked=true/dispatch_canary_sum_checked=false}" ;;
  uuid_substitution) printf '%s\n' "${runtime/GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6/GPU-11111111-1111-1111-1111-111111111111}" ;;
  pci_substitution) printf '%s\n' "${runtime/00000000:17:00.0/00000000:18:00.0}" ;;
  *) printf '%s\n' "$runtime" ;;
esac
EOF

cat >"$tools/fake-probe" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "$tools"/*
export FAKE_PROBE=$tools/fake-probe
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
"$root/scripts/inspect-luna-cuda-aot-driver.sh" "$tools/nvcc" >"$scratch/driver.v1"
driver_identity=$(sed -n 's/^driver_identity_sha256=//p' "$scratch/driver.v1")
cat >"$scratch/approved-policy.v1" <<EOF
schema=lunaflux-paged-attention-readonly-approved-physical-policy.v1
target=sm_120
compiler_version=13.1.115
nvcc_sha256=$(sha256_file "$tools/nvcc")
ptxas_sha256=$(sha256_file "$tools/ptxas")
cuobjdump_sha256=$(sha256_file "$tools/cuobjdump")
nvdisasm_sha256=$(sha256_file "$tools/nvdisasm")
compute_sanitizer_sha256=$(sha256_file "$tools/compute-sanitizer")
driver_identity_sha256=$driver_identity
driver_record_sha256=$(sha256_file "$scratch/driver.v1")
device_uuid=GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6
device_pci_bus_id=00000000:17:00.0
device_name_hex=66616b652d736d313230
device_total_memory_bytes=25769803776
device_compute=12.0
policy_authority=deployment-approved
EOF
approved_policy=$(realpath "$scratch/approved-policy.v1")

success=$scratch/success
if ! LUNAFLUX_SYNTHETIC_TEST_ONLY=true FAKE_CASE=success "$runner" "$tools/nvcc" "$tools/compute-sanitizer" \
  "$tools/cuobjdump" "$tools/nvdisasm" "$approved_policy" "$success" 2 >"$scratch/success.stdout" 2>"$scratch/success.stderr"; then
  cat "$scratch/success.stderr" >&2
  printf '%s\n' 'successful fake physical campaign was rejected' >&2
  exit 1
fi
[[ ! -s $scratch/success.stderr ]]
grep -Fx 'schema=lunaflux-paged-attention-readonly-physical-campaign.v1' "$success/RESULT.txt" >/dev/null
grep -Fx 'outcome=paged-attention-readonly-synthetic-campaign-pass' "$success/RESULT.txt" >/dev/null
grep -Fx 'physical_cuda_observed=false' "$success/RESULT.txt" >/dev/null
grep -Fx 'synthetic_test_only=true' "$success/RESULT.txt" >/dev/null
grep -Fx 'manifest_bindable=false' "$success/RESULT.txt" >/dev/null
[[ $(wc -l <"$success/OUTER_SEAL.sha256" | tr -d ' ') == 2 ]]
[[ -z $(find "$success" -type f -perm -u+w -print -quit) ]]
if LUNAFLUX_SYNTHETIC_TEST_ONLY=true FAKE_CASE=success "$runner" "$tools/nvcc" "$tools/compute-sanitizer" "$tools/cuobjdump" \
  "$tools/nvdisasm" "$approved_policy" "$success" 2 >/dev/null 2>&1; then echo 'overwrite accepted' >&2; exit 1; fi

reordered_policy=$scratch/reordered-policy.v1
{
  sed -n '2p' "$approved_policy"
  sed -n '1p' "$approved_policy"
  sed -n '3,16p' "$approved_policy"
} >"$reordered_policy"
reordered_policy=$(realpath "$reordered_policy")
reordered_output=$scratch/reordered-policy-output
if LUNAFLUX_SYNTHETIC_TEST_ONLY=true FAKE_CASE=success "$runner" "$tools/nvcc" "$tools/compute-sanitizer" \
  "$tools/cuobjdump" "$tools/nvdisasm" "$reordered_policy" "$reordered_output" 2 >/dev/null 2>&1; then
  printf '%s\n' 'reordered approved policy was accepted' >&2
  exit 1
fi
[[ ! -e $reordered_output ]] || { printf '%s\n' 'reordered policy published evidence' >&2; exit 1; }

expect_rejection() {
  local case_name=$1 output=$scratch/$1
  rm -f "$tools/nvcc.counter" "$tools/cuobjdump.counter"
  if LUNAFLUX_SYNTHETIC_TEST_ONLY=true FAKE_CASE=$case_name "$runner" "$tools/nvcc" "$tools/compute-sanitizer" \
    "$tools/cuobjdump" "$tools/nvdisasm" "$approved_policy" "$output" 2 >"$scratch/$case_name.stdout" 2>"$scratch/$case_name.stderr"; then
    printf 'hostile physical campaign accepted: %s\n' "$case_name" >&2; exit 1
  fi
  [[ ! -e $output ]] || { printf 'failed transaction published: %s\n' "$case_name" >&2; exit 1; }
}
for case_name in partial nondeterministic tool_drift comment_only_sass dirty_sanitizer \
  resource_overflow cache_mutation input_guard output_guard canary_zero canary_wrong \
  canary_tail canary_overflow uuid_substitution pci_substitution; do expect_rejection "$case_name"; done
printf '%s\n' 'paged-attention read-only physical hostile transaction tests passed'
