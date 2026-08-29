#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
runner=$repo_root/scripts/run-luna-tensor-core-physical-campaign.sh
scratch=$(mktemp -d /private/tmp/lunaflux-tensor-core-campaign-test.XXXXXX)
cleanup() {
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
output=
resource=false
while [[ $# -gt 0 ]]; do
  case $1 in
    -o) output=$2; shift 2 ;;
    -Xptxas=-v) resource=true; shift ;;
    *) shift ;;
  esac
done
[[ -n $output ]]
cp kernel.cu "$output"
if [[ ${FAKE_CASE:-success} == nondeterministic ]]; then
  counter=$(dirname -- "$0")/nvcc.counter
  value=0
  [[ ! -f $counter ]] || value=$(cat "$counter")
  value=$((value + 1))
  printf '%s' "$value" >"$counter"
  printf '\nnonce=%s\n' "$value" >>"$output"
fi
if $resource; then
  registers=64
  [[ ${FAKE_CASE:-success} != resource_overflow ]] || registers=129
  printf 'ptxas info    : 0 bytes gmem\n' >&2
  printf 'ptxas info    : Compiling entry function for sm_120\n' >&2
  printf 'ptxas info    : Function properties\n' >&2
  printf '    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n' >&2
  printf 'ptxas info    : Used %s registers, 0 bytes smem, 0 bytes cmem[0]\n' "$registers" >&2
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
      counter=$(dirname -- "$0")/cuobjdump.counter
      value=0
      [[ ! -f $counter ]] || value=$(cat "$counter")
      value=$((value + 1))
      printf '%s' "$value" >"$counter"
      printf 'cuobjdump 13.1.%s\n' "$value"
    else
      printf '%s\n' 'cuobjdump 13.1.0'
    fi
    ;;
  --dump-sass)
    case ${FAKE_CASE:-success} in
      missing_sass) ;;
      comment_only) printf '%s\n' '// MMA is required but not observed' ;;
      ambiguous_sass)
        printf '%s\n' '/*0000*/ TCGEN05.MMA.16x16x16.BF16 R0, R2;'
        printf '%s\n' '/*0010*/ HMMA.16816.F32 R4, R6;'
        ;;
      *) printf '%s\n' '/*0000*/ TCGEN05.MMA.16x16x16.BF16 R0, R2;' ;;
    esac
    ;;
  --dump-resource-usage)
    registers=64
    [[ ${FAKE_CASE:-success} != resource_overflow ]] || registers=129
    printf 'Function lunaflux_lunatile_wmma_bf16_m16n16k16_sm120_v1:\n'
    local_bytes=0
    [[ ${FAKE_CASE:-success} != resource_nonzero_local ]] || local_bytes=16
    printf ' REG:%s STACK:0 SHARED:0 LOCAL:%s\n' "$registers" "$local_bytes"
    ;;
  *) exit 2 ;;
esac
EOF

cat >"$tools/nvdisasm" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --version ]]; then
  printf '%s\n' 'nvdisasm 13.1.0'
else
  case ${FAKE_CASE:-success} in
    missing_sass) ;;
    comment_only) printf '%s\n' '// MMA is required but not observed' ;;
    ambiguous_sass)
      printf '%s\n' '/*0000*/ TCGEN05.MMA.16x16x16.BF16 R0, R2;'
      printf '%s\n' '/*0010*/ HMMA.16816.F32 R4, R6;'
      ;;
    *) printf '%s\n' '/*0000*/ TCGEN05.MMA.16x16x16.BF16 R0, R2;' ;;
  esac
fi
EOF

cat >"$tools/compute-sanitizer" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --version ]]; then
  printf '%s\n' 'NVIDIA Compute Sanitizer 2026.1.0'
  exit 0
fi
log=
while [[ $# -gt 0 ]]; do
  if [[ $1 == --log-file ]]; then
    log=$2
    shift 2
  else
    shift
  fi
done
[[ -n $log ]]
if [[ ${FAKE_CASE:-success} == dirty_sanitizer ]]; then
  printf '%s\n' '========= ERROR SUMMARY: 1 error' >"$log"
else
  printf '%s\n' '========= ERROR SUMMARY: 0 errors' >"$log"
fi
printf '%s\n' \
  'outcome=lunatile-tensor-core-sm120-qualification-pass cycles=2 numeric_case_count=32768 launches=4 cpu_vs_serial_max_abs_error=0 cpu_vs_tensor_core_max_abs_error=0 serial_vs_tensor_core_max_abs_error=0 absolute_tolerance=0.001 relative_tolerance=0.0001 cpu_oracle=independent-ordered-f32-v1 serial_cuda_oracle=passed serial_source_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa tensor_core_source_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb tensor_core_candidate_sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc target=sm_120 device_ordinal=0 device_name_sha256=dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd device_total_memory_bytes=25769803776 resources=context0,stream0,allocation0,module0,function0,device_bytes0,pending0,cleanup0 manifest_bindable=false promotion_authority=absent'
EOF
chmod +x "$tools"/*

success=$scratch/success
if ! FAKE_CASE=success "$runner" "$tools/nvcc" "$tools/compute-sanitizer" \
  "$tools/cuobjdump" "$tools/nvdisasm" "$success" 2 >"$scratch/success.stdout" \
  2>"$scratch/success.stderr"; then
  cat "$scratch/success.stderr" >&2
  printf '%s\n' 'successful fake campaign was rejected' >&2
  exit 1
fi
[[ ! -s $scratch/success.stderr ]] || {
  cat "$scratch/success.stderr" >&2
  printf '%s\n' 'successful fake campaign emitted stderr' >&2
  exit 1
}
grep -F 'outcome=lunatile-tensor-core-physical-campaign-published ' \
  "$scratch/success.stdout" >/dev/null
grep -Fx 'schema=lunaflux-lunatile-tensor-core-physical-campaign.v1' \
  "$success/RESULT.txt" >/dev/null
[[ $(wc -l <"$success/RESULT.txt" | tr -d ' ') == 67 ]]
grep -E '^driver_record_sha256=[0-9a-f]{64}$' "$success/RESULT.txt" >/dev/null
grep -Fx 'sass_instruction_family=TCGEN05_MMA' "$success/RESULT.txt" >/dev/null
grep -Fx 'manifest_bindable=false' "$success/RESULT.txt" >/dev/null
grep -Fx 'promotion_authority=absent' "$success/RESULT.txt" >/dev/null
[[ $(wc -l <"$success/OUTER_SEAL.sha256" | tr -d ' ') == 2 ]]
(
  cd "$success"
  while read -r digest path; do
    [[ $(shasum -a 256 "$path" | awk '{print $1}') == "$digest" ]]
  done <OUTER_SEAL.sha256
)
[[ -z $(find "$success" -type f -perm -u+w -print -quit) ]]

if FAKE_CASE=success "$runner" "$tools/nvcc" "$tools/compute-sanitizer" \
  "$tools/cuobjdump" "$tools/nvdisasm" "$success" 2 \
  >"$scratch/overwrite.stdout" 2>"$scratch/overwrite.stderr"; then
  printf '%s\n' 'campaign overwrote existing evidence' >&2
  exit 1
fi

expect_rejection() {
  local case_name=$1 output_path
  output_path=$scratch/$case_name
  rm -f "$tools/nvcc.counter" "$tools/cuobjdump.counter"
  if FAKE_CASE=$case_name "$runner" "$tools/nvcc" "$tools/compute-sanitizer" \
    "$tools/cuobjdump" "$tools/nvdisasm" "$output_path" 2 \
    >"$scratch/$case_name.stdout" 2>"$scratch/$case_name.stderr"; then
    printf 'hostile fake campaign was accepted: %s\n' "$case_name" >&2
    exit 1
  fi
  [[ ! -e $output_path ]] || {
    printf 'failed transaction published evidence: %s\n' "$case_name" >&2
    exit 1
  }
}

expect_rejection missing_sass
expect_rejection comment_only
expect_rejection ambiguous_sass
expect_rejection dirty_sanitizer
expect_rejection resource_overflow
expect_rejection resource_nonzero_local
expect_rejection nondeterministic
expect_rejection tool_drift

printf '%s\n' 'Luna tensor-core physical campaign hostile transaction tests passed'
