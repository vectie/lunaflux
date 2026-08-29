#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"
package=tests/fp8_v2_cuda_probe
runner=scripts/run-fp8-v2-sm120-physical-campaign.sh

fail() {
  printf 'FP8 v2 sm120 physical-probe validation failed: %s\n' "$1" >&2
  exit 1
}

moon check "$package" --target native --deny-warn --warn-list +73
moon test "$package" --target native --deny-warn --warn-list +73

for anchor in \
  'lower_fp8_projection_cuda_aot_candidate_v2' \
  'derive_fp8_projection_operands_v2' \
  'compute_major=12' \
  'compute_minor=0' \
  'patch=115' \
  'operation=id' \
  'simple: lower(plan, 2' \
  'gated: lower(plan, 8'; do
  rg -Fq "$anchor" "$package/candidate.mbt" || fail "production source anchor missing: $anchor"
done
for anchor in \
  'create_ordered_kernel_executor' \
  'OrderedKernelEagerOnly' \
  'executor.enqueue(0)' \
  'executor.record_completion()' \
  'executor.wait_completion()' \
  'executor.reset()' \
  'copy_to_fixed_host' \
  'production scale evidence is invalid' \
  'balance.is_zero()'; do
  rg -Fq "$anchor" "$package/run.mbt" || fail "execution/lifecycle anchor missing: $anchor"
done
for anchor in \
  'fp8_probe_round' \
  'fp8_probe_qkv_expected' \
  'fp8_probe_gated_expected' \
  '@math.expf' \
  '0.03125F + 0.015625F'; do
  rg -Fq "$anchor" "$package/numeric.mbt" || fail "independent referee anchor missing: $anchor"
done
if rg -n 'vectie/lunaflux/release/|vectie/lunaflux/internal/cuda|extern\s+"[cC]"|#external' \
  --glob '*.mbt' "$package" >/dev/null; then
  fail 'probe imports release evidence or bypasses the public CUDA wrapper'
fi
if rg -ni 'nvrtc|--ptx|compute_[0-9]+,code=compute|runtime.{0,12}compil|dispatch.canary' \
  --glob '*.mbt' "$package" "$runner" >/dev/null; then
  fail 'probe introduced runtime compilation, PTX fallback, or canaries'
fi
for anchor in \
  'ABSOLUTE_NVCC_13_1' \
  'compiler_version == 13.1.115' \
  'arch=compute_120,code=sm_120' \
  'cmp -s "$stage/builds/$family/first/kernel.cubin"' \
  'run_sanitizer memcheck' \
  'run_sanitizer racecheck' \
  'run_sanitizer initcheck' \
  'physical_cuda_observed=true' \
  'manifest_bindable=false' \
  'readiness=false' \
  'lunaflux_prepare_evidence_manifest' \
  'OUTER_SEAL.sha256'; do
  rg -Fq "$anchor" "$runner" || fail "campaign anchor missing: $anchor"
done
scripts/validate-fp8-reusable-paged-executor-v3.sh >/dev/null

if [[ ${1:-} == --static-only ]]; then
  printf '%s\n' 'FP8 v2 sm120 physical probe static boundary passed.'
  exit 0
fi
[[ $# == 0 ]] || fail 'usage: validate-fp8-v2-sm120-physical-probe.sh [--static-only]'

scratch=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-fp8-v2-physical-fake.XXXXXX")
scratch=$(CDPATH= cd -- "$scratch" && pwd -P)
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
while [[ $# -gt 0 ]]; do
  case $1 in
    -o) output=$2; shift 2 ;;
    *) shift ;;
  esac
done
[[ -n $output ]]
cp kernel.cu "$output"
if [[ ${FAKE_FP8_CASE:-pass} == nondeterministic ]]; then
  counter=$(dirname -- "$0")/counter
  value=0
  [[ ! -f $counter ]] || value=$(<"$counter")
  value=$((value + 1))
  printf '%s' "$value" >"$counter"
  printf '\nnonce=%s\n' "$value" >>"$output"
fi
EOF
cat >"$tools/ptxas" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' 'ptxas release 13.1, V13.1.115'
EOF
cat >"$tools/compute-sanitizer" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
log=
cycles=
while [[ $# -gt 0 ]]; do
  case $1 in
    --log-file) log=$2; shift 2 ;;
    sm_120) cycles=${2:-2}; shift ;;
    *) shift ;;
  esac
done
[[ -n $log ]]
if [[ ${FAKE_FP8_CASE:-pass} == sanitizer_error ]]; then
  printf '%s\n' '========= ERROR SUMMARY: 1 error' >"$log"
else
  printf '%s\n' '========= ERROR SUMMARY: 0 errors' >"$log"
fi
cycles=${FP8_FAKE_CYCLES:-2}
printf '%s\n' "outcome=fp8-v2-sm120-physical-pass cycles=$cycles families=qkv,gated-mlp launches=$((cycles * 2)) numeric_values=$((cycles * 48)) qkv_max_abs_error=0 gated_max_abs_error=0 absolute_tolerance=0.03125 relative_tolerance=0.015625 cpu_oracle=independent-ordered-f32-v1 abi=production-paged-v4-operands-v2 diagnostic=scale-cells-finite-positive target=sm_120 device_ordinal=0 device_name_sha256=dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd device_total_memory_bytes=17179869184 resources=context0,stream0,executor0,allocation0,module0,function0,device_bytes0,pending0,cleanup0 manifest_bindable=false readiness=false"
EOF
chmod 0555 "$tools/nvcc" "$tools/ptxas" "$tools/compute-sanitizer"

passed=$scratch/passed
FP8_FAKE_CYCLES=2 "$runner" "$tools/nvcc" "$tools/compute-sanitizer" \
  "$passed" 2 >"$scratch/pass.stdout" 2>"$scratch/pass.stderr" || {
  sed -n '1,160p' "$scratch/pass.stderr" >&2
  fail 'ordinary synthetic campaign was rejected'
}
if [[ -s $scratch/pass.stderr ]]; then
  sed -n '1,160p' "$scratch/pass.stderr" >&2
  fail 'ordinary synthetic campaign emitted stderr'
fi
grep -Fx 'outcome=fp8-v2-sm120-physical-campaign-pass' "$passed/RESULT.txt" >/dev/null
grep -Fx 'physical_cuda_observed=true' "$passed/RESULT.txt" >/dev/null
grep -Fx 'manifest_bindable=false' "$passed/RESULT.txt" >/dev/null
grep -Fx 'readiness=false' "$passed/RESULT.txt" >/dev/null
[[ $(wc -l <"$passed/OUTER_SEAL.sha256" | tr -d ' ') == 2 ]] || fail 'outer seal drifted'
[[ -z $(find "$passed" -type f -perm -u+w -print -quit) ]] || fail 'evidence is writable'

if "$runner" "$tools/nvcc" "$tools/compute-sanitizer" "$passed" 2 \
  >"$scratch/overwrite.stdout" 2>"$scratch/overwrite.stderr"; then
  fail 'campaign overwrote existing evidence'
fi
for hostile in nondeterministic sanitizer_error; do
  rm -f "$tools/counter"
  if FAKE_FP8_CASE=$hostile FP8_FAKE_CYCLES=2 "$runner" "$tools/nvcc" \
    "$tools/compute-sanitizer" "$scratch/$hostile" 2 \
    >"$scratch/$hostile.stdout" 2>"$scratch/$hostile.stderr"; then
    fail "hostile synthetic campaign was accepted: $hostile"
  fi
  [[ ! -e $scratch/$hostile ]] || fail "failed campaign published: $hostile"
done

printf '%s\n' 'FP8 v2 sm120 physical probe synthetic campaign passed.'
