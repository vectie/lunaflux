#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
export LC_ALL
export FAKE_LUNA_CUDA_VERSION=13.1.115
export LUNAFLUX_SYNTHETIC_PHYSICAL_CAMPAIGN=1

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
scratch=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-fused-campaign-test.XXXXXX")
scratch=$(CDPATH= cd -- "$scratch" && pwd -P)
cleanup() {
  chmod -R u+rwX "$scratch" 2>/dev/null || true
  rm -rf -- "$scratch"
}
trap cleanup EXIT HUP INT TERM

fail() {
  printf 'fused physical campaign test failed: %s\n' "$1" >&2
  exit 1
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

toolchain=$scratch/toolchain
mkdir "$toolchain"
cp "$repo_root/scripts/fixtures/luna-cuda-aot/fake-nvcc.sh" "$toolchain/nvcc"
cp "$repo_root/scripts/fixtures/luna-cuda-aot/fake-ptxas.sh" "$toolchain/ptxas"
cp "$repo_root/scripts/fixtures/fused-physical-campaign/fake-cuobjdump.sh" \
  "$toolchain/cuobjdump"
cp "$repo_root/scripts/fixtures/fused-physical-campaign/fake-nvdisasm.sh" \
  "$toolchain/nvdisasm"
cp "$repo_root/scripts/fixtures/fused-physical-campaign/fake-nvidia-smi.sh" \
  "$toolchain/nvidia-smi"
cp "$repo_root/scripts/fixtures/fused-physical-campaign/fake-compute-sanitizer.sh" \
  "$toolchain/compute-sanitizer"
chmod 0555 "$toolchain/nvcc" "$toolchain/ptxas" "$toolchain/cuobjdump" \
  "$toolchain/nvdisasm" \
  "$toolchain/nvidia-smi" \
  "$toolchain/compute-sanitizer"

driver_report=$scratch/driver.txt
"$repo_root/scripts/inspect-luna-cuda-aot-driver.sh" \
  "$toolchain/nvcc" >"$driver_report"
driver_identity=$(sed -n '6s/^driver_identity_sha256=//p' "$driver_report")
[[ $driver_identity =~ ^[0-9a-f]{64}$ ]] || fail 'fake driver identity missing'
manifest=$scratch/toolchain.manifest
fake_name_sha=$(printf %s 'Fake SM120 GPU' | shasum -a 256 | awk '{print $1}')
printf '%s\n' \
  'schema=lunaflux-fused-approved-physical-policy.v1' \
  'target=sm_120' \
  'compiler_version=13.1.115' \
  "nvcc_sha256=$(sha256_file "$toolchain/nvcc")" \
  "ptxas_sha256=$(sha256_file "$toolchain/ptxas")" \
  "cuobjdump_sha256=$(sha256_file "$toolchain/cuobjdump")" \
  "nvdisasm_sha256=$(sha256_file "$toolchain/nvdisasm")" \
  "compute_sanitizer_sha256=$(sha256_file "$toolchain/compute-sanitizer")" \
  "nvidia_smi_sha256=$(sha256_file "$toolchain/nvidia-smi")" \
  "driver_identity_sha256=$driver_identity" \
  "driver_record_sha256=$(sha256_file "$driver_report")" \
  'device_uuid=GPU-fake-sm120-qualification' \
  'device_pci=00000000:01:00.0' \
  "device_name_sha256=$fake_name_sha" \
  'device_total_memory_bytes=17179869184' \
  'device_compute=12.0' \
  'host_driver_version=590.48.01' \
  'policy_authority=deployment-approved' >"$manifest"
manifest_digest=$(sha256_file "$manifest")

runner=$repo_root/scripts/run-fused-parallel-physical-campaign.sh
success=$scratch/success
if ! "$runner" "$toolchain/nvcc" "$manifest" "$manifest_digest" "$toolchain/compute-sanitizer" \
  "$toolchain/nvidia-smi" "$success" 1 >"$scratch/success.stdout" 2>"$scratch/success.stderr"; then
  sed -n '1,160p' "$scratch/success.stderr" >&2
  fail 'ordinary fake campaign was rejected'
fi
[[ ! -s $scratch/success.stderr ]] || fail 'successful campaign emitted stderr'
[[ -d $success && ! -L $success ]] || fail 'successful campaign was not published'
grep -F 'outcome=fused-physical-campaign-published' \
  "$scratch/success.stdout" >/dev/null || fail 'publication outcome missing'
grep -Fx 'outcome=fused-synthetic-campaign-pass' \
  "$success/CAMPAIGN_RESULT.txt" >/dev/null ||
  fail 'sealed result is missing'
grep -Fx 'physical_cuda_observed=false' \
  "$success/CAMPAIGN_RESULT.txt" >/dev/null ||
  fail 'synthetic campaign claimed physical CUDA observation'
grep -Fx 'admission_authority=absent' \
  "$success/CAMPAIGN_RESULT.txt" >/dev/null ||
  fail 'synthetic campaign claimed evidence admission authority'
grep -F '  CAMPAIGN_RESULT.txt' "$success/FILES.sha256" >/dev/null ||
  fail 'campaign outcome is outside the outer manifest'
grep -F 'avoiding a self-referential digest' \
  "$success/admission/SEAL_SCOPE.txt" >/dev/null || fail 'seal scope is missing'
for payload in "$success/artifacts" "$success/measurements" "$success"; do
  [[ -f $payload/FILES.sha256 ]] || fail 'sealed manifest is missing'
  while read -r digest relative; do
    relative=${relative#  }
    [[ $(sha256_file "$payload/$relative") == "$digest" ]] ||
      fail "sealed manifest entry drifted: $relative"
  done <"$payload/FILES.sha256"
done
[[ ! -w $success && ! -w $success/artifacts && ! -w $success/measurements ]] ||
  fail 'published campaign is writable'

success_manifest=$(sha256_file "$success/FILES.sha256")
if "$runner" "$toolchain/nvcc" "$manifest" "$manifest_digest" "$toolchain/compute-sanitizer" \
  "$toolchain/nvidia-smi" "$success" 1 >/dev/null 2>&1; then
  fail 'runner overwrote an existing campaign'
fi
[[ $(sha256_file "$success/FILES.sha256") == "$success_manifest" ]] ||
  fail 'overwrite rejection mutated existing evidence'

unsafe_output="$scratch/unsafe output"
if "$runner" "$toolchain/nvcc" "$manifest" "$manifest_digest" "$toolchain/compute-sanitizer" \
  "$toolchain/nvidia-smi" "$unsafe_output" 1 >/dev/null 2>&1; then
  fail 'runner accepted an ambiguous output basename'
fi
[[ ! -e $unsafe_output && ! -L $unsafe_output ]] ||
  fail 'ambiguous output basename created a path'

assert_rejected_without_partial() {
  local mode=$1
  local output=$scratch/rejected-$mode
  if FAKE_FUSED_CAMPAIGN_MODE=$mode "$runner" "$toolchain/nvcc" "$manifest" "$manifest_digest" \
    "$toolchain/compute-sanitizer" "$toolchain/nvidia-smi" "$output" 1 >/dev/null 2>&1; then
    fail "hostile campaign passed: $mode"
  fi
  [[ ! -e $output && ! -L $output ]] ||
    fail "hostile campaign published partial output: $mode"
  compgen -G "$scratch/.rejected-$mode.partial.*" >/dev/null &&
    fail "hostile campaign retained staging output: $mode"
  return 0
}

assert_rejected_without_partial dirty-sanitizer
assert_rejected_without_partial trailing-summary
assert_rejected_without_partial extra-summary
assert_rejected_without_partial stderr
assert_rejected_without_partial ambiguous-number

if FAKE_FUSED_CAMPAIGN_MODE=identity-drift \
  FAKE_FUSED_CAMPAIGN_NVCC=$toolchain/nvcc \
  "$runner" "$toolchain/nvcc" "$manifest" "$manifest_digest" "$toolchain/compute-sanitizer" \
  "$toolchain/nvidia-smi" "$scratch/rejected-identity" 1 >/dev/null 2>&1; then
  fail 'toolchain identity drift passed'
fi
[[ ! -e $scratch/rejected-identity ]] || fail 'identity drift published output'
compgen -G "$scratch/.rejected-identity.partial.*" >/dev/null &&
  fail 'identity drift retained staging output'

printf '%s\n' 'Fused physical campaign transaction gate passed.'
