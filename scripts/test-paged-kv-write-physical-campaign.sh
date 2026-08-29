#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
export LC_ALL
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
scratch=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-kv-write-campaign.XXXXXX"); scratch=$(CDPATH= cd -- "$scratch" && pwd -P)
cleanup() { chmod -R u+rwX "$scratch" 2>/dev/null || true; rm -rf -- "$scratch"; }
trap cleanup EXIT HUP INT TERM
fail() { printf 'paged KV-write campaign test failed: %s\n' "$1" >&2; exit 1; }
sha256_file() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
tools=$scratch/tools; mkdir "$tools"
cp "$root/scripts/fixtures/luna-cuda-aot/fake-nvcc.sh" "$tools/nvcc"
cp "$root/scripts/fixtures/luna-cuda-aot/fake-ptxas.sh" "$tools/ptxas"
cp "$root/scripts/fixtures/fused-physical-campaign/fake-cuobjdump.sh" "$tools/cuobjdump"
cp "$root/scripts/fixtures/fused-physical-campaign/fake-nvdisasm.sh" "$tools/nvdisasm"
cp "$root/scripts/fixtures/fused-physical-campaign/fake-nvidia-smi.sh" "$tools/nvidia-smi"
cp "$root/scripts/fixtures/paged-kv-write-physical/fake-compute-sanitizer.sh" "$tools/compute-sanitizer"
chmod 0555 "$tools"/*
export FAKE_LUNA_CUDA_VERSION=13.1.115
driver=$scratch/driver.v1; "$root/scripts/inspect-luna-cuda-aot-driver.sh" "$tools/nvcc" >"$driver"
driver_identity=$(sed -n 's/^driver_identity_sha256=//p' "$driver")
name_sha=$(printf %s 'Fake SM120 GPU' | shasum -a 256 | awk '{print $1}')
policy=$scratch/policy.v1
printf '%s\n' \
  'schema=lunaflux-paged-kv-write-approved-physical-policy.v1' 'target=sm_120' 'compiler_version=13.1.115' \
  "nvcc_sha256=$(sha256_file "$tools/nvcc")" "ptxas_sha256=$(sha256_file "$tools/ptxas")" \
  "cuobjdump_sha256=$(sha256_file "$tools/cuobjdump")" "nvdisasm_sha256=$(sha256_file "$tools/nvdisasm")" \
  "compute_sanitizer_sha256=$(sha256_file "$tools/compute-sanitizer")" "nvidia_smi_sha256=$(sha256_file "$tools/nvidia-smi")" \
  "driver_identity_sha256=$driver_identity" "driver_record_sha256=$(sha256_file "$driver")" \
  'device_uuid=GPU-fake-sm120-qualification' 'device_pci=00000000:01:00.0' "device_name_sha256=$name_sha" \
  'device_total_memory_bytes=17179869184' 'device_compute=12.0' 'host_driver_version=590.48.01' 'policy_authority=deployment-approved' >"$policy"
policy_argument=$policy#sha256=$(sha256_file "$policy")
runner=$root/scripts/run-paged-kv-write-physical-campaign.sh
export LUNAFLUX_SYNTHETIC_TEST_ONLY=true
success=$scratch/success
if ! "$runner" "$tools/nvcc" "$policy_argument" "$tools/compute-sanitizer" "$tools/nvidia-smi" "$success" 1 >"$scratch/success.stdout" 2>"$scratch/success.stderr"; then
  sed -n '1,160p' "$scratch/success.stdout" >&2
  sed -n '1,160p' "$scratch/success.stderr" >&2
  fail 'ordinary fake campaign rejected'
fi
[[ ! -s $scratch/success.stderr && -d $success && ! -w $success ]] || fail 'successful campaign publication invalid'
grep -Fx 'outcome=paged-kv-write-synthetic-campaign-pass' "$success/RESULT.txt" >/dev/null || fail 'synthetic result missing'
grep -Fx 'physical_cuda_observed=false' "$success/RESULT.txt" >/dev/null || fail 'synthetic result fabricated physical observation'
grep -Fx 'admission_exercised=false' "$success/admission/synthetic-test-only.txt" >/dev/null || fail 'synthetic run exercised production admission'
grep -Fx 'physical_cuda_observed=false' "$success/admission/paged-kv-write-physical-evidence.v1" >/dev/null || fail 'synthetic canonical fabricated physical observation'
grep -Fx 'synthetic_test_only=true' "$success/admission/paged-kv-write-physical-evidence.v1" >/dev/null || fail 'synthetic canonical lacks test marker'
for payload in "$success/artifacts" "$success/measurements" "$success"; do
  while read -r digest relative; do relative=${relative#  }; [[ $(sha256_file "$payload/$relative") == "$digest" ]] || fail "seal drift: $relative"; done <"$payload/FILES.sha256"
done
probe=$root/_build/native/debug/build/tests/paged_kv_write_cuda_probe/paged_kv_write_cuda_probe.exe
candidate=$(find "$success/artifacts/build-candidate-first/kernels/sha256" -type f -name '*.cubin' -print)
receipt=$success/artifacts/positioned-paged-kv-write-compile-receipt.v1
canonical=$success/admission/paged-kv-write-physical-evidence.v1
evidence_sha=$(sed -n 's/^evidence_sha256=//p' "$success/RESULT.txt")
files_sha=$(sed -n 's/^files_manifest_sha256=//p' "$success/RESULT.txt")
outer_sha=$(sed -n 's/^outer_seal_sha256=//p' "$success/RESULT.txt")
if "$probe" admit "${policy_argument##*#sha256=}" "$candidate" "$candidate" "$receipt" "$canonical" "$evidence_sha" "${policy_argument##*#sha256=}" "$files_sha" "$outer_sha" >/dev/null 2>&1; then
  fail 'public typed admission accepted synthetic canonical replay'
fi
if "$runner" "$tools/nvcc" "$policy_argument" "$tools/compute-sanitizer" "$tools/nvidia-smi" "$success" 1 >/dev/null 2>&1; then fail 'existing campaign overwritten'; fi
reject() {
  local mode=$1 output=$scratch/rejected-$mode
  if FAKE_KV_WRITE_MODE=$mode "$runner" "$tools/nvcc" "$policy_argument" "$tools/compute-sanitizer" "$tools/nvidia-smi" "$output" 1 >/dev/null 2>&1; then fail "hostile campaign passed: $mode"; fi
  [[ ! -e $output ]] || fail "hostile campaign published: $mode"
  compgen -G "$scratch/.rejected-$mode.partial.*" >/dev/null && fail "hostile campaign retained staging: $mode"
  return 0
}
for mode in dirty trailing stderr wrong-canary guard sentinel silent; do reject "$mode"; done
if FAKE_LUNA_CUDA_NONDETERMINISTIC=1 "$runner" "$tools/nvcc" "$policy_argument" "$tools/compute-sanitizer" "$tools/nvidia-smi" "$scratch/rejected-nondeterministic" 1 >/dev/null 2>&1; then fail 'nondeterministic CUBIN campaign passed'; fi
[[ ! -e $scratch/rejected-nondeterministic ]] || fail 'nondeterministic campaign published'
wrong_policy=$scratch/wrong-policy.v1; cp "$policy" "$wrong_policy"; chmod u+w "$tools/cuobjdump"; printf '%s\n' '# substituted' >>"$tools/cuobjdump"; chmod 0555 "$tools/cuobjdump"
wrong_argument=$wrong_policy#sha256=$(sha256_file "$wrong_policy")
if "$runner" "$tools/nvcc" "$wrong_argument" "$tools/compute-sanitizer" "$tools/nvidia-smi" "$scratch/rejected-tool" 1 >/dev/null 2>&1; then fail 'unapproved tool substitution passed'; fi
if "$runner" "$tools/nvcc" "$policy#sha256=$(printf '%064d' 0)" "$tools/compute-sanitizer" "$tools/nvidia-smi" "$scratch/rejected-policy-pin" 1 >/dev/null 2>&1; then fail 'wrong policy pin passed'; fi
printf '%s\n' 'Paged KV-write physical campaign transaction gate passed.'
