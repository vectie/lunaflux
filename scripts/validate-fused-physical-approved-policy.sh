#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
export LC_ALL

fail() { printf 'fused approved physical policy rejected: %s\n' "$1" >&2; exit 1; }
[[ $# == 9 ]] || fail 'usage: POLICY EXPECTED_SHA256 NVCC PTXAS CUOBJDUMP NVDISASM SANITIZER NVIDIA_SMI DRIVER_RECORD'
policy=$1 expected_sha=$2 nvcc=$3 ptxas=$4 cuobjdump=$5 nvdisasm=$6 sanitizer=$7 nvidia_smi=$8 driver_record=$9
[[ $expected_sha =~ ^[0-9a-f]{64}$ ]] || fail 'policy lacks an out-of-band positional SHA-256 pin'

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
sha256_text() {
  if command -v sha256sum >/dev/null 2>&1; then printf %s "$1" | sha256sum | awk '{print $1}'
  else printf %s "$1" | shasum -a 256 | awk '{print $1}'; fi
}
[[ $policy == /* && $(realpath -- "$policy") == "$policy" && -f $policy && ! -L $policy ]] || fail 'policy is not a canonical regular file'
[[ $(sha256_file "$policy") == "$expected_sha" ]] || fail 'policy digest does not match independent pin'
[[ $(wc -l <"$policy" | tr -d ' ') == 18 ]] || fail 'policy is not canonical 18-line v1'
field() {
  local count value
  count=$(grep -c "^$1=" "$policy")
  [[ $count == 1 ]] || fail "policy $1 is absent or duplicated"
  value=$(sed -n "s/^$1=//p" "$policy")
  [[ -n $value ]] || fail "policy $1 is empty"
  printf '%s\n' "$value"
}
[[ $(field schema) == lunaflux-fused-approved-physical-policy.v1 &&
   $(field target) == sm_120 && $(field compiler_version) == 13.1.115 &&
   $(field nvcc_sha256) == "$(sha256_file "$nvcc")" &&
   $(field ptxas_sha256) == "$(sha256_file "$ptxas")" &&
   $(field cuobjdump_sha256) == "$(sha256_file "$cuobjdump")" &&
   $(field nvdisasm_sha256) == "$(sha256_file "$nvdisasm")" &&
   $(field compute_sanitizer_sha256) == "$(sha256_file "$sanitizer")" &&
   $(field nvidia_smi_sha256) == "$(sha256_file "$nvidia_smi")" ]] || fail 'tool bytes do not match approved policy'
driver_identity=$(sed -n 's/^driver_identity_sha256=//p' "$driver_record")
[[ $(field driver_identity_sha256) == "$driver_identity" &&
   $(field driver_record_sha256) == "$(sha256_file "$driver_record")" ]] || fail 'driver record does not match approved policy'
query() {
  local value
  value=$("$nvidia_smi" --id=0 --query-gpu="$1" --format=csv,noheader,nounits)
  [[ $value != *$'\n'* && -n $value ]] || fail "nvidia-smi $1 observation is ambiguous"
  printf '%s\n' "$value"
}
uuid=$(query uuid); pci=$(query pci.bus_id); name=$(query name)
compute=$(query compute_cap); host_driver=$(query driver_version)
[[ $(field device_uuid) == "$uuid" && $(field device_pci) == "$pci" &&
   $(field device_name_sha256) == "$(sha256_text "$name")" &&
   $(field device_compute) == "$compute" &&
   $(field host_driver_version) == "$host_driver" &&
   $(field policy_authority) == deployment-approved ]] || fail 'device does not match approved policy'
memory=$(field device_total_memory_bytes)
[[ $memory =~ ^[1-9][0-9]*$ ]] || fail 'approved device memory is not canonical'
{
  printf '%s\n' 'schema=lunaflux-fused-approved-device-observation.v1'
  printf 'device_uuid=%s\n' "$uuid"
  printf 'device_pci=%s\n' "$pci"
  printf 'device_name_sha256=%s\n' "$(sha256_text "$name")"
  printf 'device_total_memory_bytes=%s\n' "$memory"
  printf 'device_compute=%s\n' "$compute"
  printf 'host_driver_version=%s\n' "$host_driver"
} | {
  observed=$(mktemp)
  trap 'rm -f -- "$observed"' EXIT
  tee "$observed"
  printf 'device_identity_record_sha256=%s\n' "$(sha256_file "$observed")"
  printf 'approved_policy_sha256=%s\n' "$expected_sha"
}
