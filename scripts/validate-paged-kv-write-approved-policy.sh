#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
export LC_ALL
fail() { printf 'paged KV-write approved policy rejected: %s\n' "$1" >&2; exit 1; }
[[ $# == 8 ]] || fail 'usage: POLICY#sha256=DIGEST NVCC PTXAS CUOBJDUMP NVDISASM SANITIZER NVIDIA_SMI DRIVER_RECORD'
argument=$1 nvcc=$2 ptxas=$3 cuobjdump=$4 nvdisasm=$5 sanitizer=$6 nvidia_smi=$7 driver_record=$8
policy=${argument%#sha256=*}; expected_sha=${argument##*#sha256=}
[[ $policy != "$argument" && $expected_sha =~ ^[0-9a-f]{64}$ ]] || fail 'policy lacks independent SHA-256 pin'
sha256_file() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
sha256_text() { if command -v sha256sum >/dev/null 2>&1; then printf %s "$1" | sha256sum | awk '{print $1}'; else printf %s "$1" | shasum -a 256 | awk '{print $1}'; fi; }
[[ $policy == /* && $(realpath -- "$policy") == "$policy" && -f $policy && ! -L $policy ]] || fail 'policy is not canonical regular file'
[[ $(sha256_file "$policy") == "$expected_sha" ]] || fail 'policy digest does not match pin'
[[ $(wc -l <"$policy" | tr -d ' ') == 18 ]] || fail 'policy is not canonical 18-line v1'
field() { local count value; count=$(grep -c "^$1=" "$policy"); [[ $count == 1 ]] || fail "policy $1 absent/duplicated"; value=$(sed -n "s/^$1=//p" "$policy"); [[ -n $value ]] || fail "policy $1 empty"; printf '%s\n' "$value"; }
[[ $(field schema) == lunaflux-paged-kv-write-approved-physical-policy.v1 &&
   $(field target) == sm_120 && $(field compiler_version) == 13.1.115 &&
   $(field nvcc_sha256) == "$(sha256_file "$nvcc")" &&
   $(field ptxas_sha256) == "$(sha256_file "$ptxas")" &&
   $(field cuobjdump_sha256) == "$(sha256_file "$cuobjdump")" &&
   $(field nvdisasm_sha256) == "$(sha256_file "$nvdisasm")" &&
   $(field compute_sanitizer_sha256) == "$(sha256_file "$sanitizer")" &&
   $(field nvidia_smi_sha256) == "$(sha256_file "$nvidia_smi")" ]] || fail 'tool bytes mismatch policy'
driver_identity=$(sed -n 's/^driver_identity_sha256=//p' "$driver_record")
[[ $(field driver_identity_sha256) == "$driver_identity" &&
   $(field driver_record_sha256) == "$(sha256_file "$driver_record")" ]] || fail 'driver record mismatch policy'
query() { local value; value=$("$nvidia_smi" --id=0 --query-gpu="$1" --format=csv,noheader,nounits); [[ $value != *$'\n'* && -n $value ]] || fail "$1 observation ambiguous"; printf '%s\n' "$value"; }
uuid=$(query uuid); pci=$(query pci.bus_id); name=$(query name); compute=$(query compute_cap); host_driver=$(query driver_version)
[[ $(field device_uuid) == "$uuid" && $(field device_pci) == "$pci" &&
   $(field device_name_sha256) == "$(sha256_text "$name")" &&
   $(field device_compute) == "$compute" && $(field host_driver_version) == "$host_driver" &&
   $(field policy_authority) == deployment-approved ]] || fail 'device mismatch policy'
memory=$(field device_total_memory_bytes); [[ $memory =~ ^[1-9][0-9]*$ ]] || fail 'memory is not canonical'
printf '%s\n' \
  "approved_policy_sha256=$expected_sha" "device_uuid=$uuid" "device_pci=$pci" \
  "host_driver_version=$host_driver" "device_name_sha256=$(sha256_text "$name")" \
  "device_total_memory_bytes=$memory" "device_compute=$compute"
