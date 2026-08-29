#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"

bash -n scripts/run-luna-tensor-core-physical-campaign.sh
bash -n scripts/test-luna-tensor-core-physical-campaign.sh
scripts/validate-luna-tensor-core-cuda-probe.sh

for required in \
  'schema=lunaflux-lunatile-tensor-core-physical-campaign.v1' \
  "compiler_version == 13.1.115" \
  'driver_record_sha256=$driver_record_sha' \
  'sass_instruction_family=$sass_family' \
  'cuobjdump_sass_sha256=$cu_sass_sha' \
  'cuobjdump_sass_instruction_count=$cu_sass_count' \
  'nvdisasm_sass_sha256=$nv_sass_sha' \
  'nvdisasm_sass_instruction_count=$nv_sass_count' \
  'register_bound=128' \
  'static_shared_bound=0' \
  'spill_store_bytes=$spill_store' \
  'cpu_vs_serial_max_abs_error=$cpu_serial' \
  'cpu_vs_tensor_core_max_abs_error=$cpu_tensor' \
  'serial_vs_tensor_core_max_abs_error=$serial_tensor' \
  'memcheck_error_summary_count=1' \
  'racecheck_error_summary_count=1' \
  'initcheck_error_summary_count=1' \
  'cleanup_balance=context0,stream0,allocation0,module0,function0,device_bytes0,pending0,cleanup0' \
  'evidence_files_manifest_sha256=$inner_sha' \
  'OUTER_SEAL.sha256' \
  'manifest_bindable=false' \
  'promotion_authority=absent'; do
  rg -F -q "$required" scripts/run-luna-tensor-core-physical-campaign.sh || {
    printf 'tensor-core physical campaign anchor missing: %s\n' "$required" >&2
    exit 1
  }
done

if rg -n 'RequireExternallyQualifiedTensorCore|manifest_bindable=true|promotion_authority=(present|granted)|NVRTC|nvrtc' \
  scripts/run-luna-tensor-core-physical-campaign.sh \
  scripts/test-luna-tensor-core-physical-campaign.sh; then
  printf '%s\n' 'tensor-core campaign acquired production authority' >&2
  exit 1
fi

scripts/test-luna-tensor-core-physical-campaign.sh
printf '%s\n' 'Luna tensor-core physical campaign boundary passed'
