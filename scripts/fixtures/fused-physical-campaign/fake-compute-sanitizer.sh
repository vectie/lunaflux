#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --version ]]; then
  printf '%s\n' 'NVIDIA Compute Sanitizer 2026.1.0 fake qualification fixture'
  exit 0
fi

tool=
log_file=
while (( $# > 0 )); do
  case $1 in
    --tool)
      tool=$2
      shift 2
      ;;
    --log-file)
      log_file=$2
      shift 2
      ;;
    --leak-check|--error-exitcode)
      shift 2
      ;;
    --*)
      shift
      ;;
    *)
      break
      ;;
  esac
done
[[ $tool == memcheck || $tool == racecheck || $tool == initcheck ]]
[[ $log_file == /* ]]

case ${FAKE_FUSED_CAMPAIGN_MODE:-pass} in
  dirty-sanitizer)
    printf '%s\n' '========= ERROR SUMMARY: 1 error' >"$log_file"
    ;;
  trailing-summary)
    printf '%s\n' '========= ERROR SUMMARY: 0 errors injected' >"$log_file"
    ;;
  extra-summary)
    printf '%s\n' \
      '========= ERROR SUMMARY: 0 errors' \
      '========= ERROR SUMMARY: 0 errors' >"$log_file"
    ;;
  *)
    printf '%s\n' '========= ERROR SUMMARY: 0 errors' >"$log_file"
    ;;
esac

cycles=${!#}
if [[ ${FAKE_FUSED_CAMPAIGN_MODE:-pass} == ambiguous-number ]]; then
  cycles=01
fi
printf '%s\n' \
  "outcome=fused-parallel-sm120-qualification-pass cycles=$cycles shapes=4 launches=$((10#$cycles * 20)) qkv_max_abs_error_ppb=0 qkv_max_relative_error_ppb=0 qkv_observed_dispatch_canary=3084 residual_max_abs_error_ppb=0 residual_max_relative_error_ppb=0 residual_observed_dispatch_canary=3156 standalone_qkv_oracle=pass deterministic_cubins=2x5 target=sm_120 device_uuid=GPU-fake-sm120-qualification device_pci=00000000:01:00.0 device_name_sha256=268f7ae0d581b571300341fdce54fb977486376b0813d5ad35ee69815504340d device_total_memory_bytes=17179869184 cuda_driver_version=13010 resources=context0,stream0,event0,graph0,graph_exec0,allocation0,module0,function0,device_bytes0,pending0,cleanup0 authority=qualification-only"

if [[ ${FAKE_FUSED_CAMPAIGN_MODE:-pass} == stderr ]]; then
  printf '%s\n' 'hostile fake sanitizer stderr' >&2
fi
if [[ ${FAKE_FUSED_CAMPAIGN_MODE:-pass} == identity-drift &&
      -n ${FAKE_FUSED_CAMPAIGN_NVCC:-} ]]; then
  chmod u+w "$FAKE_FUSED_CAMPAIGN_NVCC"
  printf '%s\n' '# hostile identity drift' >>"$FAKE_FUSED_CAMPAIGN_NVCC"
  chmod 0555 "$FAKE_FUSED_CAMPAIGN_NVCC"
fi
