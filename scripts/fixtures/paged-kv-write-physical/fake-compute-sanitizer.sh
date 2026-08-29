#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --version ]]; then
  printf '%s\n' 'NVIDIA Compute Sanitizer 2025.4.1.0 fake KV-write fixture'
  exit 0
fi
tool= log_file=
while (( $# > 0 )); do
  case $1 in
    --tool) tool=$2; shift 2 ;;
    --log-file) log_file=$2; shift 2 ;;
    --error-exitcode|--leak-check) shift 2 ;;
    --*) shift ;;
    *) break ;;
  esac
done
[[ $tool == memcheck || $tool == racecheck || $tool == initcheck ]]
case ${FAKE_KV_WRITE_MODE:-pass} in
  dirty) printf '%s\n' '========= ERROR SUMMARY: 1 error' >"$log_file" ;;
  trailing) printf '%s\n' '========= ERROR SUMMARY: 0 errors injected' >"$log_file" ;;
  *) printf '%s\n' '========= ERROR SUMMARY: 0 errors' >"$log_file" ;;
esac
cycles=${!#}; canary=379457; guards=pass; non_target=pass; key=1088
case ${FAKE_KV_WRITE_MODE:-pass} in
  wrong-canary) canary=0 ;;
  guard) guards=fail ;;
  sentinel) non_target=fail ;;
  silent) key=0 ;;
esac
printf '%s\n' "outcome=paged-kv-write-sm120-qualification-pass cycles=$cycles cases=6 launches=$((10#$cycles*12)) qualified_token_count=17 key_mutations=$key value_mutations=1088 observed_dispatch_canary=$canary prefill=pass decode=pass mixed=pass origin=pass page_tail=pass cross_page=pass multirow=pass full_grid=pass guards=$guards non_target=$non_target scalar_oracle=pass serial_cuda_oracle=pass device_uuid=GPU-fake-sm120-qualification device_pci=00000000:01:00.0 device_name_sha256=268f7ae0d581b571300341fdce54fb977486376b0813d5ad35ee69815504340d device_total_memory_bytes=17179869184 cuda_driver_version=13010 resources=context0,stream0,allocation0,module0,function0,device_bytes0,pending0,cleanup0 authority=qualification-only"
[[ ${FAKE_KV_WRITE_MODE:-pass} != stderr ]] || printf '%s\n' 'hostile stderr' >&2
