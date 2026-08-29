#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --version ]]; then
  printf '%s\n' 'NVIDIA Compute Sanitizer 2026.1.0 fake LunaTile fixture'
  exit 0
fi

tool=
log_file=
while (( $# > 0 )); do
  case $1 in
    --tool) tool=$2; shift 2 ;;
    --log-file) log_file=$2; shift 2 ;;
    --leak-check|--error-exitcode) shift 2 ;;
    --*) shift ;;
    *) break ;;
  esac
done
[[ $tool == memcheck || $tool == racecheck ]]
[[ $log_file == /* ]]

mode=${FAKE_LUNATILE_CAMPAIGN_MODE:-pass}
case $mode in
  dirty-sanitizer) printf '%s\n' '========= ERROR SUMMARY: 1 error' >"$log_file" ;;
  extra-summary)
    printf '%s\n' \
      '========= ERROR SUMMARY: 0 errors' \
      '========= ERROR SUMMARY: 0 errors' >"$log_file"
    ;;
  *) printf '%s\n' '========= ERROR SUMMARY: 0 errors' >"$log_file" ;;
esac

cycles=${!#}
serial=120d858e0275f15a14c4a60ccb66c99c07126e887d7ec2afad26b9fb942adbef
parallel=fd3b1b0cbd64e7717f053fd5adbb0177ae7f6e406718a16da4ca8c61c1ce6362
candidate=d35f611375f6f8ada2d412eef93d31eef377a241eaca22e846a853b070486fbf
if [[ $mode == identity-mismatch ]]; then
  candidate=0000000000000000000000000000000000000000000000000000000000000000
fi
printf '%s\n' \
  "outcome=lunatile-parallel-sm120-qualification-pass cycles=$cycles launches=$((cycles * 2)) bytes_per_launch=256 serial_source_sha256=$serial parallel_source_sha256=$parallel parallel_candidate_sha256=$candidate target=sm_120 resources=context0,stream0,allocation0,module0,function0,device_bytes0,pending0,cleanup0 manifest_bindable=false promotion_authority=absent"
if [[ $mode == stderr ]]; then
  printf '%s\n' 'hostile LunaTile sanitizer stderr' >&2
fi
