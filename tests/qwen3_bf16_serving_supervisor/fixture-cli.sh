#!/bin/sh
set -eu

[ "$#" -eq 2 ] && [ "$1" = run ] || exit 2
printf '%s\n' \
  'LunaFlux native framed service' \
  'health: healthy' \
  'readiness: true' \
  'runtime_origin=luna+tcp://127.0.0.1:19001' \
  'control_origin=http://127.0.0.1:19002' \
  'runtime_protocol=native-framed-v1'
IFS= read -r command <&5
[ "$command" = LFD1DRN ] || exit 3
printf 'LFD1ACK\n' >&5
