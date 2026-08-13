#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test engine/device_worker_child \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/engine/device_worker_child/device_worker_child.whitebox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'device-worker-child release C output is missing' >&2
  exit 1
fi

extract_function() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    $0 ~ pattern { copying = 1 }
    copying { print }
    copying && /^}$/ { exit }
  ' "$generated_c"
}

steady_body="$(extract_function '^struct .*23serve__serialized__with')"
if [ -z "$steady_body" ]; then
  printf '%s\n' 'serialized steady-loop release function is missing' >&2
  exit 1
fi
if printf '%s\n' "$steady_body" |
  rg -q 'moonbit_malloc|moonbit_make_.*array|moonbit_add_string'; then
  printf '%s\n' 'serialized steady-loop success branch allocates' >&2
  exit 1
fi

if ! rg -q 'moonbit_make_int32_array\(257,' "$generated_c"; then
  printf '%s\n' 'device-worker-child allocation positive control is ineffective' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux device-worker-child allocation gate passed.'
