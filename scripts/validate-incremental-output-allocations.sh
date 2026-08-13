#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test service/incremental_output \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/service/incremental_output/incremental_output.whitebox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'incremental-output release C output is missing' >&2
  exit 1
fi

steady_body="$(awk '
  /^struct .*17push__token__into\(/ { copying = 1 }
  copying { print }
  copying && /^}$/ { exit }
' "$generated_c")"
if [ -z "$steady_body" ]; then
  printf '%s\n' 'incremental-output token function is missing' >&2
  exit 1
fi
if printf '%s\n' "$steady_body" |
  rg -q 'moonbit_make_.*array|moonbit_make_bytes|moonbit_add_string'; then
  printf '%s\n' \
    'incremental-output token path constructs a collection or string' >&2
  exit 1
fi

# Positive control: the test driver deliberately allocates its fixed
# destination. If this disappears, the generated-C scan is no longer capable
# of observing the allocation vocabulary it is meant to reject.
if ! rg -q 'moonbit_make_bytes\(8, 33\)' "$generated_c"; then
  printf '%s\n' 'incremental-output allocation positive control is ineffective' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux incremental-output allocation gate passed.'
