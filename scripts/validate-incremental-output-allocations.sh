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

extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 && $0 ~ /^struct moonbit_result_/ {
      candidate = 1
      body = $0 ORS
      next
    }
    candidate {
      body = body $0 ORS
      if ($0 ~ /^\);$/) { candidate = 0; body = ""; next }
      if ($0 ~ /^\) \{$/) {
        copying = 1
        depth = 1
        printf "%s", body
        candidate = 0
        next
      }
    }
    copying {
      print
      opens = gsub(/\{/, "{")
      closes = gsub(/\}/, "}")
      depth += opens - closes
      if (depth == 0) exit
    }
  ' "$generated_c"
}

steady_body="$(extract_definition '17push__token__into(')"
if [ -z "$steady_body" ]; then
  printf '%s\n' 'incremental-output token function is missing' >&2
  exit 1
fi
copy_piece_body="$(extract_definition '11copy__piece(')"
tokenizer_copy_body="$(extract_definition '20copy__decoded__piece(')"
if [ -z "$copy_piece_body" ] || [ -z "$tokenizer_copy_body" ]; then
  printf '%s\n' 'incremental-output transitive copy functions are missing' >&2
  exit 1
fi
if printf '%s\n%s\n%s\n' \
  "$steady_body" "$copy_piece_body" "$tokenizer_copy_body" |
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
