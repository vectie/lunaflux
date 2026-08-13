#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test service/framed_wire \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/service/framed_wire/framed_wire.whitebox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'framed-wire release C output is missing' >&2
  exit 1
fi

extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 &&
      $0 ~ /^(struct|int|uint|void|moonbit_)[A-Za-z0-9_ *]*_M0/ &&
      $0 ~ /\($/ {
      candidate = 1; body = $0 ORS; next
    }
    candidate {
      body = body $0 ORS
      if ($0 ~ /^\);$/) { candidate = 0; body = ""; next }
      if ($0 ~ /^\) \{$/) {
        copying = 1; depth = 1; printf "%s", body; candidate = 0; next
      }
    }
    copying {
      print
      opens = gsub(/\{/, "{"); closes = gsub(/\}/, "}")
      depth += opens - closes
      if (depth == 0) exit
    }
  ' "$generated_c"
}

forbidden='moonbit_malloc|moonbit_make_|Bytes4make|moonbit_add_string'

contains_forbidden_allocation() {
  rg "$forbidden" | rg -v 'Error|Failure' | rg -q .
}

# The writer constructor intentionally allocates its owner and fixed buffers.
# It is extracted with the exact same predicate used for the steady path, so a
# compiler/output-shape change cannot make the scan pass vacuously.
positive_body="$(extract_definition 'CanonicalEventWriter3new(')"
if [ -z "$positive_body" ] ||
  ! printf '%s\n' "$positive_body" | contains_forbidden_allocation; then
  printf '%s\n' 'framed-wire allocation positive control is ineffective' >&2
  exit 1
fi

for symbol in \
  'prepare__token__bytes(' \
  'CanonicalEventWriter15require__credit(' \
  'valid__utf8__range(' \
  'prepare__event__header(' \
  'CanonicalEventWriter7publish(' \
  'frame__checksum(' \
  'copy__bytes('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'framed-wire direct event function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_forbidden_allocation; then
    printf 'framed-wire direct Token path allocates: %s\n' "$symbol" >&2
    exit 1
  fi
done

printf '%s\n' 'LunaFlux framed-wire direct event allocation gate passed.'
