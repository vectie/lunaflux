#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test service/framed_wire service/request_admission \
  --target native --release --deny-warn

generated_c="_build/native/release/test/service/request_admission/request_admission.whitebox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'framed request-receipt release C output is missing' >&2
  exit 1
fi

extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 &&
      $0 ~ /^(struct|int|uint|void|double|moonbit_)[A-Za-z0-9_ *]*_M0/ &&
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

positive_body="$(extract_definition 'RequestFrameBuffer3new(')"
if [ -z "$positive_body" ] ||
  ! printf '%s\n' "$positive_body" | rg -q "$forbidden"; then
  printf '%s\n' 'framed request-receipt allocation positive control is ineffective' >&2
  exit 1
fi

contains_unexpected_allocation() {
  rg "$forbidden" |
    rg -v 'moonbit_malloc.*FramedWireError_2e(InvalidFrame|RequestReaderIncomplete|RequestReaderPoisoned|StaleFrame)' |
    rg -v 'moonbit_malloc.*RequestAdmissionError_2e(Invalid|ClockUnavailable)' |
    rg -q .
}

# Final canonical parsing intentionally constructs immutable request values.
# This gate covers the fixed owner-resident prefix/incomplete append path up to
# that exact final RequestFrameBuffer::load boundary, plus receipt sampling and
# publication ordering. The constructor above is the positive control.
for symbol in \
  'IncrementalRequestReader13virtual__byte(' \
  'IncrementalRequestReader12virtual__u32(' \
  'IncrementalRequestReader17preflight__prefix(' \
  'IncrementalRequestReader14next__capacity(' \
  'IncrementalRequestReader4take(' \
  'capture__receipt__with__clock(' \
  'IncrementalRequestReceiver17preflight__append(' \
  'IncrementalRequestReceiver17append__validated(' \
  'IncrementalRequestReceiver19append__with__clock(' \
  'IncrementalRequestReceiver6append(' \
  'IncrementalRequestReceiver14next__capacity('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'framed request-receipt function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_unexpected_allocation; then
    printf 'framed request-receipt prefix path allocates: %s\n' "$symbol" >&2
    exit 1
  fi
done

reader_append="$(extract_definition 'IncrementalRequestReader6append(')"
reader_prefix="$(printf '%s\n' "$reader_append" | \
  awk '/RequestFrameBuffer4load/ { exit } { print }')"
if [ -z "$reader_prefix" ] ||
  printf '%s\n' "$reader_prefix" | contains_unexpected_allocation; then
  printf '%s\n' \
    'framed request-receipt prefix path allocates before canonical load' >&2
  exit 1
fi
if ! printf '%s\n' "$reader_append" | rg -q \
  'IncrementalRequestReader17preflight__prefix' ||
  ! printf '%s\n' "$reader_append" | rg -q 'RequestFrameBuffer4load'; then
  printf '%s\n' \
    'framed request reader lost prefix-preflight/final-load composition' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux framed request-receipt allocation gate passed.'
