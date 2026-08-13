#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test scheduler/core \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/scheduler/core/core.whitebox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'scheduler cancellation release C output is missing' >&2
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

positive_body="$(extract_definition 'cancel__reservation__allocation__positive__control(')"
if [ -z "$positive_body" ] ||
  ! printf '%s\n' "$positive_body" | contains_forbidden_allocation; then
  printf '%s\n' 'scheduler cancellation allocation positive control is ineffective' >&2
  exit 1
fi

for symbol in \
  'reserve__cancel__exact(' \
  'take__reserved__generated__token(' \
  'commit__cancel__reservation(' \
  'abort__cancel__reservation(' \
  'require__no__cancel__reservation(' \
  'has__exact__terminal(' \
  'clear__cancel__reservation(' \
  'allocate__request__generation(' \
  'enqueue__terminal(' \
  'remove__waiting(' \
  'recycle__slot(' \
  'release__active__request('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'scheduler cancellation function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_forbidden_allocation; then
    printf 'scheduler cancellation path allocates: %s\n' "$symbol" >&2
    exit 1
  fi
done

scripts/validate-hot-path-allocations.sh
printf '%s\n' 'LunaFlux scheduler cancellation reservation allocation gate passed.'
