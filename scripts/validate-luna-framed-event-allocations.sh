#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test service/framed_wire \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/service/framed_wire/framed_wire.whitebox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'Luna framed-event release C output is missing' >&2
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
  # Typed exception envelopes are language error plumbing. Nothing else,
  # including Error/Failure-named production code, is filtered from evidence.
  rg "$forbidden" |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0DTPC15error5Error' |
    rg -q .
}

# The adapter constructor intentionally allocates its owner and fixed storage.
# The same extractor/predicate is the positive control for the warm path.
positive_body="$(extract_definition 'LunaFramedEventAdapter3new(')"
if [ -z "$positive_body" ] ||
  ! printf '%s\n' "$positive_body" | contains_forbidden_allocation; then
  printf '%s\n' 'Luna framed-event allocation positive control is ineffective' >&2
  exit 1
fi

for symbol in \
  'LunaFramedEventAdapter5stage(' \
  'LunaFramedEventAdapter15stage__accepted(' \
  'LunaFramedEventAdapter12stage__token(' \
  'LunaFramedEventAdapter12stage__usage(' \
  'LunaFramedEventAdapter16stage__completed(' \
  'LunaFramedEventAdapter13stage__failed(' \
  'LunaFramedEventAdapter15require__credit(' \
  'LunaFramedEventAdapter26require__payload__capacity(' \
  'LunaFramedEventAdapter7publish(' \
  'LunaFramedEventAdapter6length(' \
  'LunaFramedEventAdapter8copy__to(' \
  'LunaFramedEventAdapter7release(' \
  'LunaEventView4kind(' \
  'LunaEventView8accepted(' \
  'LunaEventView5token(' \
  'LunaEventView5usage(' \
  'LunaEventView9completed(' \
  'LunaEventView6failed(' \
  'LunaAcceptedEventView15model__identity(' \
  'LunaAcceptedEventView17effective__limits(' \
  'LunaTokenEventView15copy__delta__to(' \
  'LunaCompletedEventView22copy__final__delta__to(' \
  'LunaFailedEventView14copy__code__to(' \
  'write__digest__ascii(' \
  'write__usage__scalars(' \
  'prepare__event__header(' \
  'frame__checksum(' \
  'copy__bytes('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'Luna framed-event function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_forbidden_allocation; then
    printf 'Luna framed-event warm path allocates: %s\n' "$symbol" >&2
    exit 1
  fi
done

if rg -n '^pub fn LunaFramedEventAdapter::(ack|retire|take_event)\(' \
  service/framed_wire/pkg.generated.mbti; then
  printf '%s\n' 'framed adapter acquired semantic ACK authority' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux semantic-to-framed event allocation gate passed.'
