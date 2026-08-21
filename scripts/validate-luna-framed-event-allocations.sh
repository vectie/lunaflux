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

# The Workspace constructor intentionally allocates its owner, byte backing,
# and one semantic-view slot. The same extractor/predicate is the positive
# control for the warmed cooperative path.
positive_body="$(extract_definition 'LunaFramedEventWorkspace3new(')"
if [ -z "$positive_body" ] ||
  ! printf '%s\n' "$positive_body" | contains_forbidden_allocation; then
  printf '%s\n' 'Luna framed-event allocation positive control is ineffective' >&2
  exit 1
fi

for symbol in \
  'LunaFramedEventWorkspace5begin(' \
  'LunaFramedEventWorkspace16reset__operation(' \
  'LunaFramedEventWorkspace11event__view(' \
  'LunaFramedEventWorkspace19authenticate__event(' \
  'LunaFramedEventWorkspace4fail(' \
  'LunaFramedEventWorkspace14failure__value(' \
  'LunaFramedEventWorkspace13progress__one(' \
  'LunaFramedEventWorkspace15progress__setup(' \
  'LunaFramedEventWorkspace14progress__copy(' \
  'LunaFramedEventWorkspace18progress__checksum(' \
  'LunaFramedEventWorkspace15setup__accepted(' \
  'LunaFramedEventWorkspace12setup__token(' \
  'LunaFramedEventWorkspace12setup__usage(' \
  'LunaFramedEventWorkspace16setup__completed(' \
  'LunaFramedEventWorkspace13setup__failed(' \
  'LunaFramedEventWorkspace24require__event__capacity(' \
  'LunaFramedEventWork18require__workspace(' \
  'LunaFramedEventWork5state(' \
  'LunaFramedEventWork7failure(' \
  'LunaFramedEventWork5abort(' \
  'LunaFramedEventWork8progress(' \
  'LunaFramedEventWork10take__view(' \
  'LunaFramedEventView18require__workspace(' \
  'LunaFramedEventView15copy__chunk__to(' \
  'LunaFramedEventView7release(' \
  'LunaEventView4kind(' \
  'LunaEventView8accepted(' \
  'LunaEventView5token(' \
  'LunaEventView5usage(' \
  'LunaEventView9completed(' \
  'LunaEventView6failed(' \
  'LunaAcceptedEventView17effective__limits(' \
  'LunaAcceptedEventView25content__digest__byte__at(' \
  'LunaAcceptedEventView22plan__digest__byte__at(' \
  'LunaTokenEventView15delta__byte__at(' \
  'LunaCompletedEventView22final__delta__byte__at(' \
  'LunaFailedEventView14code__byte__at(' \
  'luna__event__prepare__header__fields(' \
  'write__usage__scalars(' \
  'valid__usage__scalars('; do
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

if ! rg -q 'semantic_views: Array::new\(capacity=1\)' \
    service/framed_wire/luna_event_work.mbt ||
  ! rg -q 'self\.semantic_views\.clear\(\)' \
    service/framed_wire/luna_event_work.mbt ||
  ! rg -q 'length > workspace\.budget\.work_units\(\)' \
    service/framed_wire/luna_event_work_view.mbt ||
  ! rg -q 'self\.storage\[destination\] = byte' \
    service/framed_wire/luna_event_work_progress.mbt ||
  ! rg -q 'self\.checksum = .*self\.storage\[self\.cursor\]\.to_uint\(\)' \
    service/framed_wire/luna_event_work_progress.mbt; then
  printf '%s\n' \
    'Luna framed-event fixed-slot or one-byte work source shape drifted' >&2
  exit 1
fi

if rg -n '^pub fn LunaFramedEvent(Workspace|Work|View)::(ack|retire|take_event)\(' \
  service/framed_wire/pkg.generated.mbti; then
  printf '%s\n' 'cooperative framed event acquired semantic ACK authority' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux cooperative semantic-to-framed event allocation gate passed.'
