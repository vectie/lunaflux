#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon build tests/worker_service_e2e \
  --target native --release --deny-warn --warn-list +73
moon test engine/worker_service \
  --target native --release --deny-warn --warn-list +73
scripts/validate-hot-path-allocations.sh
scripts/validate-worker-process-exchange-allocations.sh

generated_c="_build/native/release/build/tests/worker_service_e2e/worker_service_e2e.c"
collection_allocation_pattern='moonbit_make_.*array|moonbit_make_bytes|moonbit_make_ref|moonbit_add_string|Bytes4make\(|FixedArray4make\('
strict_allocation_pattern="moonbit_malloc|${collection_allocation_pattern}"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'worker-service release C output is missing' >&2
  exit 1
fi

extract_definition() {
  local source_file="$1"
  local pattern="$2"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 && $0 ~ /^[A-Za-z_]/ {
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
  ' "$source_file"
}

hot_body=""
for symbol in \
  'WorkerService8progress(' \
  'WorkerService16commit__received(' \
  'worker__service18bridge__completion(' \
  'worker__service27submit__bridge__after__open(' \
  'worker__service23append__bridge__prefill(' \
  'worker__service22append__bridge__decode(' \
  'Scheduler19complete__submitted(' \
  'Scheduler8complete(' \
  'Scheduler21preflight__completion(' \
  'Scheduler25request__stops__on__token(' \
  'LunaRequestStopTokenRetentionSlot15is__stop__token(' \
  'LunaRequestSemanticStorage19stop__token__status('; do
  body="$(extract_definition "$generated_c" "$symbol")"
  if [ -z "$body" ]; then
    printf 'worker-service allocation function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  hot_body="${hot_body}${body}"
done

if printf '%s\n' "$hot_body" |
  rg -q "$collection_allocation_pattern"; then
  printf '%s\n' 'worker-service progress constructs a collection, ref, or string' >&2
  exit 1
fi
if printf '%s\n' "$hot_body" |
  rg 'moonbit_malloc' |
  rg -q -v 'Error|Failure'; then
  printf '%s\n' 'worker-service progress contains a non-error heap allocation' >&2
  exit 1
fi

semantic_membership_body=""
for symbol in \
  'Scheduler25request__stops__on__token(' \
  'LunaRequestStopTokenRetentionSlot8is__live(' \
  'LunaRequestStopTokenRetentionSlot15is__stop__token(' \
  'LunaRequestSemanticStorage19stop__token__status('; do
  body="$(extract_definition "$generated_c" "$symbol")"
  if [ -z "$body" ]; then
    printf 'worker semantic-membership function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  semantic_membership_body="${semantic_membership_body}${body}"
done
if printf '%s\n' "$semantic_membership_body" |
  rg "$strict_allocation_pattern" |
  rg -v \
    'moonbit_malloc\(sizeof\(struct _M0DTPC15error5Error63vectie_2flunaflux_2fscheduler_2fcore_2eSchedulerError_2eInvalid\)\)' |
  rg -q .; then
  printf '%s\n' \
    'worker semantic membership contains a non-typed-error allocation' >&2
  exit 1
fi

request_stops_source="$(awk '
  /fn Scheduler::request_stops_on_token\(/ { copying = 1 }
  copying {
    print
    opens += gsub(/\{/, "{"); closes += gsub(/\}/, "}")
    if (opens > 0 && opens == closes) exit
  }
' scheduler/core/completion_preflight.mbt)"
if [ -z "$request_stops_source" ] ||
  ! printf '%s\n' "$request_stops_source" | rg -q \
    'request_stop_tokens\[slot\]' ||
  ! printf '%s\n' "$request_stops_source" | rg -q \
    'is_stop_token\(token\)' ||
  ! printf '%s\n' "$request_stops_source" | rg -q 'is_stale\(\)' ||
  printf '%s\n' "$request_stops_source" | rg -q \
    'LunaRequestSemanticView|StopConditions|CachePolicy|\.stops\(|\.cache\('; then
  printf '%s\n' 'worker semantic-membership dependency shape drifted' >&2
  exit 1
fi

whitebox_c="_build/native/release/test/engine/worker_service/worker_service.whitebox_test.c"
positive_body="$(extract_definition \
  "$whitebox_c" \
  'owned__preparation__allocation__positive__control(')"
if [ -z "$positive_body" ] || ! printf '%s\n' "$positive_body" |
  rg -q "$strict_allocation_pattern"; then
  printf '%s\n' 'worker-service allocation positive control is ineffective' >&2
  exit 1
fi

# The scalar capacity branch is the strict proof point for backpressure: unlike
# semantic error paths above, no constructor name is exempted from this check.
capacity_body="$(extract_definition \
  "$generated_c" \
  'Scheduler28completion__capacity__commit(')"
if [ -z "$capacity_body" ]; then
  printf '%s\n' 'scheduler scalar completion-capacity function is missing' >&2
  exit 1
fi
if printf '%s\n' "$capacity_body" |
  rg -q "$strict_allocation_pattern"; then
  printf '%s\n' 'scheduler scalar completion backpressure branch allocates' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux worker-service progress allocation gate passed.'
