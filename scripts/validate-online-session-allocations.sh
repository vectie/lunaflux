#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon build tests/worker_service_e2e --target native --release --deny-warn

generated_c="_build/native/release/build/tests/worker_service_e2e/worker_service_e2e.c"

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

# The aggregate constructor deliberately allocates all owners and fixed
# storage before rooted worker preparation. The same extractor and predicate
# must observe that positive control before checking publication/steady paths.
positive_body="$(extract_definition 'CanonicalEventWriter3new(')"
if [ -z "$positive_body" ] ||
  ! printf '%s\n' "$positive_body" | contains_forbidden_allocation; then
  printf '%s\n' 'online-session allocation positive control is ineffective' >&2
  exit 1
fi

for symbol in \
  'publish__online__session(' \
  'publish__online__cleanup(' \
  'OnlineSession10has__event(' \
  'OnlineSession13event__length(' \
  'OnlineSession15copy__event__to(' \
  'OnlineSession10ack__event(' \
  'OnlineSession12begin__abort(' \
  'OnlineSession17progress__cleanup(' \
  'OnlineWorkerLease28progress__terminal__recovery(' \
  'OnlineWorkerLease27commit__prepared__admission(' \
  'Scheduler28commit__exclusive__admission(' \
  'PreparedMonotonicRead12read__status(' \
  'now__millis__status(' \
  'CanonicalEventWriter27preflight__online__capacity('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'online-session allocation function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_forbidden_allocation; then
    printf 'online-session steady/publication path allocates: %s\n' "$symbol" >&2
    exit 1
  fi
done

# The exact aggregate/session/cleanup shells must precede rooted authority.
prepare_body="$(extract_definition 'prepare__owned__session(')"
if [ -z "$prepare_body" ]; then
  printf '%s\n' 'online-session owned constructor is missing' >&2
  exit 1
fi
session_line="$(printf '%s\n' "$prepare_body" | rg -n 'OnlineSession\*\)moonbit_malloc' | head -n 1 | cut -d: -f1)"
cleanup_line="$(printf '%s\n' "$prepare_body" | rg -n 'FailedOnlineSessionPreparation\*\)moonbit_malloc' | head -n 1 | cut -d: -f1)"
outcome_line="$(printf '%s\n' "$prepare_body" | rg -n 'OnlineSessionPreparation\*\)moonbit_malloc' | head -n 1 | cut -d: -f1)"
rooted_line="$(printf '%s\n' "$prepare_body" | rg -n 'worker__service22prepare__owned__online' | head -n 1 | cut -d: -f1)"
if [ -z "$session_line" ] || [ -z "$cleanup_line" ] ||
  [ -z "$outcome_line" ] || [ -z "$rooted_line" ] ||
  [ "$session_line" -ge "$rooted_line" ] ||
  [ "$cleanup_line" -ge "$rooted_line" ] ||
  [ "$outcome_line" -ge "$rooted_line" ]; then
  printf '%s\n' 'online-session owners are not all allocated before rooted preparation' >&2
  exit 1
fi

owned_body="$(extract_definition 'prepare__owned__internal(')"
if [ -z "$owned_body" ]; then
  printf '%s\n' 'owned online internal constructor is missing' >&2
  exit 1
fi
admission_line="$(printf '%s\n' "$owned_body" | rg -n 'Scheduler29prepare__exclusive__admission' | head -n 1 | cut -d: -f1)"
clock_line="$(printf '%s\n' "$owned_body" | rg -n 'MonotonicClock13prepare__read' | head -n 1 | cut -d: -f1)"
lease_line="$(printf '%s\n' "$owned_body" | rg -n 'OnlineWorkerLease\*\)moonbit_malloc' | head -n 1 | cut -d: -f1)"
owned_outcome_line="$(printf '%s\n' "$owned_body" | rg -n 'OwnedWorkerServicePreparation\*\)moonbit_malloc' | head -n 1 | cut -d: -f1)"
owned_rooted_line="$(printf '%s\n' "$owned_body" | rg -n 'prepare__exchange__with__approved__roots' | head -n 1 | cut -d: -f1)"
if [ -z "$admission_line" ] || [ -z "$clock_line" ] ||
  [ -z "$lease_line" ] || [ -z "$owned_outcome_line" ] ||
  [ -z "$owned_rooted_line" ] ||
  [ "$admission_line" -ge "$owned_rooted_line" ] ||
  [ "$clock_line" -ge "$owned_rooted_line" ] ||
  [ "$lease_line" -ge "$owned_rooted_line" ] ||
  [ "$owned_outcome_line" -ge "$owned_rooted_line" ]; then
  printf '%s\n' 'online admission/clock/lease shells were not prepared before rooted activation' >&2
  exit 1
fi

exclusive_prepare_body="$(extract_definition 'Scheduler29prepare__exclusive__admission(')"
if [ -z "$exclusive_prepare_body" ] ||
  ! printf '%s\n' "$exclusive_prepare_body" | contains_forbidden_allocation; then
  printf '%s\n' 'exclusive admission pre-root allocation/order control is missing' >&2
  exit 1
fi
if ! printf '%s\n' "$exclusive_prepare_body" | rg -q -- '->\$[0-9]+ = 1;'; then
  printf '%s\n' 'exclusive admission does not publish its reservation after shell allocation' >&2
  exit 1
fi

# After the exact lower take_online succeeds, aggregate publication is only
# owner installation, lease admission, or already-preallocated result return.
suffix="$(printf '%s\n' "$prepare_body" | tail -n "+$(printf '%s\n' "$prepare_body" | rg -n 'take__prepared__online' | head -n 1 | cut -d: -f1)")"
if [ -z "$suffix" ] ||
  printf '%s\n' "$suffix" | contains_forbidden_allocation; then
  printf '%s\n' 'online-session post-transfer publication introduced allocation' >&2
  exit 1
fi

scripts/validate-worker-service-online-lease-allocations.sh
scripts/validate-framed-wire-event-writer-allocations.sh
printf '%s\n' 'LunaFlux online-session allocation gate passed.'
