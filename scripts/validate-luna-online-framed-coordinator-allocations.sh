#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon build tests/worker_service_e2e \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/build/tests/worker_service_e2e/worker_service_e2e.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'Luna online framed coordinator release C output is missing' >&2
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
  # Only MoonBit's typed exception envelope is language plumbing. No broad
  # Error/Failure-name filter is permitted in this strict path.
  rg "$forbidden" |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0DTPC15error5Error' |
    rg -q .
}

# Startup construction deliberately allocates the aggregate and all fixed
# queue/capability slots. This proves the extractor and predicate can see a
# real allocation before they are used on warmed transport/event paths.
positive_body="$(extract_definition 'new__luna__online__framed__coordinator(')"
if [ -z "$positive_body" ] ||
  ! printf '%s\n' "$positive_body" | contains_forbidden_allocation; then
  printf '%s\n' 'coordinator allocation positive control is ineffective' >&2
  exit 1
fi

# These functions cover accepted ingress, exact tail accounting, framed event
# copy/confirmation, semantic ACK, rejection credit, and disconnect/drain.
# Preparation Ready assembly has one separately documented constant shell and
# is intentionally not misrepresented as a strict zero-allocation transition.
for symbol in \
  'LunaOnlineFramedIngress4kind(' \
  'luna__framed__ingress(' \
  'luna__framed__ingress__consumed(' \
  'LunaOnlineFramedCoordinator19offer__luna__framed(' \
  'LunaOnlineFramedCoordinator28observe__receipt__completion(' \
  'LunaOnlineFramedCoordinator22current__receipt__work(' \
  'LunaOnlineFramedCoordinator30copy__framed__event__chunk__to(' \
  'LunaOnlineFramedCoordinator24framed__event__remaining(' \
  'LunaOnlineFramedEventOffer14require__owner(' \
  'LunaOnlineFramedEventOffer6length(' \
  'LunaOnlineFramedEventOffer7confirm(' \
  'LunaOnlineFramedCoordinator22start__semantic__event(' \
  'LunaOnlineFramedCoordinator25progress__outbound__event(' \
  'LunaOnlineFramedCoordinator18finish__event__ack(' \
  'LunaOnlineFramedCoordinator18progress__instance(' \
  'LunaOnlineFramedCoordinator8progress(' \
  'LunaOnlineFramedCoordinator22enforce__output__stall(' \
  'LunaOnlineFramedCoordinator14stall__expired(' \
  'LunaOnlineFramedCoordinator28release__outbound__authority(' \
  'LunaOnlineFramedCoordinator15take__rejection(' \
  'LunaOnlineFramedRejectionCredit14require__owner(' \
  'LunaOnlineFramedRejectionCredit4rule(' \
  'LunaOnlineFramedRejectionCredit3ack(' \
  'LunaOnlineFramedCoordinator10disconnect(' \
  'LunaOnlineFramedCoordinator12begin__drain(' \
  'LunaOnlineFramedCoordinator35progress__off__reactor__maintenance('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'coordinator allocation function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_forbidden_allocation; then
    printf 'coordinator warmed path allocates: %s\n' "$symbol" >&2
    exit 1
  fi
done

# Release compilation inlines these two authenticated scalar projections into
# their E2E callers. Pin their exact source bodies while their transitive
# require_owner helper remains in the strict generated-C set above.
if ! rg -q --pcre2 -U \
    'pub fn LunaOnlineFramedIngress::consumed_bytes[^{]*\{\s*self\.consumed\s*\}' \
    service/online_session/coordinator_ingress.mbt ||
  ! rg -q --pcre2 -U \
    'pub fn LunaOnlineFramedRejectionCredit::sequence[^{]*\{\s*self\.require_owner\(\)\.rejection_sequence\s*\}' \
    service/online_session/coordinator_progress.mbt; then
  printf '%s\n' \
    'inlined coordinator scalar projection stopped being an authenticated field read' >&2
  exit 1
fi

coordinator_fields="$(sed -n \
  '/^pub struct LunaOnlineFramedCoordinator {/,/^}/p' \
  service/online_session/coordinator_types.mbt)"
if [ -z "$coordinator_fields" ] ||
  printf '%s\n' "$coordinator_fields" | rg -q '\?' ||
  ! printf '%s\n' "$coordinator_fields" |
    rg -q 'priv work_slots : FixedArray\[' ||
  ! rg -q 'work_slots: FixedArray::makei\(lane_count, _ => Array::new\(capacity=1\)\)' \
    service/online_session/coordinator_prepare.mbt ||
  ! rg -q 'event_credits: Array::new\(capacity=1\)' \
    service/online_session/coordinator_prepare.mbt ||
  ! rg -q 'outbound_works: Array::new\(capacity=1\)' \
    service/online_session/coordinator_prepare.mbt ||
  ! rg -q 'outbound_views: Array::new\(capacity=1\)' \
    service/online_session/coordinator_prepare.mbt; then
  printf '%s\n' \
    'coordinator warmed owner acquired Option boxing or dynamic authority slots' >&2
  exit 1
fi

if rg -n \
    'GenerateRequest|ReceivedRequest|RequestFrameBuffer|EventFrameBuffer|LunaFramedEventAdapter|pub async fn|Tcp|Socket|Listener' \
    service/online_session/coordinator_types.mbt \
    service/online_session/coordinator_prepare.mbt \
    service/online_session/coordinator_ingress.mbt \
    service/online_session/coordinator_progress.mbt \
    service/online_session/coordinator_events.mbt \
    service/online_session/coordinator_lifecycle.mbt; then
  printf '%s\n' \
    'coordinator reintroduced object materialization, compatibility framing, or network IO' >&2
  exit 1
fi

if ! rg -q --pcre2 -U \
    'try! owner\.outbound_views\[0\]\.release\(\)[\s\S]*owner\.outbound_views\.clear\(\)[\s\S]*owner\.event_ack_pending = true[\s\S]*if owner\.event_kind == 3 \{[\s\S]*owner\.maintenance_kind = 3[\s\S]*LunaOnlineFramedCoordinatorMaintenanceRequired' \
    service/online_session/coordinator_events.mbt ||
  ! rg -q --pcre2 -U \
    'if self\.maintenance_kind == 3 \{[\s\S]*return self\.finish_event_ack\(\)' \
    service/online_session/coordinator_lifecycle.mbt ||
  ! rg -q --pcre2 -U \
    'if self\.tickets\.length\(\) == 1 \{[\s\S]*self\.begin_output_disconnect\(\)[\s\S]*\} else \{[\s\S]*self\.release_outbound_authority\(\)' \
    service/online_session/coordinator_lifecycle.mbt ||
  ! rg -q --pcre2 -U \
    'fn LunaOnlineFramedCoordinator::progress_preparation[\s\S]*if self\.tickets\.length\(\) == 1 \{[\s\S]*return match self\.pool\.progress\(\)[\s\S]*\}[\s\S]*if self\.prepared\.length\(\) == 1' \
    service/online_session/coordinator_progress.mbt ||
  ! rg -q --pcre2 -U \
    'if self\.disconnect_requested \{[\s\S]*if self\.tickets\.is_empty\(\)[\s\S]*return self\.progress_preparation\(\)[\s\S]*return LunaOnlineFramedCoordinatorMaintenanceRequired' \
    service/online_session/coordinator_progress.mbt ||
  ! rg -q --pcre2 -U \
    'LunaOnlineInstanceClosed => \{[\s\S]*self\.drain_requested = true[\s\S]*self\.disconnect_requested = true[\s\S]*self\.pool\.begin_drain\(\)[\s\S]*LunaOnlineFramedCoordinatorDraining' \
    service/online_session/coordinator_lifecycle.mbt; then
  printf '%s\n' \
    'coordinator framing, FIFO publication, or disconnect authority order drifted' >&2
  exit 1
fi

printf '%s\n' \
  'LunaFlux online framed coordinator allocation and authority gate passed.'
