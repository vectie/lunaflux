#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Reuse the stronger payload-backing proof. The endpoint proof below is only
# for warmed synchronous ownership/scalar helpers; async coroutine, timer, and
# operating-system socket implementations may allocate and are out of scope.
scripts/validate-luna-online-tcp-buffer-allocations.sh
moon build tests/worker_service_e2e \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/build/tests/worker_service_e2e/worker_service_e2e.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'Luna online TCP endpoint release C output is missing' >&2
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

forbidden='moonbit_malloc|moonbit_make_|Bytes4make|Bytes5makei|memcpy|memmove|blit'

contains_forbidden_allocation_or_copy() {
  rg "$forbidden" |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0DTPC15error5Error' |
    rg -q .
}

extract_source_definition() {
  local pattern="$1"
  local source="$2"
  awk -v pattern="$pattern" '
    !copying && index($0, pattern) > 0 { copying = 1 }
    copying {
      print
      opens = gsub(/\{/, "{"); closes = gsub(/\}/, "}")
      depth += opens - closes
      if (opens > 0) { entered = 1 }
      if (entered && depth == 0) exit
    }
  ' "$source"
}

# Positive control: the endpoint-owned scratch has exactly one startup Bytes
# backing, already pinned more fully by the reused buffer gate.
positive_body="$(extract_definition 'LunaOnlineTcpOutputScratch3new(')"
if [ -z "$positive_body" ] ||
  ! printf '%s\n' "$positive_body" | rg -q 'moonbit_make_bytes\('; then
  printf '%s\n' 'online TCP endpoint allocation positive control is ineffective' >&2
  exit 1
fi

for symbol in \
  'LunaOnlineTcpActivity14enter__reactor(' \
  'LunaOnlineTcpActivity14leave__reactor(' \
  'LunaOnlineTcpActivity18enter__maintenance(' \
  'LunaOnlineTcpActivity18leave__maintenance(' \
  'LunaOnlineTcpMaintenanceHandoff16require__pending(' \
  'LunaOnlineTcpMaintenanceHandoff3arm(' \
  'LunaOnlineTcpMaintenanceHandoff5clear(' \
  'LunaOnlineTcpEndpoint11coordinator(' \
  'LunaOnlineTcpEndpoint14latch__failure(' \
  'LunaOnlineTcpEndpoint24release__revoked__output(' \
  'LunaOnlineTcpEndpoint17begin__disconnect(' \
  'LunaOnlineTcpEndpoint18capture__rejection(' \
  'LunaOnlineTcpEndpoint11offer__tail(' \
  'LunaOnlineTcpEndpoint24bounded__transport__wait(' \
  'luna__online__tcp__safe__wait(' \
  'LunaOnlineTcpEndpoint26publish__closed__if__ready(' \
  'LunaOnlineTcpEndpoint35progress__off__reactor__maintenance(' \
  'LunaOnlineTcpEndpoint10disconnect(' \
  'LunaOnlineTcpEndpoint5state(' \
  'LunaOnlineTcpEndpoint19rejection__sequence(' \
  'LunaOnlineTcpEndpoint15rejection__rule(' \
  'LunaOnlineFramedCoordinator34transport__wait__remaining__millis(' \
  'LunaOnlineFramedCoordinator24stall__remaining__millis(' \
  'LunaOnlineFramedCoordinator22current__receipt__work(' \
  'LunaRequestPreparationWork40luna__framed__receipt__remaining__millis('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'online TCP endpoint allocation function is missing: %s\n' \
      "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_forbidden_allocation_or_copy; then
    printf 'online TCP warmed synchronous helper allocates or copies: %s\n' \
      "$symbol" >&2
    exit 1
  fi
done

# The exact wait must be sampled after output publication and directly before
# each async body operation. Expiry starts no read/write and confirmation must
# precede Flight release.
if ! rg -q --pcre2 -U \
    'let wait = self\.bounded_transport_wait\([\s\S]*if wait <= 0 \{[\s\S]*return LunaOnlineTcpCleanupRequired[\s\S]*let received = @async\.with_timeout_opt' \
    service/online_tcp/endpoint_ingress.mbt ||
  ! rg -q --pcre2 -U \
    'self\.offers\.push\(offer\)[\s\S]*self\.flights\.push\(flight\)[\s\S]*let wait = self\.bounded_transport_wait[\s\S]*if wait <= 0 \{[\s\S]*return LunaOnlineTcpCleanupRequired[\s\S]*let written = @async\.with_timeout_opt' \
    service/online_tcp/endpoint_output.mbt ||
  ! rg -q --pcre2 -U \
    'let progress = self\.offers\[0\]\.confirm\(length=count\)[\s\S]*try! self\.flights\[0\]\.release\(\)[\s\S]*self\.flights\.clear\(\)[\s\S]*self\.offers\.clear\(\)' \
    service/online_tcp/endpoint_output.mbt; then
  printf '%s\n' 'online TCP exact wait or confirm-release order drifted' >&2
  exit 1
fi

# A partial receipt is the only queued-work state that authorizes another body
# read. Terminal Ready/Failed work pinned behind an active ticket is semantic
# progress, not transport idleness, and tail backpressure must honor the exact
# reactor transition quantum before the coordinator is sampled again.
if ! rg -q --pcre2 -U \
    'if self\.tickets\.length\(\) == 1 \{[\s\S]*return match self\.pool\.progress\(\) \{[\s\S]*LunaRequestPreparationPoolIdle => LunaOnlineFramedCoordinatorAdvanced[\s\S]*LunaRequestPreparationPoolAwaitingInput =>[[:space:]]*LunaOnlineFramedCoordinatorAwaitingInput[\s\S]*LunaRequestPreparationPoolAdvanced => LunaOnlineFramedCoordinatorAdvanced' \
    service/online_session/coordinator_progress.mbt ||
  ! rg -q --pcre2 -U \
    'LunaOnlineFramedCoordinatorAwaitingInput => \{[\s\S]*if self\.tail_length > 0 \{[\s\S]*continue[\s\S]*return self\.read_on_reactor\(\)' \
    service/online_tcp/endpoint_reactor.mbt ||
  ! rg -q --pcre2 -U \
    'LunaOnlineTcpBackpressured =>[\s\S]*if transitions >= self\.limits\.reactor_transition_budget \{[\s\S]*return LunaOnlineTcpBackpressured' \
    service/online_tcp/endpoint_reactor.mbt; then
  printf '%s\n' \
    'online TCP awaiting-input, pinned-terminal, or transition-quantum mapping drifted' >&2
  exit 1
fi

if rg -n \
    'read_some|read_exactly|run_forever|with_task_group|spawn|Writer::write|\.write\(|pub fn .*-> .*(Tcp|TcpServer|Bytes|FixedArray|LunaOnlineFramedCoordinator|LunaOnlineFramedEventOffer|LunaOnlineTcpOutput)' \
    service/online_tcp --glob '*.mbt' --glob '!*_test.mbt' --glob '!*_wbtest.mbt'; then
  printf '%s\n' 'online TCP endpoint escaped its one-shot bounded authority shell' >&2
  exit 1
fi

offreact_body="$(extract_source_definition \
  'pub fn LunaOnlineTcpEndpoint::progress_off_reactor_maintenance' \
  service/online_tcp/endpoint_reactor.mbt)"
if [ -z "$offreact_body" ] ||
  printf '%s\n' "$offreact_body" |
    rg -q '@async|@socket|servers|connections|close_transport_on_reactor|\.close\(|\.read\(|\.write'; then
  printf '%s\n' \
    'online TCP off-reactor maintenance acquired reactor or socket authority' >&2
  exit 1
fi

if [ "$(rg -c '\.accept\(\)' service/online_tcp/endpoint_ingress.mbt)" -ne 1 ] ||
  [ "$(rg -c '\.read\(' service/online_tcp/endpoint_ingress.mbt)" -ne 1 ] ||
  [ "$(rg -c '\.write_once\(' service/online_tcp/endpoint_output.mbt)" -ne 2 ] ||
  [ "$(rg -c 'connection\.write_once\(' service/online_tcp/endpoint_output.mbt)" -ne 1 ]; then
  printf '%s\n' 'online TCP endpoint socket operation count drifted' >&2
  exit 1
fi

printf '%s\n' \
  'LunaFlux online TCP endpoint synchronous allocation and authority gate passed.'
