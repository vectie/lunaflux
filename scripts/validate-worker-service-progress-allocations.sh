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
whitebox_c="_build/native/release/test/engine/worker_service/worker_service.whitebox_test.c"
collection_allocation_pattern='moonbit_make_.*array|moonbit_make_bytes|moonbit_make_ref|moonbit_add_string|Bytes4make\(|FixedArray4make\('
strict_allocation_pattern="moonbit_malloc|${collection_allocation_pattern}"
worker_service_error_allocation_pattern='moonbit_malloc\(sizeof\(struct _M0DTPC15error5Error(75vectie_2flunaflux_2fengine_2fworker__service_2eWorkerServiceError_2eInvalid|77vectie_2flunaflux_2fengine_2fworker__service_2eWorkerServiceError_2eScheduler|84vectie_2flunaflux_2fengine_2fworker__service_2eWorkerServiceError_2ePhysicalTransport)\)\)'
scheduler_error_allocation_pattern='moonbit_malloc\(sizeof\(struct _M0DTPC15error5Error(63vectie_2flunaflux_2fscheduler_2fcore_2eSchedulerError_2eInvalid|66vectie_2flunaflux_2fscheduler_2fcore_2eSchedulerError_2eBlockTable|69vectie_2flunaflux_2fscheduler_2fcore_2eSchedulerError_2ePageAllocator|70vectie_2flunaflux_2fscheduler_2fcore_2eSchedulerError_2eWorkerProtocol)\)\)'
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'worker-service release C output is missing' >&2
  exit 1
fi
if [ ! -f "$whitebox_c" ]; then
  printf '%s\n' 'worker-service release white-box C output is missing' >&2
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
  'WorkerService14progress__impl(' \
  'WorkerService20start__pending__plan(' \
  'WorkerService20stage__pending__plan(' \
  'WorkerService23raw__outstanding__count(' \
  'WorkerService25finish__recovered__flight(' \
  'WorkerService25recover__pending__failure(' \
  'WorkerService16commit__received(' \
  'WorkerPhysicalTransport23has__exchange__capacity(' \
  'WorkerPhysicalTransport15begin__exchange(' \
  'WorkerPhysicalTransport18progress__exchange(' \
  'WorkerPhysicalTransport15received__frame(' \
  'WorkerPhysicalTransport18accept__completion(' \
  'WorkerPhysicalTransport16retire__accepted(' \
  'Scheduler11build__next(' \
  'Scheduler12select__rows(' \
  'Scheduler26preempt__for__aged__waiter(' \
  'Scheduler26aged__requires__preemption(' \
  'Scheduler18preemption__victim(' \
  'Scheduler31release__active__for__recompute(' \
  'Scheduler15oldest__prefill(' \
  'Scheduler18prefill__token__at(' \
  'Scheduler23store__generated__token(' \
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

worker_service_source="$(printf '%s\n' \
  engine/worker_service/*.mbt | sort | xargs awk 'FNR == 1 { print "" } { print }')"
if ! printf '%s\n' "$worker_service_source" | rg -q \
  'pending_plan_sequence : UInt64' ||
  printf '%s\n' "$worker_service_source" | rg -q \
    'pending_plan[^\n]*:.*\?|Option\[[^]]*(SubmittedSchedulePlan|SchedulePlan|Plan)' ||
  printf '%s\n' "$worker_service_source" | rg -q \
    'pending_plan[^\n]*:.*(SubmittedSchedulePlan|SchedulePlan)'; then
  printf '%s\n' \
    'worker-service pending plan is not an exact scalar owner' >&2
  exit 1
fi

worker_service_interface="engine/worker_service/pkg.generated.mbti"
for signature in \
  'pub fn WorkerService::begin_recovery_maintenance(Self, @worker_protocol.WorkerFailure) -> Unit raise WorkerServiceError' \
  'pub fn WorkerService::begin_shutdown_maintenance(Self) -> Unit raise WorkerServiceError' \
  'pub fn WorkerService::begin_restart_cleanup_maintenance(Self) -> Unit raise WorkerServiceError' \
  'pub fn WorkerService::progress_maintenance(Self) -> LunaWorkerServiceMaintenanceProgress raise WorkerServiceError' \
  'pub fn WorkerService::maintenance_remaining_millis(Self) -> Int raise WorkerServiceError' \
  'pub fn OnlineWorkerLease::begin_recovery_maintenance(Self, @worker_protocol.WorkerFailure) -> Unit raise WorkerServiceError' \
  'pub fn OnlineWorkerLease::begin_shutdown_maintenance(Self) -> Unit raise WorkerServiceError' \
  'pub fn OnlineWorkerLease::begin_restart_cleanup_maintenance(Self) -> Unit raise WorkerServiceError' \
  'pub fn OnlineWorkerLease::progress_maintenance(Self) -> LunaWorkerServiceMaintenanceProgress raise WorkerServiceError' \
  'pub fn OnlineWorkerLease::maintenance_remaining_millis(Self) -> Int raise WorkerServiceError' \
  'pub fn WorkerRestartBackoffPolicy::new(initial_delay_millis~ : UInt64, maximum_delay_millis~ : UInt64, stable_success_millis~ : UInt64, maximum_attempts~ : Int) -> Self raise WorkerServiceError' \
  'pub fn OnlineWorkerLease::progress_restart_backoff(Self) -> OnlineWorkerRestartProgress raise WorkerServiceError' \
  'pub fn OnlineWorkerLease::restart_backoff_remaining_millis(Self) -> Int raise WorkerServiceError'; do
  if ! rg -Fq "$signature" "$worker_service_interface"; then
    printf 'worker-service maintenance interface drifted: %s\n' "$signature" >&2
    exit 1
  fi
done
if rg -q '^pub fn OnlineWorkerLease::restart\(' "$worker_service_interface" ||
  rg -q '^pub fn OnlineWorkerLease::restart\(' \
    engine/worker_service/*.mbt; then
  printf '%s\n' \
    'online lease exposes a replacement bypass outside bounded backoff' >&2
  exit 1
fi
maintenance_interface="$(awk '
  /pub\(all\) enum LunaWorkerServiceMaintenanceProgress/ { copying = 1 }
  copying { print }
  copying && /^}/ { exit }
' "$worker_service_interface")"
for variant in \
  LunaWorkerServiceMaintenancePending \
  LunaWorkerServiceMaintenanceAdvanced \
  LunaWorkerServiceMaintenanceCleanupRequired \
  LunaWorkerServiceMaintenanceCleanupStuck \
  LunaWorkerServiceRecoveryReady \
  LunaWorkerServiceFailCloseReady \
  LunaWorkerServiceRestartReady \
  LunaWorkerServiceClosed; do
  if ! printf '%s\n' "$maintenance_interface" | rg -q "^  ${variant}$"; then
    printf 'worker-service maintenance result drifted: %s\n' "$variant" >&2
    exit 1
  fi
done
if printf '%s\n' "$maintenance_interface" | rg -q \
  'RootBound|WorkerProcess|SchedulePlan|RequestHandle|WorkerProcessExit'; then
  printf '%s\n' 'worker-service maintenance result leaks lower authority' >&2
  exit 1
fi

if printf '%s\n' "$hot_body" |
  rg -q "$collection_allocation_pattern"; then
  printf '%s\n' 'worker-service progress constructs a collection, ref, or string' >&2
  exit 1
fi

maintenance_body=""
for symbol in \
  'WorkerService21progress__maintenance(' \
  'WorkerService27progress__maintenance__impl(' \
  'WorkerService28progress__child__maintenance(' \
  'WorkerService35apply__child__maintenance__progress(' \
  'WorkerService30progress__recovery__retirement(' \
  'WorkerService32progress__recovery__invalidation(' \
  'WorkerService25progress__recovery__drain(' \
  'WorkerService32progress__begin__terminal__close(' \
  'WorkerService36maintenance__remaining__millis__impl(' \
  'OnlineWorkerLease21progress__maintenance(' \
  'OnlineWorkerLease30maintenance__remaining__millis('; do
  body="$(extract_definition "$whitebox_c" "$symbol")"
  if [ -z "$body" ]; then
    printf 'worker-service maintenance allocation function is missing: %s\n' \
      "$symbol" >&2
    exit 1
  fi
  maintenance_body="${maintenance_body}${body}"
done
if printf '%s\n' "$maintenance_body" |
  rg -q "$collection_allocation_pattern"; then
  printf '%s\n' 'worker-service cooperative maintenance constructs storage' >&2
  exit 1
fi
if printf '%s\n' "$maintenance_body" |
  rg 'moonbit_malloc' |
  rg -v "$worker_service_error_allocation_pattern" |
  rg -v "$scheduler_error_allocation_pattern" |
  rg -q .; then
  printf '%s\n' \
    'worker-service cooperative maintenance contains a non-error allocation' >&2
  exit 1
fi

maintenance_source="$(sed '/^[[:space:]]*\/\//d' \
  engine/worker_service/maintenance.mbt)"
if printf '%s\n' "$maintenance_source" | rg -q \
    '\b(for|while|loop)\b|sleep|wait|shutdown_clean\(|\.close\(' ||
  ! printf '%s\n' "$maintenance_source" | rg -q \
    'require_transport\(\)\.progress_maintenance\(\)' ||
  ! printf '%s\n' "$maintenance_source" | rg -q \
    'recover_physical_flight_impl|recover_pending_failure' ||
  ! printf '%s\n' "$maintenance_source" | rg -q \
    'invalidate_device_state_impl|drain_restart_forbidden_impl'; then
  printf '%s\n' \
    'worker-service cooperative maintenance source boundary drifted' >&2
  exit 1
fi
if ! printf '%s\n' "$maintenance_source" | rg -q \
    'begin_restart_cleanup_maintenance_impl' ||
  ! printf '%s\n' "$maintenance_source" | rg -q \
    'begin_replacement_cleanup_maintenance\(\)' ||
  printf '%s\n' "$maintenance_source" | rg -q \
    'retry_restart_cleanup|retry_replacement_cleanup'; then
  printf '%s\n' \
    'worker-service cooperative replacement cleanup boundary drifted' >&2
  exit 1
fi
if ! printf '%s\n' "$worker_service_source" | rg -q \
    'maintenance_mode : Int' ||
  ! printf '%s\n' "$worker_service_source" | rg -q \
    'maintenance_phase : Int' ||
  printf '%s\n' "$worker_service_source" | rg -q \
    'maintenance[^\n]*:.*\?|Option\[[^]]*Maintenance'; then
  printf '%s\n' 'worker-service maintenance owner is not scalar' >&2
  exit 1
fi

restart_source="$(sed '/^[[:space:]]*\/\//d' \
  engine/worker_service/restart_backoff.mbt)"
if printf '%s\n' "$restart_source" | rg -q \
    '\b(sleep|wait|loop|while|for)\b' ||
  ! printf '%s\n' "$restart_source" | rg -q \
    'restart_checked_deadline\(now, delay\)' ||
  ! printf '%s\n' "$restart_source" | rg -q \
    'restart_attempts >= self\.restart_policy\.maximum_attempts' ||
  ! printf '%s\n' "$restart_source" | rg -q \
    'self\.restart_generation == 0xffffffffffffffffUL' ||
  ! printf '%s\n' "$restart_source" | rg -q \
    'now < self\.restart_last_clock_millis'; then
  printf '%s\n' 'worker restart backoff lost its bounded fail-close policy' >&2
  exit 1
fi
restart_observation_source="$(awk '
  /pub fn OnlineWorkerLease::restart_backoff_remaining_millis\(/ {
    copying = 1
  }
  copying {
    print
    opens += gsub(/\{/, "{"); closes += gsub(/\}/, "}")
    if (opens > 0 && opens == closes) exit
  }
' engine/worker_service/restart_backoff.mbt)"
if [ -z "$restart_observation_source" ] ||
  ! printf '%s\n' "$restart_observation_source" | rg -q \
    'restart_cached_remaining_millis' ||
  printf '%s\n' "$restart_observation_source" | rg -q \
    'clock|sample_restart_clock|observe_restart_clock|remaining_impl|remaining_at|restart_phase[[:space:]]*=|restart_forbidden[[:space:]]*='; then
  printf '%s\n' \
    'worker restart wake accessor stopped being observational' >&2
  exit 1
fi
if ! rg -U -q \
  'ServiceCommitted\(sequence\)[\s\S]*record_stable_replacement_commit\(service, sequence\)' \
  engine/worker_service/online_lease.mbt ||
  ! rg -U -q \
  'record_stable_replacement_commit_at[\s\S]*restart_stability_generation != self\.restart_generation[\s\S]*sequence <= self\.restart_stability_predecessor[\s\S]*last_retired != sequence[\s\S]*now < self\.restart_stability_deadline_millis' \
  engine/worker_service/restart_backoff.mbt; then
  printf '%s\n' \
    'stable restart reset lost current-flight sequence authentication' >&2
  exit 1
fi
if printf '%s\n' "$hot_body" |
  rg 'moonbit_malloc' |
  rg -v "$worker_service_error_allocation_pattern" |
  rg -v "$scheduler_error_allocation_pattern" |
  rg -q .; then
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
