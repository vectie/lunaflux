#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

package="engine/worker_service"
transport="$package/physical_transport.mbt"

if [ ! -f "$transport" ]; then
  printf '%s\n' 'worker-service physical transport adapter is missing' >&2
  exit 1
fi

if ! rg -q '^priv enum WorkerPhysicalTransport \{' "$transport" ||
  rg -q '^pub.*(enum|struct) WorkerPhysicalTransport \{|^pub fn WorkerPhysicalTransport::' \
    "$transport" "$package/types.mbt"; then
  printf '%s\n' 'worker-service physical transport is not package-private' >&2
  exit 1
fi
if rg -n 'RootBoundProcess\(' \
    "$package/types.mbt" "$package/online_lease.mbt" \
    "$package/restart_backoff.mbt" >/dev/null ||
  ! rg -q 'PhysicalTransport\(WorkerPhysicalTransportFailure\)' \
    "$package/types.mbt"; then
  printf '%s\n' 'worker-service runtime error surface leaks a backend' >&2
  exit 1
fi
if rg -n 'OwnedProcessFailure|OwnedWorkerServiceFailure[^{]*\{[^}]*RootBound' \
    "$package/types.mbt" >/dev/null ||
  ! rg -q 'OwnedPhysicalTransportFailure\(WorkerPhysicalTransportFailure\)' \
    "$package/types.mbt"; then
  printf '%s\n' 'owned service failure publication leaks a backend' >&2
  exit 1
fi

# Scheduler, request, KV, and online owners must never select a physical
# topology. The single implementation-specific owner is confined to its
# adapter plus startup/cleanup publication shells.
for file in \
  "$package/progress.mbt" \
  "$package/pending_plan.mbt" \
  "$package/recovery.mbt" \
  "$package/maintenance.mbt" \
  "$package/online_lease.mbt" \
  "$package/online_recovery.mbt" \
  "$package/online_close.mbt" \
  "$package/restart_backoff.mbt"; do
  if rg -n '@worker_process|RootBoundWorker|@rank_group_process|RankGroupProcess' \
      "$file" >/dev/null; then
    printf 'worker-service physical branch escaped adapter: %s\n' "$file" >&2
    exit 1
  fi
done

if rg -n 'rank_group|tensor_parallel|world_size|device_ordinal' \
    scheduler/core --glob '*.mbt' --glob 'moon.pkg' >/dev/null; then
  printf '%s\n' 'scheduler contains a physical topology branch' >&2
  exit 1
fi

# A warmed flight retains only scalar sequence/state. Received and accepted
# process capabilities are rederived inside the transport adapter.
if ! rg -q 'FlightAccepted\(UInt64\)' "$package/types.mbt" ||
  rg -n \
    'Option\[.*(Received|Accepted|Submitted).*(Completion|Plan|Submission)|priv mut .*:.*(Received|Accepted|Submitted).*(Completion|Plan|Submission)\?' \
    "$package"/*.mbt >/dev/null; then
  printf '%s\n' 'worker-service retained a warmed transport capability' >&2
  exit 1
fi
for method in \
  received_frame \
  accept_completion \
  retire_accepted \
  retire_received_failure \
  abandon_submission; do
  if ! rg -q "fn WorkerPhysicalTransport::${method}\\(" "$transport"; then
    printf 'worker-service scalar reauthentication method is missing: %s\n' \
      "$method" >&2
    exit 1
  fi
done
for category in \
  PhysicalBackpressured \
  PhysicalInvalidOutput \
  PhysicalUnavailable \
  PhysicalCleanupFailed \
  PhysicalLifecycleViolation; do
  if ! rg -q "$category" "$package/physical_transport_wbtest.mbt"; then
    printf 'worker-service physical failure test is missing: %s\n' \
      "$category" >&2
    exit 1
  fi
done

# A future tensor-parallel variant must derive its execution deadline from its
# own immutable transport policy and monotonic clock. Request deadline values
# cannot enter this adapter.
if rg -n \
    'TokenizedRequest|GenerateRequest|RequestDeadline|deadline_millis\(\)|receipt|admission_deadline' \
    "$package"/physical_transport*.mbt >/dev/null; then
  printf '%s\n' 'worker-service transport depends on a request deadline' >&2
  exit 1
fi

# Terminal cleanup is a cold path, but its lifecycle transitions still need to
# be allocation-independent and explicit. Captured defer environments obscure
# which failure owns the retry state and previously allocated compiler closure
# frames. Keep recovery failures direct and typed.
if rg -n '\bdefer\b' "$package/recovery.mbt" >/dev/null; then
  printf '%s\n' \
    'worker-service recovery cleanup contains a captured defer environment' >&2
  exit 1
fi
if ! rg -U -q \
    'shutdown_clean\(\) catch \{[[:space:]]*error => \{[[:space:]]*self\.lifecycle = ServiceRecovering[[:space:]]*raise error' \
    "$package/recovery.mbt" ||
  ! rg -U -q \
    'close\(\) catch \{[[:space:]]*error => \{[[:space:]]*self\.lifecycle = ServiceClosingFailed[[:space:]]*raise error' \
    "$package/recovery.mbt"; then
  printf '%s\n' \
    'worker-service recovery cleanup lost its explicit retry lifecycle' >&2
  exit 1
fi

if rg -q '^pub fn RootBoundWorkerProcessSupervisor::recovery_startup_contract' \
    engine/worker_process/pkg.generated.mbti ||
  rg -q 'recovery_startup_contract' \
    "$package/physical_transport_lifecycle.mbt" ||
  ! rg -q '^pub fn RootBoundWorkerProcessSupervisor::validate_recovery_binding' \
    engine/worker_process/pkg.generated.mbti; then
  printf '%s\n' \
    'generic service extracts a concrete single-worker recovery contract' >&2
  exit 1
fi

for file in "$package"/*.mbt; do
  if [ "$(wc -l < "$file")" -gt 500 ]; then
    printf '%s exceeds 500 lines\n' "$file" >&2
    exit 1
  fi
done

printf '%s\n' 'LunaFlux worker-service physical transport boundary passed.'
