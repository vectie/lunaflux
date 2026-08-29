#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test engine/worker_process \
  --target native --release --deny-warn --warn-list +73
moon test internal/process \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/engine/worker_process/worker_process.whitebox_test.c"
process_c="_build/native/release/test/internal/process/process.whitebox_test.c"
interface="engine/worker_process/pkg.generated.mbti"
if [ ! -f "$generated_c" ] || [ ! -f "$process_c" ] || [ ! -f "$interface" ]; then
  printf '%s\n' 'worker-process maintenance evidence is missing' >&2
  exit 1
fi

extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 &&
      ($0 ~ /^struct moonbit_result_/ || $0 ~ /^struct _M0TP/ ||
       $0 ~ /^int32_t / || $0 ~ /^void\*/) {
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

hot_body=""
for symbol in \
  'begin__shutdown__maintenance(' \
  'progress__shutdown__maintenance(' \
  'apply__maintenance__progress(' \
  'shutdown__maintenance__remaining__millis(' \
  'shutdown__maintenance__exit(' \
  'begin__child__maintenance(' \
  'begin__recovery__maintenance(' \
  'begin__close__maintenance(' \
  'progress__cleanup__maintenance(' \
  'progress__maintenance(' \
  'record__child__maintenance__failure(' \
  '32apply__retained__child__progress(' \
  'apply__root__close__progress(' \
  'maintenance__remaining__millis(' \
  'maintenance__exit('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'worker maintenance allocation function is missing: %s\n' \
      "$symbol" >&2
    exit 1
  fi
  hot_body="${hot_body}${body}"
done

if printf '%s\n' "$hot_body" |
  rg -q 'moonbit_make_.*array|moonbit_make_bytes|moonbit_make_ref|moonbit_add_string|moonbit_array_copy|memcpy|memmove'; then
  printf '%s\n' 'worker maintenance constructs or copies managed storage' >&2
  exit 1
fi
if printf '%s\n' "$hot_body" |
  rg 'moonbit_malloc' |
  rg -F -v 'moonbit_malloc(sizeof(struct _M0DTPC15error5Error75vectie_2flunaflux_2fengine_2fworker__process_2eWorkerProcessError_2eInvalid))' |
  rg -F -v 'moonbit_malloc(sizeof(struct _M0DTPC15error5Error77vectie_2flunaflux_2fengine_2fworker__process_2eWorkerProcessError_2eTransport))' |
  rg -F -v 'moonbit_malloc(sizeof(struct _M0DTP46vectie8lunaflux6engine15worker__process29RootBoundWorkerProcessFailure7Process))' |
  rg -F -v 'moonbit_malloc(sizeof(struct _M0DTPC15error5Error93vectie_2flunaflux_2fengine_2fworker__process_2eRootBoundWorkerProcessError_2eRootBoundFailure))' |
  rg -q .; then
  printf '%s\n' 'worker maintenance contains a non-error heap allocation' >&2
  exit 1
fi
if ! rg -q 'moonbit_make_bytes\(13, 90\)' "$process_c"; then
  printf '%s\n' 'worker maintenance allocation positive control is ineffective' >&2
  exit 1
fi

for signature in \
  'pub fn WorkerProcessSupervisor::begin_shutdown_maintenance(Self) -> Unit raise WorkerProcessError' \
  'pub fn WorkerProcessSupervisor::progress_shutdown_maintenance(Self) -> WorkerProcessMaintenanceProgress raise WorkerProcessError' \
  'pub fn WorkerProcessSupervisor::shutdown_maintenance_remaining_millis(Self) -> Int raise WorkerProcessError' \
  'pub fn WorkerProcessSupervisor::shutdown_maintenance_exit(Self) -> WorkerProcessExit raise WorkerProcessError' \
  'pub fn WorkerProcessMaintenanceProgress::is_cleanup_stuck(Self) -> Bool' \
  'pub fn RootBoundWorkerProcessSupervisor::begin_recovery_maintenance(Self) -> Unit raise RootBoundWorkerProcessError' \
  'pub fn RootBoundWorkerProcessSupervisor::begin_close_maintenance(Self) -> Unit raise RootBoundWorkerProcessError' \
  'pub fn RootBoundWorkerProcessSupervisor::begin_replacement_cleanup_maintenance(Self) -> Unit raise RootBoundWorkerProcessError' \
  'pub fn RootBoundWorkerProcessSupervisor::progress_maintenance(Self) -> WorkerProcessMaintenanceProgress raise RootBoundWorkerProcessError' \
  'pub fn RootBoundWorkerProcessSupervisor::maintenance_remaining_millis(Self) -> Int raise RootBoundWorkerProcessError' \
  'pub fn RootBoundWorkerProcessSupervisor::maintenance_exit(Self) -> WorkerProcessExit raise RootBoundWorkerProcessError'; do
  if ! grep -Fqx "$signature" "$interface"; then
    printf 'worker maintenance interface drift: %s\n' "$signature" >&2
    exit 1
  fi
done

for opaque_type in WorkerProcessMaintenanceProgress WorkerProcessExit; do
  if [ "$(grep -Fxc "pub struct ${opaque_type} {" "$interface")" -ne 1 ] ||
    ! awk -v type="$opaque_type" '
      $0 == "pub struct " type " {" { seen = 1; next }
      seen && $0 == "  // private fields" { private_fields = 1; next }
      seen && $0 == "}" { exit !(private_fields == 1) }
    ' "$interface"; then
    printf 'worker maintenance type is not private-field opaque: %s\n' \
      "$opaque_type" >&2
    exit 1
  fi
  if rg -q "impl Debug for ${opaque_type}" "$interface"; then
    printf 'worker maintenance type exposes Debug: %s\n' "$opaque_type" >&2
    exit 1
  fi
done

if rg -n '^pub fn .*maintenance.*(@process|FixedArray|ApprovedRoot|WorkerApprovedRoots)' \
    "$interface" ||
  rg -n '^pub (struct|enum) .*Maintenance(Token|Lease|View|Handle)' \
    "$interface"; then
  printf '%s\n' 'worker maintenance leaks internal authority' >&2
  exit 1
fi

if ! rg -Uq 'self\.clear_exchange\(\)\n  self\.process\.begin_shutdown_maintenance\(\)' \
    engine/worker_process/root_bound_maintenance.mbt ||
  ! rg -Uq 'self\.maintenance_mode = 3\n  worker_maintenance_advanced' \
    engine/worker_process/root_bound_maintenance.mbt ||
  ! rg -Uq 'progress\.is_cleanup_stuck\(\) \{\n    return worker_maintenance_cleanup_stuck' \
    engine/worker_process/root_bound_maintenance.mbt ||
  ! rg -Uq 'let status = close_root_pair_status\(self\.require_roots\(\)\)' \
    engine/worker_process/root_bound_maintenance.mbt; then
  printf '%s\n' 'root maintenance child-before-root ordering drifted' >&2
  exit 1
fi

if ! rg -q 'failed\.begin_cleanup_maintenance\(\)' \
    engine/worker_process/root_bound_maintenance.mbt ||
  ! rg -q 'self\.child\.begin_shutdown_maintenance\(\)' \
    engine/worker_process/types.mbt ||
  rg -n 'Array::|FixedArray::|Bytes::|String::|@process\.ChildProcess' \
    engine/worker_process/root_bound_maintenance.mbt |
    rg -q 'begin_replacement_cleanup_maintenance'; then
  printf '%s\n' 'replacement cleanup start boundary drifted' >&2
  exit 1
fi

printf '%s\n' \
  'LunaFlux worker-process cooperative maintenance allocation/source gate passed.'
