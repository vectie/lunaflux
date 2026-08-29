#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

if rg -n 'system\s*\(|popen\s*\(|posix_spawnp\s*\(' \
  internal/process --glob '*.c'; then
  printf '%s\n' 'worker process ABI must not invoke a shell or PATH lookup' >&2
  exit 1
fi

if rg -n 'fork\s*\(' internal/process --glob '*.c' \
    --glob '!process_approved_spawn.c' ||
  ! rg -q '^#if defined\(__linux__\)$' \
    internal/process/process_approved_spawn.c ||
  ! rg -q 'fexecve\(5, argv, sanitized_environment\)' \
    internal/process/process_approved_spawn.c ||
  ! rg -q 'syscall\(SYS_close_range, 6u, UINT_MAX, 0u\)' \
    internal/process/process_approved_spawn.c ||
  ! rg -U -q 'pthread_sigmask\(SIG_BLOCK,[\s\S]*fork\(\)' \
    internal/process/process_approved_spawn.c ||
  ! rg -q 'signal_number < NSIG' internal/process/process_approved_spawn.c ||
  ! rg -U -q 'sigprocmask\(SIG_SETMASK, &empty_mask, NULL\)[\s\S]*fexecve' \
    internal/process/process_approved_spawn.c ||
  ! rg -q 'return ENOTSUP' internal/process/process_approved_spawn.c; then
  printf '%s\n' \
    'approved executable spawn must be the sole Linux-only fork/fexecve seam' >&2
  exit 1
fi

if ! rg -q '#include "process_status.h"' internal/process/process.c ||
  ! rg -q '#include "process_status.h"' internal/process/child_control.c ||
  ! rg -q 'lunaflux_process_inherited_expect_clean_eof' \
    internal/process/child_control.c ||
  ! rg -q 'lunaflux_process_inherited_read_prefix_or_eof' \
    internal/process/child_control.c ||
  ! rg -q '_exit\(1\)' internal/process/child_control.c; then
  printf '%s\n' 'worker child control is missing shared status or exact exit/EOF primitives' >&2
  exit 1
fi

if rg -q 'posix_spawn\s*\(|execve\s*\(' internal/process/process.c ||
  ! rg -q 'socketpair\s*\(' internal/process/process.c ||
  ! rg -q 'CLOCK_MONOTONIC' internal/process/process_io.c; then
  printf '%s\n' 'worker process ABI is missing exact spawn, private channel, or monotonic timeout' >&2
  exit 1
fi

if ! rg -q 'lunaflux_process_prepare_child' internal/process/process.c ||
  ! rg -q 'lunaflux_process_spawn_prepared_with_approved_executable_and_roots' \
    internal/process/process.c ||
  ! rg -q 'lf_approved_executable_duplicate' internal/process/process.c ||
  ! rg -q 'lunaflux_process_is_closed' internal/process/process.c ||
  rg -q 'raw_spawn_with_approved_roots|raw_spawn\(' internal/process/ffi.mbt; then
  printf '%s\n' 'worker process ABI must use prepared child spawn only' >&2
  exit 1
fi

if ! rg -U -q \
    'raw_spawn_prepared_with_approved_executable_and_roots\([\s\S]*@inheritance\.WorkerApprovedRootSpawnAuthority' \
    internal/process/ffi.mbt ||
  rg -q 'raw_spawn_prepared\(|raw_spawn_prepared_with_approved_roots' \
    internal/process/ffi.mbt ||
  rg -n 'WorkerApprovedRoots|PreparedWorkerApprovedRoots' \
    internal/process --glob '*.mbt'; then
  printf '%s\n' \
    'worker process spawn must accept only the capability-limited root view' >&2
  exit 1
fi

if rg -n 'process_path_spawn|spawn_prepared_with_approved_roots|spawn_prepared\(' \
    internal/process --glob '*.c' --glob '*.h' --glob '*.mbt'; then
  printf '%s\n' 'production process ABI must expose no pathname spawn route' >&2
  exit 1
fi

if ! rg -q 'MSG_DONTWAIT' internal/process/process_io.c ||
  ! rg -q 'LF_PROCESS_PENDING' internal/process/process_io.c ||
  ! rg -q 'error_number == EINTR' internal/process/process_io.c ||
  ! rg -q 'lunaflux_process_try_write' internal/process/process_nonblocking.c ||
  ! rg -q 'lunaflux_process_try_read' internal/process/process_nonblocking.c ||
  ! rg -q 'lunaflux_process_inherited_try_write' \
    internal/process/process_nonblocking.c ||
  ! rg -q 'lunaflux_process_inherited_try_read' \
    internal/process/process_nonblocking.c ||
  ! rg -q 'Moonbit_array_length\(bytes\)' internal/process/process_nonblocking.c; then
  printf '%s\n' 'worker process ABI is missing one-attempt nonblocking I/O semantics' >&2
  exit 1
fi

if ! rg -q 'lunaflux_process_inherited_wait' \
    internal/process/inherited_wait.c ||
  ! rg -q 'errno == EINTR' internal/process/inherited_wait.c ||
  ! rg -q 'POLLIN' internal/process/inherited_wait.c ||
  ! rg -q 'POLLOUT' internal/process/inherited_wait.c ||
  ! rg -q 'raw_inherited_wait' \
    internal/process/inherited_wait_ffi.mbt ||
  [ "$(rg -c 'extern\s+"[cC]"' \
      internal/process/inherited_wait_ffi.mbt)" -ne 1 ]; then
  printf '%s\n' 'inherited wait ABI must remain one exact EINTR-safe poll seam' >&2
  exit 1
fi

if rg -q 'PendingFrame(Read|Write)' internal/process/types.mbt ||
  rg -q 'transfer_kind|transfer_primary|transfer_secondary' \
    internal/process --glob '*.mbt' ||
  ! rg -q 'priv mut write_transfer_active : Bool' \
    internal/process/types.mbt ||
  ! rg -q 'priv mut read_transfer_active : Bool' \
    internal/process/types.mbt ||
  ! rg -q 'pub fn ChildProcess::begin_write_frame' \
    internal/process/pending_frame.mbt ||
  ! rg -q 'pub fn ChildProcess::progress_write_frame' \
    internal/process/pending_frame.mbt ||
  ! rg -q 'pub fn ChildProcess::begin_read_frame' \
    internal/process/pending_frame.mbt ||
  ! rg -q 'pub fn ChildProcess::progress_read_frame' \
    internal/process/pending_frame.mbt; then
  printf '%s\n' \
    'pending frame state must remain owner-resident without public tokens' >&2
  exit 1
fi

if ! rg -q 'pub fn InheritedChannel::begin_write_frame' \
    internal/process/inherited_pending_frame.mbt ||
  ! rg -q 'pub fn InheritedChannel::progress_write_frame' \
    internal/process/inherited_pending_frame.mbt ||
  ! rg -q 'pub fn InheritedChannel::begin_read_frame' \
    internal/process/inherited_pending_frame.mbt ||
  ! rg -q 'pub fn InheritedChannel::progress_read_frame' \
    internal/process/inherited_pending_frame.mbt ||
  ! rg -q 'pub fn InheritedChannel::begin_read_frame_or_eof' \
    internal/process/inherited_pending_frame.mbt ||
  ! rg -q 'pub fn InheritedChannel::progress_read_frame_or_eof' \
    internal/process/inherited_pending_frame.mbt; then
  printf '%s\n' \
    'worker-side cooperative duplex framing APIs are incomplete' >&2
  exit 1
fi

if ! rg -q 'fn ChildProcess::write_source_conflicts_active_read' \
    internal/process/duplex_alias.mbt ||
  ! rg -q 'fn ChildProcess::read_buffer_conflicts_active_write' \
    internal/process/duplex_alias.mbt ||
  ! rg -q 'fn InheritedChannel::write_source_conflicts_active_read' \
    internal/process/duplex_alias.mbt ||
  ! rg -q 'fn InheritedChannel::read_buffer_conflicts_active_write' \
    internal/process/duplex_alias.mbt ||
  ! rg -q 'write_source_conflicts_active_read' \
    internal/process/pending_frame.mbt ||
  ! rg -q 'read_buffer_conflicts_active_write' \
    internal/process/pending_frame.mbt ||
  ! rg -q 'write_source_conflicts_active_read' \
    internal/process/inherited_pending_frame.mbt ||
  ! rg -q 'read_buffer_conflicts_active_write' \
    internal/process/inherited_pending_frame.mbt ||
  ! rg -q 'write_source_conflicts_active_read' \
    internal/process/process.mbt ||
  ! rg -q 'read_buffer_conflicts_active_write' \
    internal/process/process.mbt ||
  ! rg -q 'write_source_conflicts_active_read' \
    internal/process/process_spawn.mbt ||
  ! rg -q 'read_buffer_conflicts_active_write' \
    internal/process/process_spawn.mbt; then
  printf '%s\n' \
    'duplex fixed-buffer identity admission is incomplete' >&2
  exit 1
fi

if ! rg -q 'fn ChildProcess::admit_frame_deadline' \
    internal/process/pending_frame.mbt ||
  ! rg -q 'pub fn ChildProcess::begin_write_frame_until' \
    internal/process/pending_frame.mbt ||
  ! rg -q 'pub fn ChildProcess::begin_read_frame_until' \
    internal/process/pending_frame.mbt; then
  printf '%s\n' 'absolute monotonic child frame-deadline seam is incomplete' >&2
  exit 1
fi

if ! rg -q 'clock_gettime\(CLOCK_MONOTONIC' \
    internal/process/process_io.c ||
  ! rg -q 'LF_MONOTONIC_CLOCK_GETTIME\(CLOCK_MONOTONIC' \
    internal/monotonic_clock/monotonic_clock.c; then
  printf '%s\n' 'process and caller deadline clocks no longer share one domain' >&2
  exit 1
fi

for source_file in internal/process/process.c \
  internal/process/process_approved_spawn.c \
  internal/process/process_approved_spawn.h \
  internal/process/process_io.c \
  internal/process/process_io_asan_probe.c \
  internal/process/process_io.h \
  internal/process/process_nonblocking.c \
  internal/process/inherited_wait.c \
  internal/process/inherited_wait_asan_probe.c \
  internal/process/process_spawn_asan_probe.c \
  internal/process/process_handle.h \
  internal/process/child_control.c \
  internal/process/child_control_asan_probe.c \
  internal/process/process_status.h; do
  line_count=$(wc -l < "$source_file" | tr -d ' ')
  if [ "$line_count" -gt 500 ]; then
    printf '%s: %s lines; process ABI files must stay below 500\n' \
      "$source_file" "$line_count" >&2
    exit 1
  fi
done

if rg -n 'extern\s+"[cC]"' internal/process --glob '*.mbt' \
  --glob '!ffi.mbt' --glob '!inherited_wait_ffi.mbt' \
  --glob '!*_wbtest.mbt'; then
  printf '%s\n' \
    'worker process extern declarations must remain in approved FFI files' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux worker process ABI boundary is valid.'
