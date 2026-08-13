#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

if rg -n 'system\s*\(|popen\s*\(|fork\s*\(|posix_spawnp\s*\(' \
  internal/process --glob '*.c'; then
  printf '%s\n' 'worker process ABI must not invoke a shell, fork, or PATH lookup' >&2
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

if ! rg -q 'posix_spawn\s*\(' internal/process/process.c ||
  ! rg -q 'socketpair\s*\(' internal/process/process.c ||
  ! rg -q 'CLOCK_MONOTONIC' internal/process/process_io.c; then
  printf '%s\n' 'worker process ABI is missing exact spawn, private channel, or monotonic timeout' >&2
  exit 1
fi

if ! rg -q 'lunaflux_process_prepare_child' internal/process/process.c ||
  ! rg -q 'lunaflux_process_spawn_prepared_with_approved_roots' \
    internal/process/process.c ||
  ! rg -q 'lunaflux_process_is_closed' internal/process/process.c ||
  rg -q 'raw_spawn_with_approved_roots|raw_spawn\(' internal/process/ffi.mbt; then
  printf '%s\n' 'worker process ABI must use prepared child spawn only' >&2
  exit 1
fi

if ! rg -q 'path_length <= 1.*path_length > 1048576' internal/process/process.c ||
  ! rg -q "path\[path_length - 1\] != '\\\\0'" internal/process/process.c ||
  ! rg -q "memchr\(path, '\\\\0', \(size_t\)\(path_length - 1\)\)" \
    internal/process/process.c; then
  printf '%s\n' 'prepared executable path length/terminator guards are missing' >&2
  exit 1
fi

if ! rg -q 'MSG_DONTWAIT' internal/process/process_io.c ||
  ! rg -q 'LF_PROCESS_PENDING' internal/process/process_io.c ||
  ! rg -q 'error_number == EINTR' internal/process/process_io.c ||
  ! rg -q 'lunaflux_process_try_write' internal/process/process_nonblocking.c ||
  ! rg -q 'lunaflux_process_try_read' internal/process/process_nonblocking.c ||
  ! rg -q 'Moonbit_array_length\(bytes\)' internal/process/process_nonblocking.c; then
  printf '%s\n' 'worker process ABI is missing one-attempt nonblocking I/O semantics' >&2
  exit 1
fi

if rg -q 'PendingFrame(Read|Write)' internal/process/types.mbt ||
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

for source_file in internal/process/process.c \
  internal/process/process_io.c \
  internal/process/process_io_asan_probe.c \
  internal/process/process_io.h \
  internal/process/process_nonblocking.c \
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
  --glob '!ffi.mbt' --glob '!*_wbtest.mbt'; then
  printf '%s\n' 'worker process extern declarations must remain in ffi.mbt' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux worker process ABI boundary is valid.'
