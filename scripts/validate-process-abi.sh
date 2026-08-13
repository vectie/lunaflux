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
  ! rg -q '_exit\(1\)' internal/process/child_control.c; then
  printf '%s\n' 'worker child control is missing shared status or exact exit/EOF primitives' >&2
  exit 1
fi

if ! rg -q 'posix_spawn\s*\(' internal/process/process.c ||
  ! rg -q 'socketpair\s*\(' internal/process/process.c ||
  ! rg -q 'CLOCK_MONOTONIC' internal/process/process.c; then
  printf '%s\n' 'worker process ABI is missing exact spawn, private channel, or monotonic timeout' >&2
  exit 1
fi

for source_file in internal/process/process.c \
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
